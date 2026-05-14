//! Custom OLED renderers for the Corne split keyboard.
//!
//! This module is read-only with respect to keyboard state: every drop of
//! information we display (layer, WPM, modifiers, caps lock, battery, BLE
//! status, sleep state) comes from RMK via the [`RenderContext`] passed to
//! [`DisplayRenderer::render`]. We never observe the keymap or vial.json,
//! and never inject keys — only paint pixels.
//!
//! Hardware layout (per user spec):
//!   * Both halves: SSD1306 128x32 OLED, mounted vertically.
//!   * Driver runs at `DisplayRotation::Rotate90` (configured in keyboard.toml),
//!     so the logical framebuffer handed to us is `32 wide × 128 tall`.
//!   * Left half is the BLE/USB central; right half is the peripheral.
//!     WPM, layer, modifiers, caps lock and sleep state are sampled on the
//!     central and synced to the peripheral by RMK over the split BLE link
//!     (see `SplitMessage::{Wpm, Modifier, SleepState, KeyboardIndicator}`
//!     in rmk/src/split/mod.rs), so both renderers can just read them off
//!     `RenderContext`.
//!
//! Per the user's spec:
//!   * Left screen: battery bar + % at the top in normal orientation; below
//!     it, two columns of "rotated" text (read bottom-to-top after a 90° CCW
//!     head tilt) showing the layer name and a longer description.
//!   * Right screen: fully rotated, two columns reading bottom-to-top —
//!     left column has active modifiers + WPM, right column has BT/USB
//!     status, optional CAPS indicator, and a small WPM sparkline.
//!   * After 30 s with no key press the screen "breathes" (content stays,
//!     pixels dither). Once RMK reports `sleeping = true` (deep sleep) we
//!     blank the panel entirely.

use core::fmt::Write as _;

use embassy_time::Instant;
use embedded_graphics::{
    geometry::{OriginDimensions, Point, Size},
    mono_font::{
        MonoTextStyle,
        ascii::{FONT_5X8, FONT_6X10, FONT_8X13_BOLD},
    },
    pixelcolor::BinaryColor,
    prelude::*,
    primitives::{PrimitiveStyle, PrimitiveStyleBuilder, Rectangle},
    text::{Baseline, Text},
};
use heapless::String;
use rmk::display::{DisplayRenderer, RenderContext};
use rmk::types::battery::BatteryStatus;
// `nrf52840_ble` (enabled in our Cargo.toml) transitively turns on rmk's
// `_ble` feature, so `ctx.ble_status` is always present and `BleState` is
// always reachable. We don't need a `#[cfg]` guard here.
use rmk::types::ble::BleState;

// ─── Geometry ────────────────────────────────────────────────────────────────

/// Logical framebuffer width (DisplayRotation::Rotate90 of a 128×32 panel).
const SCREEN_W: i32 = 32;
/// Logical framebuffer height.
const SCREEN_H: i32 = 128;

/// Y boundary between the upright "battery" zone (above) and the rotated
/// "layer" zone (below) on the left screen.
const LEFT_BATTERY_ZONE_H: i32 = 34;

// ─── Styling ─────────────────────────────────────────────────────────────────

const STROKE: PrimitiveStyle<BinaryColor> = PrimitiveStyle::with_stroke(BinaryColor::On, 1);
const FILL: PrimitiveStyle<BinaryColor> = PrimitiveStyle::with_fill(BinaryColor::On);

// `'static` (not `'_`) because const item lifetimes must be explicit.
const FONT_SMALL: MonoTextStyle<'static, BinaryColor> = MonoTextStyle::new(&FONT_5X8, BinaryColor::On);
const FONT_MED: MonoTextStyle<'static, BinaryColor> = MonoTextStyle::new(&FONT_6X10, BinaryColor::On);
const FONT_BIG: MonoTextStyle<'static, BinaryColor> = MonoTextStyle::new(&FONT_8X13_BOLD, BinaryColor::On);

// ─── Activity / sleep tracker ────────────────────────────────────────────────

/// Tracks the last time a key press was observed so we can drive a "breathing"
/// effect after 30 s of inactivity. Distinct from RMK's own deep-sleep state
/// (`ctx.sleeping`) — that wipes the display entirely.
struct ActivityTracker {
    last_active_ms: u64,
    last_render_ms: u64,
}

impl ActivityTracker {
    const fn new() -> Self {
        Self {
            last_active_ms: 0,
            last_render_ms: 0,
        }
    }

    /// Sleep / breathing thresholds.
    const BREATH_AFTER_MS: u64 = 30_000;
    /// Full breath cycle (one in, one out) — slow enough to feel calm.
    const BREATH_PERIOD_MS: u64 = 3_000;

    /// Call once per render before drawing. Returns true on the first render
    /// after init (so callers can seed `last_active_ms`).
    fn tick(&mut self, ctx: &RenderContext) -> u64 {
        let now = Instant::now().as_millis();
        // Seed on first call so a freshly powered-on keyboard doesn't start
        // breathing before the user has even pressed anything.
        if self.last_active_ms == 0 {
            self.last_active_ms = now;
        }
        if ctx.key_press_latch || ctx.key_pressed {
            self.last_active_ms = now;
        }
        self.last_render_ms = now;
        now
    }

    /// If the keyboard has been idle long enough, return a `0..=255`
    /// "darkness" mask density. 0 = fully bright, 255 = fully dark. The
    /// renderer applies this as a dither so the *content stays visible* but
    /// appears to pulse, matching the user's requested behaviour.
    fn breath_mask(&self) -> Option<u8> {
        let now = self.last_render_ms;
        let idle = now.saturating_sub(self.last_active_ms);
        if idle < Self::BREATH_AFTER_MS {
            return None;
        }
        // Triangle wave 0..=255 over BREATH_PERIOD_MS.
        let phase = (now % Self::BREATH_PERIOD_MS) as i32;
        let half = (Self::BREATH_PERIOD_MS / 2) as i32;
        let tri = if phase < half {
            (phase * 255) / half
        } else {
            ((Self::BREATH_PERIOD_MS as i32 - phase) * 255) / half
        };
        // Clamp and cap the darkness to keep some content visible at the
        // deepest dip (a fully-dark mask is just "screen off", which is
        // confusing during breathing).
        Some(tri.clamp(0, 200) as u8)
    }
}

// ─── 90° CCW DrawTarget wrapper ──────────────────────────────────────────────
//
// Lets us write `Text::new("BASE", point, FONT).draw(&mut rot)` in
// "horizontal" coordinates and have the result appear as text reading
// bottom-to-top on the physical panel (after the user has mounted the
// display vertically).
//
// Local (x, y) → inner (y, SCREEN_H - 1 - x).
//
//   inner (32 W × 128 H)        local view (128 W × 32 H)
//   ┌──────────┐                ┌──────────────────────────┐
//   │   top    │                │       left = inner top   │
//   │          │   90° CCW      │ ...                      │
//   │          │     →          │       right = inner bot. │
//   │ bottom   │                └──────────────────────────┘
//   └──────────┘
//
// So a string drawn at local Point::new(2, 4) starts near inner.y = 125
// (very bottom) and reads upward — exactly the "bottom-to-top" direction
// requested by the user.

struct Rot90<'a, D: DrawTarget<Color = BinaryColor>> {
    inner: &'a mut D,
}

impl<'a, D: DrawTarget<Color = BinaryColor>> Rot90<'a, D> {
    fn new(inner: &'a mut D) -> Self {
        Self { inner }
    }
}

// Only implement `OriginDimensions` — `Dimensions` comes for free via the
// blanket `impl<T: OriginDimensions> Dimensions for T` in embedded-graphics.
impl<D: DrawTarget<Color = BinaryColor>> OriginDimensions for Rot90<'_, D> {
    fn size(&self) -> Size {
        Size::new(SCREEN_H as u32, SCREEN_W as u32)
    }
}

impl<D: DrawTarget<Color = BinaryColor>> DrawTarget for Rot90<'_, D> {
    type Color = BinaryColor;
    type Error = D::Error;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        let mapped = pixels.into_iter().filter_map(|Pixel(p, c)| {
            // Drop pixels outside our (rotated) canvas — embedded-graphics
            // sometimes asks us to draw slightly out-of-bounds for glyphs
            // near the edge, and the inner driver would clip anyway, but
            // doing it here keeps the math symmetric.
            if p.x < 0 || p.x >= SCREEN_H || p.y < 0 || p.y >= SCREEN_W {
                None
            } else {
                Some(Pixel(Point::new(p.y, SCREEN_H - 1 - p.x), c))
            }
        });
        self.inner.draw_iter(mapped)
    }
}

// ─── Layer mapping ───────────────────────────────────────────────────────────

/// Short, large-font name for each layer. Matches the user's spec.
fn layer_short_name(layer: u8) -> &'static str {
    match layer {
        0 => "BASE",
        1 => "RAISE",
        2 => "LOWER",
        _ => "L?",
    }
}

/// Smaller, descriptive label for each layer.
fn layer_description(layer: u8) -> &'static str {
    match layer {
        0 => "QWERTY",
        1 => "NUM + NAV",
        2 => "SYMB + MEDIA",
        _ => "",
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Extract a 0..=100 battery percent from a [`BatteryStatus`], or [`None`]
/// if the battery driver hasn't reported a level yet.
fn battery_percent(b: BatteryStatus) -> Option<u8> {
    match b {
        BatteryStatus::Available { level: Some(p), .. } => Some(p),
        BatteryStatus::Available { level: None, .. } => Some(100),
        BatteryStatus::Unavailable => None,
    }
}

/// Apply a dither mask to the visible region of a display, simulating a
/// brightness reduction without actually toggling SSD1306 contrast (which
/// we can't reach from inside a `DisplayRenderer` — it only sees a
/// `DrawTarget`, not the underlying `Ssd1306Async`).
///
/// `darkness` is a 0..=255 value; higher = more pixels turned off.
fn apply_breath_dither<D: DrawTarget<Color = BinaryColor>>(display: &mut D, darkness: u8) {
    // 8×8 ordered Bayer matrix scaled to 0..=255 — gives a clean perceived
    // grey ramp without obvious banding.
    static BAYER: [[u8; 8]; 8] = [
        [0, 128, 32, 160, 8, 136, 40, 168],
        [192, 64, 224, 96, 200, 72, 232, 104],
        [48, 176, 16, 144, 56, 184, 24, 152],
        [240, 112, 208, 80, 248, 120, 216, 88],
        [12, 140, 44, 172, 4, 132, 36, 164],
        [204, 76, 236, 108, 196, 68, 228, 100],
        [60, 188, 28, 156, 52, 180, 20, 148],
        [252, 124, 220, 92, 244, 116, 212, 84],
    ];
    // Single `draw_iter` call with a lazy iterator that yields one OFF pixel
    // per "dimmed" position. Because the underlying DrawTarget is
    // write-only we can't AND existing pixels — instead we paint OFF on
    // top, which acts as a dim/erase mask wherever a previously-drawn ON
    // pixel coincides with our dither pattern.
    let iter = (0..SCREEN_H).flat_map(move |y| {
        (0..SCREEN_W).filter_map(move |x| {
            let t = BAYER[(y & 7) as usize][(x & 7) as usize];
            if t < darkness {
                Some(Pixel(Point::new(x, y), BinaryColor::Off))
            } else {
                None
            }
        })
    });
    let _ = display.draw_iter(iter);
}

/// Centre `text` horizontally inside a `width`-pixel-wide column at `(x, y)`
/// using the given monospaced style.
fn draw_text_centered<D: DrawTarget<Color = BinaryColor>>(
    display: &mut D,
    text: &str,
    x: i32,
    y: i32,
    width: i32,
    style: MonoTextStyle<'_, BinaryColor>,
    baseline: Baseline,
) {
    let char_w = style.font.character_size.width as i32;
    let text_w = text.chars().count() as i32 * char_w;
    let pad = (width - text_w).max(0) / 2;
    let _ = Text::with_baseline(text, Point::new(x + pad, y), style, baseline).draw(display);
}

// ─── Battery section (left screen top) ───────────────────────────────────────

fn draw_left_battery_section<D: DrawTarget<Color = BinaryColor>>(
    display: &mut D,
    ctx: &RenderContext,
) {
    let pct = battery_percent(*ctx.battery).unwrap_or(0);

    // Bar outline: x = 2..30 (28 px usable interior), y = 4..12 (centred in
    // the top zone). 1-px stroke + interior fill that grows with charge.
    let outline = Rectangle::new(Point::new(2, 4), Size::new(28, 8));
    let _ = outline.into_styled(STROKE).draw(display);

    if battery_percent(*ctx.battery).is_some() {
        let fill_w = ((pct as i32 * 26) / 100).clamp(0, 26) as u32;
        if fill_w > 0 {
            let fill_rect = Rectangle::new(Point::new(3, 5), Size::new(fill_w, 6));
            let _ = fill_rect
                .into_styled(PrimitiveStyleBuilder::new().fill_color(BinaryColor::On).build())
                .draw(display);
        }
    }

    // Battery percentage text below the bar — small font, centred.
    let mut label: String<8> = String::new();
    if battery_percent(*ctx.battery).is_some() {
        let _ = write!(label, "{}%", pct);
    } else {
        let _ = label.push_str("--");
    }
    draw_text_centered(display, &label, 0, 16, SCREEN_W, FONT_SMALL, Baseline::Top);

    // Hair-line separator between this zone and the rotated layer zone below.
    let _ = Rectangle::new(
        Point::new(0, LEFT_BATTERY_ZONE_H - 2),
        Size::new(SCREEN_W as u32, 1),
    )
    .into_styled(FILL)
    .draw(display);
}

// ─── Layer section (left screen bottom, rotated) ─────────────────────────────
//
// In Rot90-local coordinates the canvas is 128 wide × 32 tall. The rotated
// "layer" zone on the physical screen spans physical y = 34..127, which maps
// to local x = 0..(128-34) = 0..93. So drawing text in local x = 0..93,
// local y = 0..31 stays inside the bottom of the panel.

fn draw_left_layer_section<D: DrawTarget<Color = BinaryColor>>(
    display: &mut D,
    ctx: &RenderContext,
) {
    let mut rot = Rot90::new(display);

    // Local x extent we're allowed to use (avoid drawing into the battery
    // zone above the separator). Physical y = 34..127 → local x = 0..93.
    let max_local_x = SCREEN_H - LEFT_BATTERY_ZONE_H; // 94

    let name = layer_short_name(ctx.layer);
    let desc = layer_description(ctx.layer);

    // Left column (= physical left half) shows the BIG short name.
    // local y span: 0..15 → physical x: 0..15.
    let big_h = FONT_BIG.font.character_size.height as i32;
    let big_baseline_y = (16 - big_h) / 2 + big_h - 2;
    let name_w = name.chars().count() as i32 * FONT_BIG.font.character_size.width as i32;
    let name_start_x = ((max_local_x - name_w) / 2).max(2);
    let _ = Text::with_baseline(
        name,
        Point::new(name_start_x, big_baseline_y),
        FONT_BIG,
        Baseline::Alphabetic,
    )
    .draw(&mut rot);

    // Right column (= physical right half) shows the smaller description.
    // local y span: 16..31 → physical x: 16..31.
    let small_h = FONT_MED.font.character_size.height as i32;
    let small_baseline_y = 16 + (16 - small_h) / 2 + small_h - 2;
    let desc_w = desc.chars().count() as i32 * FONT_MED.font.character_size.width as i32;
    let desc_start_x = ((max_local_x - desc_w) / 2).max(2);
    let _ = Text::with_baseline(
        desc,
        Point::new(desc_start_x, small_baseline_y),
        FONT_MED,
        Baseline::Alphabetic,
    )
    .draw(&mut rot);
}

// ─── Right screen ────────────────────────────────────────────────────────────

/// WPM samples kept for the sparkline on the right half. Push policy: every
/// time RMK delivers a different WPM value via the split-link sync we
/// shift-and-append a single sample.
const WPM_SAMPLES: usize = 8;

pub struct RightRenderer {
    activity: ActivityTracker,
    wpm_history: [u8; WPM_SAMPLES],
    last_observed_wpm: u16,
}

impl Default for RightRenderer {
    fn default() -> Self {
        Self {
            activity: ActivityTracker::new(),
            wpm_history: [0; WPM_SAMPLES],
            last_observed_wpm: u16::MAX,
        }
    }
}

impl RightRenderer {
    fn record_wpm(&mut self, wpm: u16) {
        if wpm != self.last_observed_wpm {
            // Shift left, append. Cap at u8::MAX since the sparkline only
            // cares about a 0..=255 range.
            for i in 1..WPM_SAMPLES {
                self.wpm_history[i - 1] = self.wpm_history[i];
            }
            self.wpm_history[WPM_SAMPLES - 1] = wpm.min(u8::MAX as u16) as u8;
            self.last_observed_wpm = wpm;
        }
    }
}

impl DisplayRenderer<BinaryColor> for RightRenderer {
    fn render<D: DrawTarget<Color = BinaryColor>>(&mut self, ctx: &RenderContext, display: &mut D) {
        let _ = display.clear(BinaryColor::Off);
        if ctx.sleeping {
            return;
        }

        self.activity.tick(ctx);
        self.record_wpm(ctx.wpm);

        let mut rot = Rot90::new(display);

        let max_local_x = SCREEN_H; // full panel available
        let left_col_y0 = 0i32;
        let right_col_y0 = 16i32;

        // ── Left column (top → bottom on screen) ─────────────────────────
        // Modifiers first (only those currently active), then WPM at bottom.
        let m = ctx.modifiers;
        let mods: &[(bool, &str)] = &[
            (m.left_gui() || m.right_gui(), "CMD"),
            (m.left_shift() || m.right_shift(), "SHFT"),
            (m.left_ctrl() || m.right_ctrl(), "CTRL"),
            (m.left_alt() || m.right_alt(), "OPT"),
        ];

        // Modifiers occupy the upper ~3/4 of the column; WPM gets the
        // bottom slot. "Top of screen" = high local x.
        let mod_slot_h = (max_local_x * 3) / 4 / 4; // 24-ish px per slot
        let mod_top_x = max_local_x - 4;
        let small_w = FONT_SMALL.font.character_size.width as i32;
        for (i, &(active, label)) in mods.iter().enumerate() {
            if !active {
                continue;
            }
            // local x range for this slot: bot..top, anchored top-down so that
            // the first modifier (CMD) sits closest to the top of the screen.
            let slot_top = mod_top_x - (i as i32) * mod_slot_h;
            let slot_bot = slot_top - mod_slot_h;
            let label_w = label.len() as i32 * small_w;
            let baseline_x = slot_bot + (mod_slot_h - label_w) / 2;
            let _ = Text::with_baseline(
                label,
                Point::new(baseline_x.max(0), left_col_y0 + 2),
                FONT_SMALL,
                Baseline::Top,
            )
            .draw(&mut rot);
        }

        // WPM label at the bottom of the left column. Two glyph-runs side by
        // side in local coords: "WPM" first (low local x → physical bottom)
        // and the numeric count "above" it on the physical screen (higher
        // local x).
        let mut wpm_count: String<8> = String::new();
        let _ = write!(wpm_count, "{}", ctx.wpm);
        let wpm_label_w = 3 * small_w; // "WPM"
        let _ = Text::with_baseline(
            "WPM",
            Point::new(2, left_col_y0 + 2),
            FONT_SMALL,
            Baseline::Top,
        )
        .draw(&mut rot);
        let _ = Text::with_baseline(
            &wpm_count,
            Point::new(2 + wpm_label_w + 4, left_col_y0 + 2),
            FONT_SMALL,
            Baseline::Top,
        )
        .draw(&mut rot);

        // ── Right column (top → bottom on screen) ────────────────────────
        // Connection status, then CAPS (middle, only when active), then
        // WPM sparkline at the bottom.
        let mut conn_label: String<8> = String::new();
        write_connection_label(&mut conn_label, ctx);
        let label_w = conn_label.len() as i32 * small_w;
        let conn_baseline_x = max_local_x - 4 - label_w; // anchored near top
        let _ = Text::with_baseline(
            &conn_label,
            Point::new(conn_baseline_x.max(0), right_col_y0 + 4),
            FONT_SMALL,
            Baseline::Top,
        )
        .draw(&mut rot);

        // CAPS in the middle — only when caps lock is active.
        if ctx.caps_lock {
            let caps_x = max_local_x / 2 - 2 * small_w;
            let _ = Text::with_baseline(
                "CAPS",
                Point::new(caps_x.max(0), right_col_y0 + 4),
                FONT_SMALL,
                Baseline::Top,
            )
            .draw(&mut rot);
        }

        // Sparkline graph at the bottom of the right column. "GRAPH" label
        // followed by 8 vertical bars whose height is proportional to the
        // recorded WPM value.
        let graph_label_x = 4i32; // very bottom of screen
        let _ = Text::with_baseline(
            "GRAPH",
            Point::new(graph_label_x, right_col_y0 + 4),
            FONT_SMALL,
            Baseline::Top,
        )
        .draw(&mut rot);

        // Determine bars area: local x = 4 + 5*small_w .. 4 + 5*small_w + 40
        // local y = right_col_y0 + 1 .. right_col_y0 + 15  (column inside right col)
        let bars_x0 = graph_label_x + 5 * small_w + 2;
        let bars_w_each = 3i32; // 2 px bar + 1 px gap
        let max_bar_h = 12i32;
        let max_sample = self.wpm_history.iter().copied().max().unwrap_or(0).max(1) as i32;
        for (i, sample) in self.wpm_history.iter().copied().enumerate() {
            let h = ((sample as i32) * max_bar_h / max_sample).clamp(0, max_bar_h);
            if h == 0 {
                continue;
            }
            let bar_local_x = bars_x0 + (i as i32) * bars_w_each;
            // Bar grows upward from bottom of the right column.
            let bar = Rectangle::new(
                Point::new(bar_local_x, right_col_y0 + (15 - h)),
                Size::new(2, h as u32),
            );
            let _ = bar.into_styled(FILL).draw(&mut rot);
        }

        // Done with Rot90 — drop it so we can mutate `display` again.
        drop(rot);

        if let Some(mask) = self.activity.breath_mask() {
            apply_breath_dither(display, mask);
        }
    }
}

/// Write either "BT N" (if BLE is connected) or "USB" into `buf`. Falls back
/// to "ADV"/"USB" depending on BLE state.
fn write_connection_label<const N: usize>(buf: &mut String<N>, ctx: &RenderContext) {
    match ctx.ble_status.state {
        BleState::Connected => {
            let _ = write!(buf, "BT {}", ctx.ble_status.profile);
        }
        BleState::Advertising => {
            let _ = buf.push_str("ADV");
        }
        BleState::Inactive => {
            // No BLE active → assume USB host on the central. The right
            // half mirrors the central's host transport state via
            // `SplitMessage::ConnectionState`, but the display renderer
            // only sees the `RenderContext` (no direct access). The user
            // asked for "BT N or USB", so map "BLE inactive" → "USB".
            let _ = buf.push_str("USB");
        }
    }
}

// ─── Left screen ─────────────────────────────────────────────────────────────

pub struct LeftRenderer {
    activity: ActivityTracker,
}

impl Default for LeftRenderer {
    fn default() -> Self {
        Self {
            activity: ActivityTracker::new(),
        }
    }
}

impl DisplayRenderer<BinaryColor> for LeftRenderer {
    fn render<D: DrawTarget<Color = BinaryColor>>(&mut self, ctx: &RenderContext, display: &mut D) {
        let _ = display.clear(BinaryColor::Off);
        if ctx.sleeping {
            return;
        }

        self.activity.tick(ctx);

        draw_left_battery_section(display, ctx);
        draw_left_layer_section(display, ctx);

        if let Some(mask) = self.activity.breath_mask() {
            apply_breath_dither(display, mask);
        }
    }
}
