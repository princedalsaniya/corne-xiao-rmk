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
//! Sleep / power story (must be read alongside `[rmk]
//! split_central_sleep_timeout_seconds = 180` in keyboard.toml):
//!
//!   * 0 – 30 s after the last key press: content drawn full-bright.
//!   * 30 s – 3 min: still awake, content drawn, but the framebuffer is
//!     dithered (Bayer mask) to produce a "breathing" pulse. RMK is still
//!     in its normal connection interval (7.5 ms).
//!   * 3 min: the RMK central sleep manager drops the BLE link to a
//!     low-power connection interval and broadcasts
//!     `SleepStateEvent(true)`. Both halves now see `ctx.sleeping = true`
//!     in their renderer.  The display processor *stops periodic polling*
//!     while sleeping (`rmk/src/display/mod.rs`), so the framebuffer just
//!     stays at whichever breathing frame was last drawn until the next
//!     state event triggers a re-render (battery poll, BLE health, etc.).
//!   * 10 min total idle = 7 min after entering RMK sleep: the renderer
//!     decides we're now in "deep sleep" and starts blanking the framebuffer
//!     entirely (all OFF pixels). Note: we can't reach the SSD1306 0xAE
//!     command from inside a `DisplayRenderer` (the trait only sees a
//!     `DrawTarget` — not the underlying `Ssd1306Async`), so this is a
//!     visual "off" rather than a panel-power-down. The screen looks
//!     blank; the SSD1306 segment drivers continue scanning the all-OFF
//!     framebuffer (a ~20 µA cost on top of the already-sleeping radio,
//!     well under the host BLE link's own consumption).
//!   * Any key press: RMK fires `SleepStateEvent(false)` + the key event;
//!     `ctx.sleeping` flips back to false, our tracker resets, and the
//!     next render restores full-bright content immediately. The host
//!     doesn't see a disconnect.

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

/// Tracks the last time a key press was observed (for the breathing pulse
/// at 30 s) and the moment `ctx.sleeping` first became true (for the
/// 7-minutes-into-sleep "deep sleep" / blank-screen cutoff).
///
/// Why a custom tracker instead of leaning on `RenderContext`: RMK only
/// hands us a *snapshot* of state on each render — it has no concept of
/// "how long has `ctx.sleeping` been true". We need our own clock to
/// implement the 7-minute deep-sleep delay.
struct ActivityTracker {
    /// Wall-clock (ms since boot) of the most recent key event.
    last_active_ms: u64,
    /// Wall-clock of the most recent render — used as the time reference
    /// inside `breath_mask` so the renderer's view of "now" stays consistent
    /// across the helper methods called within one `render()` invocation.
    last_render_ms: u64,
    /// Wall-clock at which `ctx.sleeping` most recently flipped from false
    /// to true. 0 means "not currently sleeping" (RMK is awake, or RMK is
    /// sleeping but we haven't yet rendered after the transition — in that
    /// case the next `tick()` will seed this).
    sleep_started_ms: u64,
}

impl ActivityTracker {
    const fn new() -> Self {
        Self {
            last_active_ms: 0,
            last_render_ms: 0,
            sleep_started_ms: 0,
        }
    }

    /// Idle time after which the breathing pulse kicks in (while still awake).
    const BREATH_AFTER_MS: u64 = 30_000;
    /// Full breath cycle (one in, one out) — slow enough to feel calm.
    const BREATH_PERIOD_MS: u64 = 3_000;
    /// How long `ctx.sleeping` must stay continuously true before we treat
    /// the state as "deep sleep" and blank the framebuffer entirely.
    /// 7 min × 60 s × 1000 ms — combined with the 3-min RMK sleep timeout
    /// in keyboard.toml this gives "10 min total idle → screens off".
    const DEEP_SLEEP_AFTER_SLEEP_MS: u64 = 7 * 60 * 1_000;

    /// Call once per render before drawing.
    ///
    /// Updates `last_active_ms` on every observed key event, and tracks the
    /// continuous-sleep window: as soon as `ctx.sleeping` becomes false
    /// (wake!) we reset `sleep_started_ms` so the 7-minute timer starts
    /// fresh on the next sleep cycle.
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

        // Sleep-window tracking.
        //
        // RMK reports a single boolean (`ctx.sleeping`). We watch the
        // transitions:
        //   false → true : record the moment to start the 7-minute timer.
        //   true  → false: clear the timer so a future sleep cycle starts
        //                  the clock fresh.
        //   stays true   : leave `sleep_started_ms` alone so elapsed grows.
        //
        // Edge case: if we first observe `ctx.sleeping = true` (the very
        // first render after `SleepStateEvent(true)` arrives) and our
        // `sleep_started_ms` is still 0, we seed it to *now* — meaning the
        // 7-minute timer effectively starts from "first render after RMK
        // told us we're sleeping", not from "the SleepStateEvent timestamp"
        // (they're within a few ms anyway).
        if ctx.sleeping {
            if self.sleep_started_ms == 0 {
                self.sleep_started_ms = now;
            }
        } else {
            self.sleep_started_ms = 0;
        }

        self.last_render_ms = now;
        now
    }

    /// True once `ctx.sleeping` has been continuously true for at least
    /// [`DEEP_SLEEP_AFTER_SLEEP_MS`]. Caller short-circuits the rest of the
    /// render pipeline and leaves the framebuffer cleared (all OFF pixels).
    ///
    /// Call this *after* [`tick`], so `sleep_started_ms` reflects the
    /// current render's view of state.
    fn is_deep_sleep(&self) -> bool {
        if self.sleep_started_ms == 0 {
            return false;
        }
        self.last_render_ms.saturating_sub(self.sleep_started_ms) >= Self::DEEP_SLEEP_AFTER_SLEEP_MS
    }

    /// If the keyboard has been idle long enough, return a `0..=255`
    /// "darkness" mask density. 0 = fully bright, 255 = fully dark. The
    /// renderer applies this as a dither so the *content stays visible* but
    /// appears to pulse, matching the user's requested behaviour.
    ///
    /// Note this triggers off `last_active_ms` only, so it also fires while
    /// RMK is in light sleep (`ctx.sleeping = true`, < 7 min) — by then
    /// idle is ≥ 3 min so we're well past the 30 s threshold.
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

// ─── Battery section (top of either screen, direct coords) ──────────────────

fn draw_battery_section<D: DrawTarget<Color = BinaryColor>>(
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

pub struct RightRenderer {
    activity: ActivityTracker,
}

impl Default for RightRenderer {
    fn default() -> Self {
        Self {
            activity: ActivityTracker::new(),
        }
    }
}

impl DisplayRenderer<BinaryColor> for RightRenderer {
    fn render<D: DrawTarget<Color = BinaryColor>>(&mut self, ctx: &RenderContext, display: &mut D) {
        let _ = display.clear(BinaryColor::Off);

        // Tick first so `is_deep_sleep` sees up-to-date sleep timing.
        self.activity.tick(ctx);

        // 7+ minutes of continuous `ctx.sleeping = true` (= 10 min total
        // idle, given keyboard.toml's 3-min sleep threshold) → blank the
        // framebuffer entirely and skip the rest of the render. On wake,
        // `ctx.sleeping` flips to false, `tick()` clears the sleep timer,
        // `is_deep_sleep()` returns false, and we resume drawing immediately.
        //
        // We can't reach the SSD1306 0xAE (display off) command from here
        // — the renderer only gets a `DrawTarget`, not the underlying
        // `Ssd1306Async`. Painting an all-OFF framebuffer is the closest
        // approximation reachable via the RMK display API.
        if self.activity.is_deep_sleep() {
            return;
        }

        // Battery bar + % at physical top — same section as left OLED.
        draw_battery_section(display, ctx);

        let mut rot = Rot90::new(display);

        // Cap local x to the space below the battery zone so Rot90 content
        // never overwrites the battery bar drawn above.
        let max_local_x = SCREEN_H - LEFT_BATTERY_ZONE_H; // 94 px
        let left_col_y0 = 0i32;
        let right_col_y0 = 16i32;
        let small_w = FONT_SMALL.font.character_size.width as i32;

        // ── Left column: WPM bottom-aligned ──────────────────────────────
        let mut wpm_str: String<12> = String::new();
        let _ = write!(wpm_str, "WPM {}", ctx.wpm);
        let _ = Text::with_baseline(
            &wpm_str,
            Point::new(2, left_col_y0 + 2),
            FONT_SMALL,
            Baseline::Top,
        )
        .draw(&mut rot);

        // ── Right column: modifiers stacked from bottom upward ────────────
        // CMD at the very bottom, then SHF, CTL, OPT. Only active ones drawn.
        let m = ctx.modifiers;
        let mod_list: &[(bool, &str)] = &[
            (m.left_gui()   || m.right_gui(),   "CMD"),
            (m.left_shift() || m.right_shift(), "SHF"),
            (m.left_ctrl()  || m.right_ctrl(),  "CTL"),
            (m.left_alt()   || m.right_alt(),   "OPT"),
        ];
        let mod_slot = 3 * small_w + 2; // 17 px per slot in local x
        let mut slot = 0i32;
        for &(active, label) in mod_list.iter() {
            if !active {
                continue;
            }
            let _ = Text::with_baseline(
                label,
                Point::new(2 + slot * mod_slot, right_col_y0 + 4),
                FONT_SMALL,
                Baseline::Top,
            )
            .draw(&mut rot);
            slot += 1;
        }

        drop(rot);

        if let Some(mask) = self.activity.breath_mask() {
            apply_breath_dither(display, mask);
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

        self.activity.tick(ctx);

        // See `RightRenderer::render` for the rationale on the deep-sleep
        // short-circuit and why we don't send the SSD1306 0xAE command.
        if self.activity.is_deep_sleep() {
            return;
        }

        draw_battery_section(display, ctx);
        draw_left_layer_section(display, ctx);

        if let Some(mask) = self.activity.breath_mask() {
            apply_breath_dither(display, mask);
        }
    }
}
