#![no_main]
#![no_std]

use rmk::macros::rmk_central;

// Display renderers live in `crate::display` so the `renderer =
// "crate::display::LeftRenderer"` path in keyboard.toml resolves. The macro
// expansion of `#[rmk_central]` emits `crate::display::LeftRenderer::default()`
// at the call site that constructs the display processor.
//
// `dead_code` is allowed because the file also defines `RightRenderer` (used
// only by the peripheral binary). Both binaries share `src/display.rs`.
#[allow(dead_code)]
mod display;

#[rmk_central]
mod keyboard_central {}
