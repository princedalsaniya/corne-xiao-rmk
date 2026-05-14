#![no_main]
#![no_std]

use rmk::macros::rmk_peripheral;

// Same as `central.rs`: the renderer path in keyboard.toml expects
// `crate::display::RightRenderer` to be reachable from this binary's root.
// `dead_code` allowed because `LeftRenderer` is only used by the central
// binary; both share `src/display.rs`.
#[allow(dead_code)]
mod display;

#[rmk_peripheral(id = 0)]
mod keyboard_peripheral {}
