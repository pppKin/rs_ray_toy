mod camera;
mod color;
mod core;
mod geometry;
mod lights;
mod material;
mod misc;
mod primitives;
mod sampler;
mod scene;
mod transform;

use std::env;
use std::str::FromStr;

extern crate image;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];
    let num = u8::from_str(&args[2]).unwrap();
    let save_to = &args[3];

    core::deploy_renderer(filename, num, save_to);
}
