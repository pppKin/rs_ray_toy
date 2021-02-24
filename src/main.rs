#![feature(iter_partition_in_place)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(associated_type_bounds)]

mod bssrdf;
mod bvh;
mod camera;
mod color;
mod film;
mod filters;
mod geometry;
mod integrator;
mod interaction;
mod interpolation;
mod lights;
mod lowdiscrepancy;
mod material;
mod medium;
mod memory;
mod microfacet;
mod mipmap;
mod misc;
mod primitives;
mod reflection;
mod samplers;
mod sampling;
mod scene;
mod shape;
mod spectrum;
mod texture;
mod transform;

extern crate image;
extern crate lazy_static;
extern crate rand;
extern crate rayon;

use std::{env, str::FromStr};

use image::{ImageBuffer, ImageError, Rgba};

use crate::{
    geometry::{Bounds2i, Point2i},
    misc::{clamp_t, gamma_correct},
    scene::Scene,
};
pub const N_SPECTRAL_SAMPLES: usize = 60;
pub const SPECTRUM_SAMPLED_N: usize = N_SPECTRAL_SAMPLES;
pub const SPECTRUM_RGB_N: usize = 3;
// Change this to use different Spectrum Representation
pub const SPECTRUM_N: usize = SPECTRUM_RGB_N;

pub const MAX_DIST: f64 = 1999999999.0;
pub const SMALL: f64 = 0.000000001;
pub const MACHINE_EPSILON: f64 = std::f64::EPSILON * 0.5;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];
    let num = u8::from_str(&args[2]).unwrap();
    let save_to = &args[3];
    let scene = make_scene(filename, num, save_to);
    todo!()
}

pub fn make_scene(filename: &str, num: u8, save_to: &str) -> Scene {
    todo!();
}

pub fn write_image(
    filename: &str,
    rgb: &[f64],
    output_bounds: Bounds2i,
    total_resolution: Point2i,
) -> Result<(), ImageError> {
    let resolution = output_bounds.diagonal();

    let mut img_buf = ImageBuffer::new(resolution.x as u32, resolution.y as u32);
    for y in 0..resolution.y {
        for x in 0..resolution.x {
            let r = rgb[(3 * (y * resolution.x + x) + 0) as usize];
            let g = rgb[(3 * (y * resolution.x + x) + 1) as usize];
            let b = rgb[(3 * (y * resolution.x + x) + 2) as usize];
            img_buf.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    (clamp_t(255.0 * gamma_correct(r) + 0.5, 0.0, 255.0)) as u8,
                    (clamp_t(255.0 * gamma_correct(g) + 0.5, 0.0, 255.0)) as u8,
                    (clamp_t(255.0 * gamma_correct(b) + 0.5, 0.0, 255.0)) as u8,
                    255 as u8,
                ]),
            )
        }
    }
    img_buf.save(filename)
}
