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
mod renderprocess;
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
extern crate serde_json;

use std::env;

use renderprocess::deploy_render;

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

    let filepath = &args[1];
    let save_to = &args[2];
    deploy_render(filepath, save_to);
}
