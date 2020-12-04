#![feature(iter_partition_in_place)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(const_generics)]

mod bvh;
mod camera;
mod color;
mod film;
mod filters;
mod geometry;
mod interaction;
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
mod rtoycore;
mod samplers;
mod scene;
mod shape;
mod spectrum;
mod texture;
mod transform;

use std::{env, str::FromStr};

extern crate image;
extern crate lazy_static;
extern crate rand;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];
    let num = u8::from_str(&args[2]).unwrap();
    let save_to = &args[3];

    // rtoycore::deploy_renderer(filename, num, save_to);
}
