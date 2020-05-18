#![feature(iter_partition_in_place)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(min_const_generics)]

extern crate image;
extern crate lazy_static;
extern crate rand;

pub mod bvh;
pub mod camera;
pub mod color;
pub mod film;
pub mod filters;
pub mod geometry;
pub mod interaction;
pub mod lights;
pub mod lowdiscrepancy;
pub mod material;
pub mod medium;
pub mod microfacet;
pub mod misc;
pub mod primitives;
pub mod reflection;
pub mod rtoycore;
pub mod samplers;
pub mod scene;
pub mod shape;
pub mod spectrum;
pub mod texture;
pub mod transform;
