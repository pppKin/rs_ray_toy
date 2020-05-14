use crate::geometry::{Point2f, Point2i};
use rand::prelude::*;
use std::sync::RwLock;

pub struct HaltonSampler {
    pub samples_per_pixel: i64,
    pub base_scales: Point2i,
    pub base_exponents: Point2i,
    pub sample_stride: u64,
    pub mult_inverse: [i64; 2],
    pub pixel_for_offset: RwLock<Point2i>,
    pub offset_for_current_pixel: RwLock<u64>,
    pub sample_at_pixel_center: bool, // default: false
    // inherited from class GlobalSampler (see sampler.h)
    pub dimension: i64,
    pub interval_sample_index: u64,
    pub array_start_dim: i64,
    pub array_end_dim: i64,
    // inherited from class Sampler (see sampler.h)
    pub current_pixel: Point2i,
    pub current_pixel_sample_index: i64,
    pub samples_1d_array_sizes: Vec<i32>,
    pub samples_2d_array_sizes: Vec<i32>,
    pub sample_array_1d: Vec<Vec<f64>>,
    pub sample_array_2d: Vec<Vec<Point2f>>,
    pub array_1d_offset: usize,
    pub array_2d_offset: usize,
}
