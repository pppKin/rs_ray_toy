use std::fmt::Debug;

use crate::{camera::*, geometry::*};
// use crate::misc::ONE_MINUS_EPSILON;
use rand::prelude::*;

pub trait RoundCount {
    fn round_count(&self, n: u32) -> u32 {
        n
    }
}

pub trait StartPixel {
    fn start_pixel(&mut self, p: Point2i);
}

pub trait ISampler: RoundCount + StartPixel + Debug + Send + Sync {
    fn start_next_sample(&mut self) -> bool;
    fn set_sample_number(&mut self, sample_num: u64) -> bool;
    fn request_1d_array(&mut self, n: u32);
    fn request_2d_array(&mut self, n: u32);
    fn get_1d_array(&mut self, n: u32) -> &[f64];
    fn get_2d_array(&mut self, n: u32) -> &[Point2f];
    fn samples_per_pixel(&self) -> u64;

    fn get_1d(&mut self) -> f64 {
        0.0
    }
    fn get_2d(&mut self) -> Point2f {
        Point2f::new(0.0, 0.0)
    }
    fn get_camerasample(&mut self, p_raster: &Point2i) -> CameraSample {
        CameraSample {
            p_film: Point2f::new(p_raster.x as f64, p_raster.y as f64) + self.get_2d(),
            p_lens: self.get_2d() + Point2f::new(0.5, 0.5),
            time: self.get_1d() + 0.5,
        }
    }

    fn current_sample_number(&self) -> u64;
}

#[derive(Clone, Debug, Default)]
pub struct BaseSampler {
    pub samples_per_pixel: u64,

    current_pixel: Point2i,
    current_pixel_sample_index: u64,
    samples1d_array_sizes: Vec<u32>,
    samples2d_array_sizes: Vec<u32>,
    sample_array1d: Vec<Vec<f64>>,
    sample_array2d: Vec<Vec<Point2f>>,

    array1d_offset: usize,
    array2d_offset: usize,
}

impl RoundCount for BaseSampler {}

impl StartPixel for BaseSampler {
    fn start_pixel(&mut self, p: Point2i) {
        self.current_pixel = p;
        self.current_pixel_sample_index = 0;
        self.array1d_offset = 0;
        self.array2d_offset = 0;
    }
}

impl BaseSampler {
    fn new(samples_per_pixel: u64) -> Self {
        let mut splr = BaseSampler::default();
        splr.samples_per_pixel = samples_per_pixel;
        return splr;
    }
    fn start_next_sample(&mut self) -> bool {
        self.array1d_offset = 0;
        self.array2d_offset = 0;
        self.current_pixel_sample_index += 1;
        self.current_pixel_sample_index < self.samples_per_pixel
    }
    fn set_sample_number(&mut self, sample_num: u64) -> bool {
        self.array1d_offset = 0;
        self.array2d_offset = 0;
        self.current_pixel_sample_index += sample_num;
        self.current_pixel_sample_index < self.samples_per_pixel
    }
    fn request_1d_array(&mut self, n: u32) {
        assert_eq!(self.round_count(n), n);
        self.samples1d_array_sizes.push(n);
        self.sample_array1d.push(Vec::<f64>::with_capacity(
            (n as u64 * self.samples_per_pixel) as usize,
        ));
    }
    fn request_2d_array(&mut self, n: u32) {
        assert_eq!(self.round_count(n), n);
        self.samples2d_array_sizes.push(n);
        self.sample_array2d.push(Vec::<Point2f>::with_capacity(
            (n as u64 * self.samples_per_pixel) as usize,
        ));
    }
    fn get_1d_array(&mut self, n: u32) -> &[f64] {
        if self.array1d_offset == self.sample_array1d.len() {
            return &[];
        }
        assert_eq!(self.samples1d_array_sizes[n as usize], n);
        assert!(self.current_pixel_sample_index < self.samples_per_pixel);
        self.array1d_offset += 1;
        let start = (n * self.current_pixel_sample_index as u32) as usize;
        let end = start + n as usize;
        return &self.sample_array1d[self.array1d_offset - 1][start..end];
    }
    fn get_2d_array(&mut self, n: u32) -> &[Point2f] {
        if self.array2d_offset == self.sample_array2d.len() {
            return &[];
        }
        assert_eq!(self.samples2d_array_sizes[n as usize], n);
        assert!(self.current_pixel_sample_index < self.samples_per_pixel);
        self.array2d_offset += 1;
        let start = (n * self.current_pixel_sample_index as u32) as usize;
        let end = start + n as usize;
        return &self.sample_array2d[self.array2d_offset - 1][start..end];
    }
}

// While some sampling algorithms can easily incrementally generate elements of
// each sample vector, others more naturally generate all of the dimensions’ sample values
// for all of the sample vectors for a pixel at the same time.

#[derive(Clone, Debug, Default)]
pub struct PixelSamplerData {
    samples1d: Vec<Vec<f64>>,
    samples2d: Vec<Vec<Point2f>>,
    current1d_dimension: u32,
    current2d_dimension: u32,
}

pub trait PixelSamplerStartPixel {
    fn start_pixel_ps(&mut self, p: Point2i, bsplr: &mut BaseSampler, psplr: &mut PixelSamplerData);
}

#[derive(Debug, Clone)]
pub struct PixelSampler<T>
where
    T: PixelSamplerStartPixel + RoundCount + Send + Sync,
{
    bsplr: BaseSampler,
    psplr: PixelSamplerData,
    splr: T,
}

impl<T> PixelSampler<T>
where
    T: PixelSamplerStartPixel + RoundCount + Send + Sync,
{
    fn new(samples_per_pixel: u64, n_sampled_dimensions: u32, splr: T) -> Self {
        let mut s = PixelSampler {
            bsplr: BaseSampler::new(samples_per_pixel),
            psplr: PixelSamplerData::default(),
            splr,
        };

        for _i in 00..n_sampled_dimensions {
            s.psplr
                .samples1d
                .push(vec![0.0; samples_per_pixel as usize]);
            s.psplr
                .samples2d
                .push(vec![Point2f::default(); samples_per_pixel as usize]);
        }
        return s;
    }
}

impl<T> RoundCount for PixelSampler<T>
where
    T: PixelSamplerStartPixel + RoundCount + Send + Sync,
{
    fn round_count(&self, n: u32) -> u32 {
        self.splr.round_count(n)
    }
}

impl<T> StartPixel for PixelSampler<T>
where
    T: PixelSamplerStartPixel + RoundCount + Send + Sync,
{
    fn start_pixel(&mut self, p: Point2i) {
        // self.s.start_pixel(p);
        self.splr
            .start_pixel_ps(p, &mut self.bsplr, &mut self.psplr);
        self.bsplr.start_pixel(p);
    }
}

impl<T> ISampler for PixelSampler<T>
where
    T: PixelSamplerStartPixel + RoundCount + Debug + Send + Sync,
{
    fn start_next_sample(&mut self) -> bool {
        self.psplr.current1d_dimension = 0;
        self.psplr.current2d_dimension = 0;
        self.bsplr.start_next_sample()
    }
    fn set_sample_number(&mut self, sample_num: u64) -> bool {
        self.psplr.current1d_dimension = 0;
        self.psplr.current2d_dimension = 0;
        self.bsplr.set_sample_number(sample_num)
    }
    fn get_1d(&mut self) -> f64 {
        assert!(self.bsplr.current_pixel_sample_index < self.bsplr.samples_per_pixel);
        if (self.psplr.current1d_dimension as usize) < self.psplr.samples1d.len() {
            self.psplr.current1d_dimension += 1;
            return self.psplr.samples1d[(self.psplr.current1d_dimension - 1) as usize]
                [self.bsplr.current_pixel_sample_index as usize];
        } else {
            let mut rng = thread_rng();
            return rng.gen_range(-1.0..1.0);
        }
    }
    fn get_2d(&mut self) -> Point2f {
        assert!(self.bsplr.current_pixel_sample_index < self.bsplr.samples_per_pixel);
        if (self.psplr.current2d_dimension as usize) < self.psplr.samples2d.len() {
            self.psplr.current2d_dimension += 1;
            return self.psplr.samples2d[(self.psplr.current1d_dimension - 1) as usize]
                [self.bsplr.current_pixel_sample_index as usize];
        } else {
            let mut rng = thread_rng();

            return Point2f::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        }
    }

    fn request_1d_array(&mut self, n: u32) {
        self.bsplr.request_1d_array(n);
    }

    fn request_2d_array(&mut self, n: u32) {
        self.bsplr.request_2d_array(n);
    }

    fn get_1d_array(&mut self, n: u32) -> &[f64] {
        self.bsplr.get_1d_array(n)
    }

    fn get_2d_array(&mut self, n: u32) -> &[Point2f] {
        self.bsplr.get_2d_array(n)
    }

    fn samples_per_pixel(&self) -> u64 {
        self.bsplr.samples_per_pixel
    }

    fn current_sample_number(&self) -> u64 {
        self.bsplr.current_pixel_sample_index
    }
}

#[derive(Clone, Debug, Default)]
pub struct GlobalSamplerData {
    dimension: u32,
    interval_sample_index: u64,
    array_start_dim: u32,
    array_end_dim: u32,
}

// Other algorithms for generating samples are very much not pixel-based
// but naturally generate consecutive samples that are spread across the entire image,
// visiting completelydifferent pixels in succession.
#[derive(Clone, Debug)]
pub struct GlobalSampler<T>
where
    T: IGlobalSampler,
{
    bsplr: BaseSampler,
    gsplr: GlobalSamplerData,
    splr: T,
}

pub trait IGlobalSampler: Send + Sync {
    fn get_index_for_sample(
        &mut self,
        sample_num: u64,
        bsplr: &mut BaseSampler,
        gsplr: &mut GlobalSamplerData,
    ) -> u64;
    fn sample_dimension(
        &mut self,
        index: u64,
        dimension: u32,
        bsplr: &mut BaseSampler,
        gsplr: &mut GlobalSamplerData,
    ) -> f64;
}

impl<T> GlobalSampler<T>
where
    T: IGlobalSampler,
{
    pub fn new(samples_per_pixel: u64, g: T) -> Self {
        let s = GlobalSampler {
            bsplr: BaseSampler::new(samples_per_pixel),
            gsplr: GlobalSamplerData::default(),
            splr: g,
        };

        return s;
    }
}

impl<T> RoundCount for GlobalSampler<T> where T: IGlobalSampler {}

impl<T> StartPixel for GlobalSampler<T>
where
    T: IGlobalSampler,
{
    fn start_pixel(&mut self, p: Point2i) {
        self.bsplr.start_pixel(p);
        self.gsplr.dimension = 0;
        self.gsplr.interval_sample_index =
            self.splr
                .get_index_for_sample(0, &mut self.bsplr, &mut self.gsplr);
        // Compute _arrayEndDim_ for dimensions used for array samples
        self.gsplr.array_end_dim = self.gsplr.array_end_dim
            + self.bsplr.sample_array1d.len() as u32
            + 2 * self.bsplr.sample_array2d.len() as u32;

        // Compute 1D array samples for _GlobalSampler_
        for i in 0..self.bsplr.samples1d_array_sizes.len() {
            let n_samples =
                self.bsplr.samples1d_array_sizes[i] as u64 * self.bsplr.samples_per_pixel;
            for j in 0..n_samples {
                let index = self
                    .splr
                    .get_index_for_sample(j, &mut self.bsplr, &mut self.gsplr);
                self.bsplr.sample_array1d[i][j as usize] = self.splr.sample_dimension(
                    index,
                    self.gsplr.array_start_dim + i as u32,
                    &mut self.bsplr,
                    &mut self.gsplr,
                );
            }
        }

        // Compute 2D array samples for _GlobalSampler_
        let mut dim = self.gsplr.array_start_dim + self.bsplr.samples1d_array_sizes.len() as u32;
        for i in 0..self.bsplr.samples2d_array_sizes.len() {
            let n_samples = (self.bsplr.samples2d_array_sizes[i] as u64
                * self.bsplr.samples_per_pixel) as usize;
            for j in 0..n_samples {
                let idx =
                    self.splr
                        .get_index_for_sample(j as u64, &mut self.bsplr, &mut self.gsplr);
                self.bsplr.sample_array2d[i][j].x =
                    self.splr
                        .sample_dimension(idx, dim, &mut self.bsplr, &mut self.gsplr);
                self.bsplr.sample_array2d[i][j].y =
                    self.splr
                        .sample_dimension(idx, dim + 1, &mut self.bsplr, &mut self.gsplr);
            }
            dim += 2;
        }
        assert_eq!(self.gsplr.array_end_dim, dim);
    }
}

impl<T> ISampler for GlobalSampler<T>
where
    T: IGlobalSampler + Debug,
{
    fn start_next_sample(&mut self) -> bool {
        self.gsplr.dimension = 0;
        self.gsplr.interval_sample_index = self.splr.get_index_for_sample(
            self.bsplr.current_pixel_sample_index + 1,
            &mut self.bsplr,
            &mut self.gsplr,
        );
        self.bsplr.start_next_sample()
    }

    fn set_sample_number(&mut self, sample_num: u64) -> bool {
        self.gsplr.dimension = 0;
        self.gsplr.interval_sample_index =
            self.splr
                .get_index_for_sample(sample_num, &mut self.bsplr, &mut self.gsplr);
        self.bsplr.set_sample_number(sample_num)
    }

    fn get_1d(&mut self) -> f64 {
        if self.gsplr.dimension >= self.gsplr.array_start_dim
            && self.gsplr.dimension < self.gsplr.array_end_dim
        {
            self.gsplr.dimension = self.gsplr.array_end_dim;
        }
        self.gsplr.dimension += 1;
        self.splr.sample_dimension(
            self.gsplr.interval_sample_index,
            self.gsplr.dimension - 1,
            &mut self.bsplr,
            &mut self.gsplr,
        )
    }

    fn get_2d(&mut self) -> Point2f {
        if self.gsplr.dimension + 1 > self.gsplr.array_start_dim
            && self.gsplr.dimension < self.gsplr.array_end_dim
        {
            self.gsplr.dimension = self.gsplr.array_end_dim;
        }
        let p = Point2f::new(
            self.splr.sample_dimension(
                self.gsplr.interval_sample_index,
                self.gsplr.dimension,
                &mut self.bsplr,
                &mut self.gsplr,
            ),
            self.splr.sample_dimension(
                self.gsplr.interval_sample_index,
                self.gsplr.dimension + 1,
                &mut self.bsplr,
                &mut self.gsplr,
            ),
        );
        self.gsplr.dimension += 2;
        return p;
    }

    fn request_1d_array(&mut self, n: u32) {
        self.bsplr.request_1d_array(n);
    }

    fn request_2d_array(&mut self, n: u32) {
        self.bsplr.request_2d_array(n);
    }

    fn get_1d_array(&mut self, n: u32) -> &[f64] {
        self.bsplr.get_1d_array(n)
    }

    fn get_2d_array(&mut self, n: u32) -> &[Point2f] {
        self.bsplr.get_2d_array(n)
    }

    fn samples_per_pixel(&self) -> u64 {
        self.bsplr.samples_per_pixel
    }

    fn current_sample_number(&self) -> u64 {
        self.bsplr.current_pixel_sample_index
    }
}

pub mod halton;
pub mod stratified;

use self::{halton::HaltonSampler, stratified::StratifiedSampler};
#[derive(Debug, Clone)]
pub enum Sampler {
    Halton(HaltonSampler),
    Stratified(StratifiedSampler),
}

impl StartPixel for Sampler {
    fn start_pixel(&mut self, p: Point2i) {
        match self {
            Sampler::Halton(s) => s.start_pixel(p),
            Sampler::Stratified(s) => s.start_pixel(p),
        }
    }
}

impl RoundCount for Sampler {
    fn round_count(&self, n: u32) -> u32 {
        match self {
            Sampler::Halton(s) => s.round_count(n),
            Sampler::Stratified(s) => s.round_count(n),
        }
    }
}

impl ISampler for Sampler {
    fn start_next_sample(&mut self) -> bool {
        match self {
            Sampler::Halton(s) => s.start_next_sample(),
            Sampler::Stratified(s) => s.start_next_sample(),
        }
    }

    fn set_sample_number(&mut self, sample_num: u64) -> bool {
        match self {
            Sampler::Halton(s) => s.set_sample_number(sample_num),
            Sampler::Stratified(s) => s.set_sample_number(sample_num),
        }
    }

    fn request_1d_array(&mut self, n: u32) {
        match self {
            Sampler::Halton(s) => s.request_1d_array(n),
            Sampler::Stratified(s) => s.request_1d_array(n),
        }
    }

    fn request_2d_array(&mut self, n: u32) {
        match self {
            Sampler::Halton(s) => s.request_2d_array(n),
            Sampler::Stratified(s) => s.request_2d_array(n),
        }
    }

    fn get_1d_array(&mut self, n: u32) -> &[f64] {
        match self {
            Sampler::Halton(s) => s.get_1d_array(n),
            Sampler::Stratified(s) => s.get_1d_array(n),
        }
    }

    fn get_2d_array(&mut self, n: u32) -> &[Point2f] {
        match self {
            Sampler::Halton(s) => s.get_2d_array(n),
            Sampler::Stratified(s) => s.get_2d_array(n),
        }
    }

    fn samples_per_pixel(&self) -> u64 {
        match self {
            Sampler::Halton(s) => s.samples_per_pixel(),
            Sampler::Stratified(s) => s.samples_per_pixel(),
        }
    }

    fn current_sample_number(&self) -> u64 {
        match self {
            Sampler::Halton(s) => s.current_sample_number(),
            Sampler::Stratified(s) => s.current_sample_number(),
        }
    }
}
