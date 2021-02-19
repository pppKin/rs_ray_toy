use super::*;
use crate::{
    misc::{round_up_pow2, ONE_MINUS_EPSILON},
    sampling::{latin_hypercube, shuffle},
};
use rand::prelude::ThreadRng;

#[derive(Debug, Default, Clone, Copy)]
pub struct Stratified {
    x_pixel_samples: u32,
    y_pixel_samples: u32,
    jitter_samples: bool,
}

pub type StratifiedSampler = PixelSampler<Stratified>;

impl Stratified {
    fn new(x_pixel_samples: u32, y_pixel_samples: u32, jitter_samples: bool) -> Self {
        Stratified {
            x_pixel_samples,
            y_pixel_samples,
            jitter_samples,
        }
    }
}

impl RoundCount for Stratified {
    fn round_count(&self, n: u32) -> u32 {
        round_up_pow2(n)
    }
}

impl PixelSamplerStartPixel for Stratified {
    fn start_pixel_ps(
        &mut self,
        p: Point2i,
        bsplr: &mut BaseSampler,
        psplr: &mut PixelSamplerData,
    ) {
        // Generate single stratified samples for the pixel
        for i in 0..psplr.samples1d.len() {
            stratified_sample1d(
                psplr.samples1d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                &mut psplr.rng,
                self.jitter_samples,
            );
            shuffle(
                psplr.samples1d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut psplr.rng,
            );
        }
        for i in 0..psplr.samples2d.len() {
            stratified_sample2d(
                psplr.samples2d[i].as_mut_slice(),
                self.x_pixel_samples,
                self.y_pixel_samples,
                &mut psplr.rng,
                self.jitter_samples,
            );
            shuffle(
                psplr.samples2d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut psplr.rng,
            );
        }

        // Generate arrays of stratified samples for the pixel
        for i in 0..bsplr.samples1d_array_sizes.len() {
            for j in 0..bsplr.samples_per_pixel {
                let count = bsplr.samples1d_array_sizes[i];
                stratified_sample1d(
                    &mut bsplr.sample_array1d[i][(j * count as u64) as usize..],
                    count,
                    &mut psplr.rng,
                    self.jitter_samples,
                );
                shuffle(
                    &mut bsplr.sample_array1d[i][(j * count as u64) as usize..],
                    count,
                    1,
                    &mut psplr.rng,
                );
            }
        }

        for i in 0..bsplr.samples2d_array_sizes.len() {
            for j in 0..bsplr.samples_per_pixel {
                let count = bsplr.samples2d_array_sizes[i];
                latin_hypercube(
                    &mut bsplr.sample_array2d[i][(j * count as u64) as usize..],
                    count,
                    &mut psplr.rng,
                );
            }
        }

        bsplr.start_pixel(p);
    }
}

fn stratified_sample1d(samp: &mut [f64], n_samples: u32, rng: &mut ThreadRng, jitter: bool) {
    let inv_n_samples = 1.0 / (n_samples as f64);
    for i in 0..n_samples as usize {
        let delta = if jitter { rng.gen_range(0.0, 1.0) } else { 0.5 };
        samp[i] = f64::min((i as f64 + delta) * inv_n_samples, ONE_MINUS_EPSILON);
    }
}

fn stratified_sample2d(samp: &mut [Point2f], nx: u32, ny: u32, rng: &mut ThreadRng, jitter: bool) {
    let dx = 1_f64 / nx as f64;
    let dy = 1_f64 / ny as f64;
    let mut samp_iter = samp.iter_mut();
    for y in 0..ny {
        for x in 0..nx {
            let jx = if jitter { rng.gen_range(0.0, 1.0) } else { 0.5 };
            let jy = if jitter { rng.gen_range(0.0, 1.0) } else { 0.5 };
            if let Some(p) = samp_iter.next() {
                p.x = f64::min((x as f64 + jx) * dx, ONE_MINUS_EPSILON);
                p.y = f64::min((y as f64 + jy) * dy, ONE_MINUS_EPSILON)
            }
        }
    }
}

fn create_stratified(
    n_sampled_dimensions: u32,
    x_pixel_samples: u32,
    y_pixel_samples: u32,
    jitter_samples: bool,
) -> StratifiedSampler {
    return StratifiedSampler::new(
        x_pixel_samples as u64 * y_pixel_samples as u64,
        n_sampled_dimensions,
        Stratified::new(x_pixel_samples, y_pixel_samples, jitter_samples),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::geometry::*;
    // use rand::prelude::*;
    #[test]
    fn test_stratified() {
        let mut s = create_stratified(2, 4, 4, true);
        s.start_pixel(Point2i::new(1, 2));
        println!("{}", s.get_1d());
        println!("{:?}", s.get_2d());
        println!("{}", s.get_1d());
        println!("{:?}", s.get_2d());
        println!("{}", s.get_1d());
        println!("{:?}", s.get_2d());
    }
}
