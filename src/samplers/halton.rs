use super::*;
use crate::{lowdiscrepancy::*, misc::mod_t};

pub const K_MAX_RESOLUTION: i64 = 128;

#[derive(Debug, Default, Clone)]
pub struct Halton {
    // HaltonSampler Private Data
    radical_inverse_permutations: Vec<u16>,
    base_scales: Point2i,
    base_exponents: Point2i,
    sample_stride: u64,
    mult_inverse: [u64; 2],
    pixel_for_offset: Point2i,
    offset_for_current_pixel: u64,
    // Added after book publication: force all image samples to be at the
    // center of the pixel area.
    sample_at_pixel_center: bool,
}
pub type HaltonSampler = GlobalSampler<Halton>;

impl Halton {
    fn new(sample_bounds: &Bounds2i, sample_at_pixel_center: bool) -> Self {
        // Generate random digit permutations for Halton sampler
        let radical_inverse_permutations = compute_radical_inverse_permutations();

        // Find radical inverse base scales and exponents that cover sampling area
        let mut base_scales = Point2i::default();
        let mut base_exponents = Point2i::default();
        let res = sample_bounds.p_max - sample_bounds.p_min;
        for i in 0..2 {
            let base = if i == 0 { 2 } else { 3 };
            let mut scale = 1;
            let mut exp = 0;
            while scale < i64::min(res[i as u8], K_MAX_RESOLUTION) {
                scale *= base;
                exp += 1;
            }
            base_scales[i] = scale;
            base_exponents[i] = exp;
        }

        // Compute stride in samples for visiting each pixel area
        let sample_stride = (base_scales[0] * base_scales[1]) as u64;

        // Compute multiplicative inverses for _baseScales_
        let mut mult_inverse = [0, 0];
        mult_inverse[0] = multiplicative_inverse(base_scales[1] as u64, base_scales[0] as u64);
        mult_inverse[1] = multiplicative_inverse(base_scales[0] as u64, base_scales[1] as u64);

        Halton {
            radical_inverse_permutations,
            base_scales,
            base_exponents,
            sample_stride,
            mult_inverse,
            pixel_for_offset: Point2i::default(),
            offset_for_current_pixel: 0,
            sample_at_pixel_center,
        }
    }

    fn permutation_for_dimension(&mut self, dim: usize) -> &[u16] {
        if dim >= PRIME_TABLE_SIZE {
            panic!(
                "HaltonSampler can only sample {} dimensions.",
                PRIME_TABLE_SIZE
            );
        }
        return &self.radical_inverse_permutations[PRIME_SUMS[dim] as usize..];
    }
}

impl IGlobalSampler for Halton {
    fn get_index_for_sample(
        &mut self,
        sample_num: u64,
        bsplr: &mut BaseSampler,
        _gsplr: &mut GlobalSamplerData,
    ) -> u64 {
        // Compute Halton sample offset for _currentPixel_
        if bsplr.current_pixel != self.pixel_for_offset {
            self.offset_for_current_pixel = 0;
            if self.sample_stride > 1 {
                let pm = Point2i::new(
                    mod_t(bsplr.current_pixel[0], K_MAX_RESOLUTION),
                    mod_t(bsplr.current_pixel[1], K_MAX_RESOLUTION),
                );
                for i in 0..2 {
                    let dim_offset = if i == 0 {
                        inverse_radical_inverse(2, pm[i] as u64, self.base_exponents[1] as u64)
                    } else {
                        inverse_radical_inverse(3, pm[i] as u64, self.base_exponents[i] as u64)
                    };

                    self.offset_for_current_pixel += dim_offset
                        * (self.sample_stride / self.base_scales[i] as u64)
                        * self.mult_inverse[i as usize];
                }
                self.offset_for_current_pixel %= self.sample_stride;
            }
            self.pixel_for_offset = bsplr.current_pixel;
        }
        self.offset_for_current_pixel + sample_num * self.sample_stride
    }

    fn sample_dimension(
        &mut self,
        index: u64,
        dimension: u32,
        _bsplr: &mut BaseSampler,
        _gsplr: &mut GlobalSamplerData,
    ) -> f64 {
        if self.sample_at_pixel_center && (dimension == 0 || dimension == 1) {
            return 0.5;
        }
        if dimension == 0 {
            return radical_inverse(dimension as usize, index >> self.base_exponents[0]);
        } else if dimension == 1 {
            return radical_inverse(dimension as usize, index / self.base_scales[1] as u64);
        } else {
            return scrambled_radical_inverse(
                dimension as usize,
                index,
                self.permutation_for_dimension(dimension as usize),
            );
        }
    }
}

fn extended_gcd(a: u64, b: u64, x: &mut i64, y: &mut i64) {
    if b == 0 {
        *x = 1;
        *y = 1;
    } else {
        let d = a / b;
        let mut xp = 0_i64;
        let mut yp = 0_i64;
        extended_gcd(b, a % b, &mut xp, &mut yp);
        *x = yp;
        *y = xp - ((d as i64) * yp);
    }
}

fn multiplicative_inverse(a: u64, n: u64) -> u64 {
    let mut x = 0_i64;
    let mut y = 0_i64;
    extended_gcd(a, n, &mut x, &mut y);
    mod_t(x as u64, n)
}

fn create_halton(
    samples_per_pixel: u64,
    sample_bounds: &Bounds2i,
    sample_at_pixel_center: bool,
) -> HaltonSampler {
    let g = Halton::new(sample_bounds, sample_at_pixel_center);
    HaltonSampler::new(samples_per_pixel, g)
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::geometry::*;
    // use rand::prelude::*;
    #[test]
    fn test_halton() {
        let mut h: Box<dyn Sampler> = Box::new(create_halton(
            4,
            &Bounds2i::new(Point2i::new(-5, -5), Point2i::new(5, 5)),
            false,
        ));
        h.start_pixel(Point2i::new(1, 2));
        for _i in 0..4 {
            // let oneds = h.get_1d();
            // let twods = h.get_2d();
            // println!("{}, {:?}", oneds, twods);
            let cs = h.get_camerasample(&Point2i::new(1, 2));
            println!("{:?}", cs);
        }
    }
}
