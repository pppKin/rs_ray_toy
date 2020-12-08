use std::ops::{Add, AddAssign, Div, Mul};

use crate::{
    geometry::{Point2, Point2f, Vector2f},
    memory::BlockedArray,
    misc::{clamp_t, is_power_of_2, lerp, mod_t, round_up_pow2},
    texture::lanczos,
};
use lazy_static::*;

pub const WEIGHT_LUT_SIZE: usize = 128;
lazy_static! {
    pub static ref WEIGHT_LUT: [f64; WEIGHT_LUT_SIZE] = {
        let mut wl = [0.0; WEIGHT_LUT_SIZE];
        for i in 0..WEIGHT_LUT_SIZE {
            let alpha: f64 = 2.0 as f64;
            let r2: f64 = i as f64 / (WEIGHT_LUT_SIZE - 1) as f64;
            wl[i] = (-alpha * r2).exp() - (-alpha).exp();
        }
        wl
    };
}

fn resample_weights(old_res: usize, new_res: usize) -> Vec<ResampleWeight> {
    assert!(new_res >= old_res);
    let mut wt = Vec::with_capacity(new_res);
    let filter_width = 2.0;
    for i in 0..new_res {
        // Compute image resampling weights for _i_th texel
        let center = (i as f64 + 0.5) * old_res as f64 / new_res as f64;
        let first_texel = (center - filter_width + 0.5).floor() as usize;
        let mut weight = [0.0; 4];
        for j in 0..4 {
            let pos = (first_texel + j) as f64 + 0.5;
            weight[j as usize] = lanczos((pos - center) / filter_width, 2.0);
        }

        // Normalize filter weights for texel resampling
        let inv_sum_wts = 1.0 / (weight[0] + weight[1] + weight[2] + weight[4]);
        for j in 0..4 {
            weight[j] *= inv_sum_wts;
        }
        wt.push(ResampleWeight::new(first_texel, weight));
    }
    wt
}

#[derive(Debug, PartialEq, PartialOrd)]
pub enum ImageWrap {
    Repeat,
    Black,
    Clamp,
}

#[derive(Debug, Default, Copy, Clone)]
struct ResampleWeight {
    first_texel: usize,
    weight: [f64; 4],
}

impl ResampleWeight {
    fn new(first_texel: usize, weight: [f64; 4]) -> Self {
        Self {
            first_texel,
            weight,
        }
    }
}

#[derive(Debug)]
pub struct MIPMap<T: Clone + From<f64> + Copy> {
    do_trilinear: bool,
    max_anisotropy: f64,
    wrap_mode: ImageWrap,
    resolution: Point2<usize>,
    pyramid: Vec<BlockedArray<T>>,
}

impl<T: Clone + From<f64> + Copy> MIPMap<T> {
    pub fn new(
        do_trilinear: bool,
        max_anisotropy: f64,
        wrap_mode: ImageWrap,
        resolution: Point2<usize>,
        pyramid: Vec<BlockedArray<T>>,
    ) -> Self {
        Self {
            do_trilinear,
            max_anisotropy,
            wrap_mode,
            resolution,
            pyramid,
        }
    }
}

impl<T> MIPMap<T>
where
    T: Copy
        + Clone
        + Default
        + Add<T, Output = T>
        + AddAssign
        + Mul<f64, Output = T>
        + PartialOrd
        + From<f64>
        + From<u8>
        + Div<f64, Output = T>,
{
    pub fn width(&self) -> usize {
        self.resolution[0]
    }
    pub fn height(&self) -> usize {
        self.resolution[1]
    }
    pub fn levels(&self) -> usize {
        self.pyramid.len()
    }
    pub fn texel(&self, level: usize, s: usize, t: usize) -> T {
        //     CHECK_LT(level, pyramid.size());
        assert!(level < self.pyramid.len());
        //     const BlockedArray<T> &l = *pyramid[level];
        let l = &self.pyramid[level];
        let mut tmp_s = 0;
        let mut tmp_t = 0;
        // Compute texel $(s,t)$ accounting for boundary conditions
        match self.wrap_mode {
            ImageWrap::Repeat => {
                tmp_s = mod_t(s, l.u_size());
                tmp_t = mod_t(t, l.v_size());
            }
            ImageWrap::Black => {
                if s >= l.u_size() || t >= l.v_size() {
                    return T::from(0.0);
                }
            }
            ImageWrap::Clamp => {
                tmp_s = clamp_t(s, 0, l.u_size());
                tmp_t = clamp_t(t, 0, l.v_size());
            }
        }
        l[(tmp_s, tmp_t)]
    }
    pub fn lookup_w(&self, st: &Point2f, width: f64) -> T {
        // Compute MIPMap level for trilinear filtering
        let level = self.levels() as f64 - 1.0 + width.max(1e-8).log2();
        // Perform trilinear interpolation at appropriate MIPMap level
        if level < 0.0 {
            return self.triangle(0, st);
        } else if level >= (self.levels() - 1) as f64 {
            return self.texel(self.levels() - 1, 0, 0);
        } else {
            let i_level = level.floor() as usize;
            let delta = level.fract();
            return lerp(
                delta,
                self.triangle(i_level, st),
                self.triangle(i_level + 1, st),
            );
        }
    }
    pub fn lookup_d(&self, st: &Point2f, dstdx: &Vector2f, dstdy: &Vector2f) -> T {
        if self.do_trilinear {
            let width = dstdx.abs().max_comp().max(dstdy.abs().max_comp());
            return self.lookup_w(st, width);
        }

        // Compute ellipse minor and major axes
        let dst0;
        let mut dst1;
        if dstdx.length_squared() < dstdy.length_squared() {
            dst0 = *dstdy;
            dst1 = *dstdx;
        } else {
            dst0 = *dstdx;
            dst1 = *dstdy;
        }

        let major_length = dst0.length();
        let mut minor_length = dst1.length();
        // Clamp ellipse eccentricity if too large
        if minor_length * self.max_anisotropy < major_length && minor_length > 0.0 {
            let scale = major_length / (minor_length * self.max_anisotropy);
            dst1 *= scale;
            minor_length *= scale;
        }
        if minor_length == 0.0 {
            return self.triangle(0, st);
        }

        // Choose level of detail for EWA lookup and perform EWA filtering
        let lod = ((self.levels() - 1) as f64 + minor_length.log2()).max(0.0);
        let i_lod = lod.floor() as usize;
        lerp(
            lod.fract(),
            self.ewa(i_lod, st, &dst0, &dst1),
            self.ewa(i_lod + 1, st, &dst0, &dst1),
        )
    }
    fn triangle(&self, level: usize, st: &Point2f) -> T {
        let level = clamp_t(level, 0, self.levels() - 1);
        let s = st[0] * self.pyramid[level].u_size() as f64 - 0.5;
        let t = st[1] * self.pyramid[level].v_size() as f64 - 0.5;
        let s0 = s.floor() as usize;
        let t0 = t.floor() as usize;
        let ds = s.fract();
        let dt = t.fract();
        self.texel(level, s0, t0) * (1.0 - ds) * (1.0 - dt)
            + self.texel(level, s0, t0 + 1) * (1.0 - ds) * dt
            + self.texel(level, s0 + 1, t0) * ds * (1.0 - dt)
            + self.texel(level, s0 + 1, t0 + 1) * ds * dt
    }
    fn ewa(&self, level: usize, st: &Point2f, dstdx: &Vector2f, dstdy: &Vector2f) -> T {
        if level > self.levels() {
            return self.texel(self.levels() - 1, 0, 0);
        }
        // Convert EWA coordinates to appropriate scale for level
        let st = Point2f::new(
            st[0] * self.pyramid[level].u_size() as f64 - 0.5,
            st[1] * self.pyramid[level].v_size() as f64 - 0.5,
        );
        let dst0 = Vector2f::new(
            dstdx[0] * self.pyramid[level].u_size() as f64,
            dstdx[1] * self.pyramid[level].v_size() as f64,
        );
        let dst1 = Vector2f::new(
            dstdy[0] * self.pyramid[level].u_size() as f64,
            dstdy[1] * self.pyramid[level].v_size() as f64,
        );

        // Compute ellipse coefficients to bound EWA filter region
        let mut a = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1.0;
        let mut b = -2.0 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
        let mut c = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1.0;
        let inv_f = 1.0 / (a * c - b * b * 0.25);
        a *= inv_f;
        b *= inv_f;
        c *= inv_f;

        // Compute the ellipse's $(s,t)$ bounding box in texture space
        let det = -b * b + 4.0 * a * c;
        let inv_det = 1.0 / det;
        let u_sqrt = (det * c).sqrt();
        let v_sqrt = (det * a).sqrt();

        let s0 = (st[0] - 2.0 * inv_det * u_sqrt).ceil() as usize;
        let s1 = (st[0] + 2.0 * inv_det * u_sqrt).floor() as usize;
        let t0 = (st[1] - 2.0 * inv_det * v_sqrt).ceil() as usize;
        let t1 = (st[1] + 2.0 * inv_det * v_sqrt).floor() as usize;

        // Scan over ellipse bound and compute quadratic equation
        let mut sum = T::from(0.0);
        let mut sum_wts = 0.0;
        for it in t0..=t1 {
            let tt = it as f64 - st[0];
            for is in s0..=s1 {
                let ss = is as f64 - st[0];
                // Compute squared radius and filter texel if inside ellipse
                let r2 = a * ss * ss + b * ss * tt + c * tt * tt;
                if r2 < 1.0 {
                    let index =
                        (r2 * WEIGHT_LUT_SIZE as f64).min((WEIGHT_LUT_SIZE - 1) as f64) as usize;
                    let weight = WEIGHT_LUT[index];
                    sum += self.texel(level, is, it) * weight;
                    sum_wts += weight;
                }
            }
        }
        sum / sum_wts
    }
    fn create(
        res: Point2<usize>,
        img: &[T],
        do_trilinear: bool,
        max_anisotropy: f64,
        wrap_mode: ImageWrap,
    ) -> Self {
        let mut resampled_image = vec![];
        let resolution;
        if !is_power_of_2(res[0]) || !is_power_of_2(res[1]) {
            // Resample image to power-of-two resolution
            let res_pow2 = Point2::<usize>::new(round_up_pow2(res[0]), round_up_pow2(res[1]));
            // Resample image in $s$ direction
            let sweights = resample_weights(res[0], res_pow2[0]);
            resampled_image.resize(res_pow2[0] * res_pow2[1], T::default());
            // Apply _sWeights_ to zoom in $s$ direction
            // TODO: ParallelFor -> rayon
            for t in 0..res[0] {
                for s in 0..res_pow2[0] {
                    // Compute texel $(s,t)$ in $s$-zoomed image
                    let tmp_idx = t * res_pow2[0] + s;
                    resampled_image[tmp_idx] = T::default();

                    for j in 0..4 {
                        let mut orig_s = sweights[s].first_texel + j;
                        match wrap_mode {
                            ImageWrap::Repeat => {
                                orig_s = mod_t(orig_s, res[0]);
                            }
                            ImageWrap::Clamp => {
                                orig_s = clamp_t(orig_s, 0, res[0] - 1);
                            }
                            ImageWrap::Black => {}
                        }
                        if orig_s < res[0] {
                            resampled_image[tmp_idx] +=
                                img[t * res[0] + orig_s] * sweights[s].weight[j];
                        }
                    }
                }
            }

            // Resample image in $t$ direction
            let tweights = resample_weights(res[1], res_pow2[1]);
            // let mut resample_bufs = vec![vec![T::default(); res_pow2[1]];4];
            // TODO: ParallelFor -> rayon
            for s in 0..res_pow2[0] {
                let mut work_data = vec![T::default(); res_pow2[1]];
                for t in 0..res_pow2[1] {
                    for j in 0..4 {
                        let mut offset = tweights[t].first_texel + j;
                        match wrap_mode {
                            ImageWrap::Repeat => {
                                offset = mod_t(offset, res[1]);
                            }
                            ImageWrap::Black => {}
                            ImageWrap::Clamp => {
                                offset = clamp_t(offset, 0, res[1] - 1);
                            }
                        }
                        if offset < res[1] {
                            work_data[t] +=
                                resampled_image[offset * res_pow2[0] + s] * tweights[t].weight[j];
                        }
                    }
                }
                for t in 0..res_pow2[1] {
                    resampled_image[t * res_pow2[0] + s] =
                        clamp_t(work_data[t], T::from(0.0), T::from(std::f64::INFINITY));
                }
            }
            resolution = res_pow2;
        } else {
            resolution = res;
        }

        // Initialize levels of MIPMap from image
        let n_levels = 1 + (resolution[0].max(resolution[1]) as f64).log2() as usize;
        let mut pyramid: Vec<BlockedArray<T>> = vec![];
        // Initialize most detailed level of MIPMap
        pyramid.push(BlockedArray::new(
            if resampled_image.len() > 0 {
                Some(resampled_image)
            } else {
                Some(Vec::from(img))
            },
            resolution[0],
            resolution[1],
        ));
        let mut mipmap = MIPMap::new(do_trilinear, max_anisotropy, wrap_mode, resolution, pyramid);
        for i in 1..n_levels {
            // Initialize $i$th MIPMap level from $i-1$st level
            let s_res = (mipmap.pyramid[i - 1].u_size() / 2).max(1);
            let t_res = (mipmap.pyramid[i - 1].v_size() / 2).max(1);
            let mut tmp = BlockedArray::new(None, s_res, t_res);
            // Filter four texels from finer level of pyramid
            // TODO: ParallelFor
            for t in 0..t_res {
                for s in 0..s_res {
                    tmp[(s, t)] = (mipmap.texel(i - 1, 2 * s, 2 * t)
                        + mipmap.texel(i - 1, 2 * s + 1, 2 * t)
                        + mipmap.texel(i - 1, 2 * s, 2 * t + 1)
                        + mipmap.texel(i - 1, 2 * s + 1, 2 * t + 1))
                        * 0.25;
                }
            }
            mipmap.pyramid.push(tmp);
        }
        mipmap
    }
}
