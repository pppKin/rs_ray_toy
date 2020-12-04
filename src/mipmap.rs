use std::ops::{Add, AddAssign, Mul};

use crate::{
    geometry::{Point2, Point2f, Vector2f},
    memory::BlockedArray,
    misc::{clamp_t, is_power_of_2, mod_t, round_up_pow2},
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

// MIPMap Declarations
// template <typename T>
// class MIPMap {
//   public:
//     // MIPMap Public Methods
//     MIPMap(const Point2i &resolution, const T *data, bool doTri = false,
//            Float maxAniso = 8.f, ImageWrap wrapMode = ImageWrap::Repeat);
//     int Width() const { return resolution[0]; }
//     int Height() const { return resolution[1]; }
//     int Levels() const { return pyramid.size(); }
//     const T &Texel(int level, int s, int t) const;
//     T Lookup(const Point2f &st, Float width = 0.f) const;
//     T Lookup(const Point2f &st, Vector2f dstdx, Vector2f dstdy) const;

//   private:
//     // MIPMap Private Methods
//     std::unique_ptr<ResampleWeight[]> resampleWeights(int oldRes, int newRes) {
//         CHECK_GE(newRes, oldRes);
//         std::unique_ptr<ResampleWeight[]> wt(new ResampleWeight[newRes]);
//         Float filterwidth = 2.f;
//         for (int i = 0; i < newRes; ++i) {
//             // Compute image resampling weights for _i_th texel
//             Float center = (i + .5f) * oldRes / newRes;
//             wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
//             for (int j = 0; j < 4; ++j) {
//                 Float pos = wt[i].firstTexel + j + .5f;
//                 wt[i].weight[j] = Lanczos((pos - center) / filterwidth);
//             }

//             // Normalize filter weights for texel resampling
//             Float invSumWts = 1 / (wt[i].weight[0] + wt[i].weight[1] +
//                                    wt[i].weight[2] + wt[i].weight[3]);
//             for (int j = 0; j < 4; ++j) wt[i].weight[j] *= invSumWts;
//         }
//         return wt;
//     }
//     Float clamp(Float v) { return Clamp(v, 0.f, Infinity); }
//     RGBSpectrum clamp(const RGBSpectrum &v) { return v.Clamp(0.f, Infinity); }
//     SampledSpectrum clamp(const SampledSpectrum &v) {
//         return v.Clamp(0.f, Infinity);
//     }
//     T triangle(int level, const Point2f &st) const;
//     T EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const;

//     // MIPMap Private Data
//     const bool doTrilinear;
//     const Float maxAnisotropy;
//     const ImageWrap wrapMode;
//     Point2i resolution;
//     std::vector<std::unique_ptr<BlockedArray<T>>> pyramid;
//     static PBRT_CONSTEXPR int WeightLUTSize = 128;
//     static Float weightLut[WeightLUTSize];
// };

// template <typename T>
// const T &MIPMap<T>::Texel(int level, int s, int t) const {
//     CHECK_LT(level, pyramid.size());
//     const BlockedArray<T> &l = *pyramid[level];
//     // Compute texel $(s,t)$ accounting for boundary conditions
//     switch (wrapMode) {
//     case ImageWrap::Repeat:
//         s = Mod(s, l.uSize());
//         t = Mod(t, l.vSize());
//         break;
//     case ImageWrap::Clamp:
//         s = Clamp(s, 0, l.uSize() - 1);
//         t = Clamp(t, 0, l.vSize() - 1);
//         break;
//     case ImageWrap::Black: {
//         static const T black = 0.f;
//         if (s < 0 || s >= (int)l.uSize() || t < 0 || t >= (int)l.vSize())
//             return black;
//         break;
//     }
//     }
//     return l(s, t);
// }

// template <typename T>
// T MIPMap<T>::Lookup(const Point2f &st, Float width) const {
//     ++nTrilerpLookups;
//     ProfilePhase p(Prof::TexFiltTrilerp);
//     // Compute MIPMap level for trilinear filtering
//     Float level = Levels() - 1 + Log2(std::max(width, (Float)1e-8));

//     // Perform trilinear interpolation at appropriate MIPMap level
//     if (level < 0)
//         return triangle(0, st);
//     else if (level >= Levels() - 1)
//         return Texel(Levels() - 1, 0, 0);
//     else {
//         int iLevel = std::floor(level);
//         Float delta = level - iLevel;
//         return Lerp(delta, triangle(iLevel, st), triangle(iLevel + 1, st));
//     }
// }

// template <typename T>
// T MIPMap<T>::triangle(int level, const Point2f &st) const {
//     level = Clamp(level, 0, Levels() - 1);
//     Float s = st[0] * pyramid[level]->uSize() - 0.5f;
//     Float t = st[1] * pyramid[level]->vSize() - 0.5f;
//     int s0 = std::floor(s), t0 = std::floor(t);
//     Float ds = s - s0, dt = t - t0;
//     return (1 - ds) * (1 - dt) * Texel(level, s0, t0) +
//            (1 - ds) * dt * Texel(level, s0, t0 + 1) +
//            ds * (1 - dt) * Texel(level, s0 + 1, t0) +
//            ds * dt * Texel(level, s0 + 1, t0 + 1);
// }

// template <typename T>
// T MIPMap<T>::Lookup(const Point2f &st, Vector2f dst0, Vector2f dst1) const {
//     if (doTrilinear) {
//         Float width = std::max(std::max(std::abs(dst0[0]), std::abs(dst0[1])),
//                                std::max(std::abs(dst1[0]), std::abs(dst1[1])));
//         return Lookup(st, width);
//     }
//     ++nEWALookups;
//     ProfilePhase p(Prof::TexFiltEWA);
//     // Compute ellipse minor and major axes
//     if (dst0.LengthSquared() < dst1.LengthSquared()) std::swap(dst0, dst1);
//     Float majorLength = dst0.Length();
//     Float minorLength = dst1.Length();

//     // Clamp ellipse eccentricity if too large
//     if (minorLength * maxAnisotropy < majorLength && minorLength > 0) {
//         Float scale = majorLength / (minorLength * maxAnisotropy);
//         dst1 *= scale;
//         minorLength *= scale;
//     }
//     if (minorLength == 0) return triangle(0, st);

//     // Choose level of detail for EWA lookup and perform EWA filtering
//     Float lod = std::max((Float)0, Levels() - (Float)1 + Log2(minorLength));
//     int ilod = std::floor(lod);
//     return Lerp(lod - ilod, EWA(ilod, st, dst0, dst1),
//                 EWA(ilod + 1, st, dst0, dst1));
// }

// template <typename T>
// T MIPMap<T>::EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const {
//     if (level >= Levels()) return Texel(Levels() - 1, 0, 0);
//     // Convert EWA coordinates to appropriate scale for level
//     st[0] = st[0] * pyramid[level]->uSize() - 0.5f;
//     st[1] = st[1] * pyramid[level]->vSize() - 0.5f;
//     dst0[0] *= pyramid[level]->uSize();
//     dst0[1] *= pyramid[level]->vSize();
//     dst1[0] *= pyramid[level]->uSize();
//     dst1[1] *= pyramid[level]->vSize();

//     // Compute ellipse coefficients to bound EWA filter region
//     Float A = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1;
//     Float B = -2 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
//     Float C = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1;
//     Float invF = 1 / (A * C - B * B * 0.25f);
//     A *= invF;
//     B *= invF;
//     C *= invF;

//     // Compute the ellipse's $(s,t)$ bounding box in texture space
//     Float det = -B * B + 4 * A * C;
//     Float invDet = 1 / det;
//     Float uSqrt = std::sqrt(det * C), vSqrt = std::sqrt(A * det);
//     int s0 = std::ceil(st[0] - 2 * invDet * uSqrt);
//     int s1 = std::floor(st[0] + 2 * invDet * uSqrt);
//     int t0 = std::ceil(st[1] - 2 * invDet * vSqrt);
//     int t1 = std::floor(st[1] + 2 * invDet * vSqrt);

//     // Scan over ellipse bound and compute quadratic equation
//     T sum(0.f);
//     Float sumWts = 0;
//     for (int it = t0; it <= t1; ++it) {
//         Float tt = it - st[1];
//         for (int is = s0; is <= s1; ++is) {
//             Float ss = is - st[0];
//             // Compute squared radius and filter texel if inside ellipse
//             Float r2 = A * ss * ss + B * ss * tt + C * tt * tt;
//             if (r2 < 1) {
//                 int index =
//                     std::min((int)(r2 * WeightLUTSize), WeightLUTSize - 1);
//                 Float weight = weightLut[index];
//                 sum += Texel(level, is, it) * weight;
//                 sumWts += weight;
//             }
//         }
//     }
//     return sum / sumWts;
// }

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
        + From<f64>,
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
    pub fn texel(&self, level: usize, s: f64, t: f64) -> T {
        todo!();
    }
    pub fn lookup_w(&self, st: &Point2f, width: f64) -> T {
        todo!();
    }
    pub fn lookup_d(&self, st: &Point2f, dstdx: &Vector2f, dstdy: &Vector2f) -> T {
        todo!();
    }
    // clamp
    fn triangle(&self, level: usize, st: &Point2f) -> T {
        todo!();
    }
    fn ewa(&self, level: usize, st: &Point2f, dst0: &Vector2f, dst1: &Vector2f) -> T {
        todo!();
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
                    tmp[(s, t)] = (mipmap.texel(i - 1, 2.0 * s as f64, 2.0 * t as f64)
                        + mipmap.texel(i - 1, 2.0 * s as f64 + 1.0, 2.0 * t as f64)
                        + mipmap.texel(i - 1, 2.0 * s as f64, 2.0 * t as f64 + 1.0)
                        + mipmap.texel(i - 1, 2.0 * s as f64 + 1.0, 2.0 * t as f64 + 1.0))
                        * 0.25;
                }
            }
            mipmap.pyramid.push(tmp);
        }
        mipmap
    }
}
