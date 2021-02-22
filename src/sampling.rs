use std::{f64::consts::PI, sync::Arc};

use crate::{
    geometry::{Point2f, Vector2f, Vector3f},
    misc::{clamp_t, INV_4_PI, INV_PI, ONE_MINUS_EPSILON, PI_OVER_2, PI_OVER_4},
};
use rand::{prelude::ThreadRng, Rng};

#[derive(Debug, Default, Clone)]
pub struct Distribution1D {
    pub func: Vec<f64>,
    pub cdf: Vec<f64>,
    pub func_int: f64,
}

impl Distribution1D {
    pub fn new(f: Vec<f64>) -> Self {
        let n: usize = f.len();
        // compute integral of step function at $x_i$
        let mut cdf: Vec<f64> = Vec::with_capacity(n + 1);
        cdf.push(0.0 as f64);
        for i in 1..=n {
            let previous: f64 = cdf[i - 1];
            cdf.push(previous + f[i - 1] / n as f64);
        }
        // transform step function integral into CDF
        let func_int: f64 = cdf[n];
        if func_int == 0.0 as f64 {
            for (i, item) in cdf.iter_mut().enumerate().skip(1).take(n) {
                *item = i as f64 / n as f64;
            }
        } else {
            for item in cdf.iter_mut().skip(1).take(n) {
                *item /= func_int;
            }
        }
        Distribution1D {
            func: f,
            cdf,
            func_int,
        }
    }
    pub fn count(&self) -> usize {
        self.func.len()
    }
    pub fn sample_continuous(&self, u: f64, pdf: Option<&mut f64>, off: Option<&mut usize>) -> f64 {
        // find surrounding CDF segments and _offset_
        // int offset = find_interval((int)cdf.size(),
        //                           [&](int index) { return cdf[index] <= u; });

        // see pbrt.h (int FindInterval(int size, const Predicate &pred) {...})
        let mut first: usize = 0;
        let mut len: usize = self.cdf.len();
        while len > 0 as usize {
            let half: usize = len >> 1;
            let middle: usize = first + half;
            // bisect range based on value of _pred_ at _middle_
            if self.cdf[middle] <= u {
                first = middle + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        let offset: usize = clamp_t(first - 1, 0, self.cdf.len() - 2) as usize;
        if let Some(off_ref) = off {
            *off_ref = offset;
        }
        // compute offset along CDF segment
        let mut du: f64 = u - self.cdf[offset];
        if (self.cdf[offset + 1] - self.cdf[offset]) > 0.0 as f64 {
            assert!(self.cdf[offset + 1] > self.cdf[offset]);
            du /= self.cdf[offset + 1] - self.cdf[offset];
        }
        assert!(!du.is_nan());
        // compute PDF for sampled offset
        if let Some(value) = pdf {
            if self.func_int > 0.0 as f64 {
                *value = self.func[offset] / self.func_int;
            } else {
                *value = 0.0;
            }
        }
        // return $x\in{}[0,1)$ corresponding to sample
        (offset as f64 + du) / self.count() as f64
    }
    pub fn sample_discrete(
        &self,
        u: f64,
        pdf: Option<&mut f64>, /* TODO: f64 *uRemapped = nullptr */
    ) -> usize {
        // find surrounding CDF segments and _offset_
        // let offset: usize = find_interval(cdf.size(),
        //                           [&](int index) { return cdf[index] <= u; });

        // see pbrt.h (int FindInterval(int size, const Predicate &pred) {...})
        let mut first: usize = 0;
        let mut len: usize = self.cdf.len();
        while len > 0 as usize {
            let half: usize = len >> 1;
            let middle: usize = first + half;
            // bisect range based on value of _pred_ at _middle_
            if self.cdf[middle] <= u {
                first = middle + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        let offset: usize = clamp_t(first - 1, 0, self.cdf.len() - 2) as usize;
        if let Some(value) = pdf {
            if self.func_int > 0.0 as f64 {
                *value = self.func[offset] / (self.func_int * self.func.len() as f64);
            } else {
                *value = 0.0;
            }
        }
        // TODO: if (uRemapped)
        //     *uRemapped = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
        // if (uRemapped) CHECK(*uRemapped >= 0.f && *uRemapped <= 1.f);
        offset
    }
    pub fn discrete_pdf(&self, index: usize) -> f64 {
        assert!(index < self.func.len());
        self.func[index] / (self.func_int * self.func.len() as f64)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Distribution2D {
    pub p_conditional_v: Vec<Arc<Distribution1D>>,
    pub p_marginal: Arc<Distribution1D>,
}

impl Distribution2D {
    pub fn new(func: Vec<f64>, nu: usize, nv: usize) -> Self {
        let mut p_conditional_v: Vec<Arc<Distribution1D>> = Vec::with_capacity(nv as usize);
        for v in 0..nv {
            // compute conditional sampling distribution for $\tilde{v}$
            let f: Vec<f64> = func[(v * nu) as usize..((v + 1) * nu) as usize].to_vec();
            p_conditional_v.push(Arc::new(Distribution1D::new(f)));
        }
        // compute marginal sampling distribution $p[\tilde{v}]$
        let mut marginal_func: Vec<f64> = Vec::with_capacity(nv as usize);
        for v in 0..nv {
            marginal_func.push(p_conditional_v[v as usize].func_int);
        }
        let p_marginal: Arc<Distribution1D> = Arc::new(Distribution1D::new(marginal_func));
        Distribution2D {
            p_conditional_v,
            p_marginal,
        }
    }
    pub fn sample_continuous(&self, u: Point2f, pdf: &mut f64) -> Point2f {
        let mut pdfs: [f64; 2] = [0.0 as f64; 2];
        let mut v: usize = 0_usize;
        let d1: f64 = self
            .p_marginal
            .sample_continuous(u[1], Some(&mut (pdfs[1])), Some(&mut v));
        let d0: f64 = self.p_conditional_v[v].sample_continuous(u[0], Some(&mut (pdfs[0])), None);
        *pdf = pdfs[0] * pdfs[1];
        Point2f { x: d0, y: d1 }
    }
    pub fn pdf(&self, p: Point2f) -> f64 {
        let iu: usize = clamp_t(
            (p[0] * self.p_conditional_v[0].count() as f64) as usize,
            0_usize,
            self.p_conditional_v[0].count() - 1_usize,
        );
        let iv: usize = clamp_t(
            (p[1] * self.p_marginal.count() as f64) as usize,
            0_usize,
            self.p_marginal.count() - 1_usize,
        );
        self.p_conditional_v[iv].func[iu] / self.p_marginal.func_int
    }
}

/// Randomly permute an array of *count* sample values, each of which
/// has *n_dimensions* dimensions.
pub fn shuffle<T>(samp: &mut [T], count: u32, n_dimensions: u32, rng: &mut ThreadRng) {
    for i in 0..count {
        let other = i + rng.gen_range(0, count - i);
        for j in 0..n_dimensions {
            samp.swap(
                (n_dimensions * i + j) as usize,
                (n_dimensions * other + j) as usize,
            );
        }
    }
}

pub fn latin_hypercube(samples: &mut [Point2f], n_samples: u32, rng: &mut ThreadRng) {
    let n_dim: usize = 2;
    // generate LHS samples along diagonal
    let inv_n_samples: f64 = 1.0 as f64 / n_samples as f64;
    for i in 0..n_samples {
        for j in 0..n_dim {
            let sj: f64 = (i as f64 + (rng.gen_range(0.0, ONE_MINUS_EPSILON))) * inv_n_samples;
            if j == 0 {
                samples[i as usize].x = sj.min(ONE_MINUS_EPSILON);
            } else {
                samples[i as usize].y = sj.min(ONE_MINUS_EPSILON);
            }
        }
    }
    // permute LHS samples in each dimension
    for i in 0..n_dim {
        for j in 0..n_samples {
            let other: u32 = j as u32 + rng.gen_range(0, (n_samples - j) as u32);
            if i == 0 {
                let tmp = samples[j as usize].x;
                samples[j as usize].x = samples[other as usize].x;
                samples[other as usize].x = tmp;
            } else {
                let tmp = samples[j as usize].y;
                samples[j as usize].y = samples[other as usize].y;
                samples[other as usize].y = tmp;
            }
            // samples.swap(
            //     (n_dim * j + i) as usize,
            //     (n_dim * other + i) as usize,
            // );
        }
    }
}

/// Uniformly sample rays in a full sphere. Choose a direction.
pub fn uniform_sample_sphere(u: Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = (0.0 as f64).max(1.0 - z * z).sqrt();
    let phi = 2.0 * PI * u[1];
    Vector3f {
        x: r * phi.cos(),
        y: r * phi.sin(),
        z,
    }
}

/// Uniformly sample rays in a hemisphere. Choose a direction.
pub fn uniform_sample_hemisphere(u: &Point2f) -> Vector3f {
    let z: f64 = u[0];
    let r: f64 = (0.0 as f64).max(1.0 as f64 - z * z).sqrt();
    let phi: f64 = 2.0 as f64 * PI * u[1];
    Vector3f {
        x: r * phi.cos(),
        y: r * phi.sin(),
        z,
    }
}

/// Uniformly sample rays in a hemisphere. Probability density
/// function (PDF).
#[inline]
pub fn uniform_hemisphere_pdf() -> f64 {
    1.0 / (2.0 * PI)
}

/// Probability density function (PDF) of a sphere.
pub fn uniform_sphere_pdf() -> f64 {
    INV_4_PI
}

/// Cosine-weighted hemisphere sampling using Malley's method.
#[inline]
pub fn cosine_sample_hemisphere(u: Point2f) -> Vector3f {
    let d: Point2f = concentric_sample_disk(u);
    let z: f64 = (0.0 as f64).max(1.0 - d.x * d.x - d.y * d.y).sqrt();
    Vector3f { x: d.x, y: d.y, z }
}

/// Returns a weight of cos_theta / PI.
pub fn cosine_hemisphere_pdf(cos_theta: f64) -> f64 {
    cos_theta * INV_PI
}

/// Uniformly distribute samples over a unit disk.
pub fn concentric_sample_disk(u: Point2f) -> Point2f {
    // map uniform random numbers to $[-1,1]^2$
    let u_offset: Point2f = u * 2.0 - Vector2f { x: 1.0, y: 1.0 };
    // handle degeneracy at the origin
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::default();
    }
    // apply concentric mapping to point
    let theta: f64;
    let r: f64;
    if u_offset.x.abs() > u_offset.y.abs() {
        r = u_offset.x;
        theta = PI_OVER_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y);
    }
    Point2f {
        x: theta.cos(),
        y: theta.sin(),
    } * r
}

/// Uniformly sample rays in a cone of directions. Probability density
/// function (PDF).
pub fn uniform_cone_pdf(cos_theta_max: f64) -> f64 {
    1.0 / (2.0 * PI * (1.0 - cos_theta_max))
}

/// Samples in a cone of directions about the (0, 0, 1) axis.
pub fn uniform_sample_cone(u: Point2f, cos_theta_max: f64) -> Vector3f {
    let cos_theta: f64 = (1.0 - u[0]) + u[0] * cos_theta_max;
    let sin_theta: f64 = (1.0 - cos_theta * cos_theta).sqrt();
    let phi: f64 = u[1] * 2.0 * PI;
    Vector3f {
        x: phi.cos() * sin_theta,
        y: phi.sin() * sin_theta,
        z: cos_theta,
    }
}

#[inline]
pub fn power_heuristic(nf: i32, f_pdf: f64, ng: i32, g_pdf: f64) -> f64 {
    let f = nf as f64 * f_pdf;
    let g = ng as f64 * g_pdf;
    (f * f) / (f * f + g * g)
}
