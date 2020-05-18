use crate::{
    geometry::{abs_dot3, spherical_direction, Point2f, Vector3f},
    misc::{erf, erf_inv},
    reflection::{
        abs_cos_theta, cos_2_phi, cos_2_theta, cos_phi, cos_theta, same_hemisphere, sin_2_phi,
        sin_phi, tan_2_theta, tan_theta,
    },
};

use std::f64::consts::PI;

pub fn roughness_to_alpha(roughness: f64) -> f64 {
    let roughness = roughness.max(1e-3);
    let x = roughness.ln(); // natural (base e) logarithm
    1.62142
        + 0.819_955 * x
        + 0.1734 * x * x
        + 0.017_120_1 * x * x * x
        + 0.000_640_711 * x * x * x * x
}

pub trait MicrofacetDistribution: std::fmt::Debug {
    fn d(&self, wh: &Vector3f) -> f64;
    fn lambda(&self, w: &Vector3f) -> f64;
    fn g1(&self, w: &Vector3f) -> f64 {
        1.0 / (1.0 + self.lambda(w))
    }
    fn g(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }
    fn pdf(&self, wo: &Vector3f, wh: &Vector3f) -> f64 {
        if self.get_sample_visible_area() {
            self.d(wh) * self.g1(wo) * abs_dot3(wo, wh) / abs_cos_theta(wo)
        } else {
            self.d(wh) * abs_cos_theta(wh)
        }
    }
    fn sample_wh(&self, wo: &Vector3f, u: Point2f) -> Vector3f;
    fn get_sample_visible_area(&self) -> bool;
}

#[derive(Debug, Default, Copy, Clone)]
pub struct BeckmannDistribution {
    alpha_x: f64,
    alpha_y: f64,
    // inherited from class MicrofacetDistribution (see microfacet.h)
    sample_visible_area: bool,
}

impl BeckmannDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64, sample_visible_area: bool) -> Self {
        BeckmannDistribution {
            alpha_x,
            alpha_y,
            sample_visible_area,
        }
    }
}

fn beckmann_sample_11(cos_theta_i: f64, u1: f64, u2: f64, slope_x: &mut f64, slope_y: &mut f64) {
    // special case (normal incidence)
    if cos_theta_i > 0.9999 {
        let r = (-((1.0 - u1).ln())).sqrt();
        let phi = 2.0 * PI * u2;
        *slope_x = r * phi.cos();
        *slope_y = r * phi.sin();
        return;
    }

    // The original inversion routine from the paper contained
    // discontinuities, which causes issues for QMC integration and
    // techniques like Kelemen-style MLT. The following code performs
    // a numerical inversion with better behavior
    let sin_theta_i = (0_f64).max(1.0 - cos_theta_i * cos_theta_i).sqrt();
    let tan_theta_i = sin_theta_i / cos_theta_i;
    let cot_theta_i = 1.0 / tan_theta_i;

    // Search interval -- everything is parameterized in the Erf() domain

    let mut a = -1.0;
    let mut c = erf(cot_theta_i);
    let sample_x = u1.max(1e-6);

    // We can do better (inverse of an approximation computed in
    // Mathematica)
    let theta_i = cos_theta_i.acos();
    let fit = 1.0 + theta_i * (-0.876 + theta_i * (0.4265 - 0.0594 * theta_i));
    let mut b = c - (1.0 + c) * f64::powf(1.0 - sample_x, fit);

    // normalization factor for the CDF
    let sqrt_pi_inv = 1.0 / PI.sqrt();
    let normalization =
        1.0 / (1.0 + c + sqrt_pi_inv * tan_theta_i * (-cot_theta_i * cot_theta_i).exp());

    for _it in 0..10 {
        // bisection criterion -- the oddly-looking Boolean expression
        // are intentional to check for NaNs at little additional cost
        if !(b >= a && b <= c) {
            b = 0.5 * (a + c);
        }
        // evaluate the CDF and its derivative (i.e. the density
        // function)
        let inv_erf = erf_inv(b);
        let value = normalization
            * (1.0 + b + sqrt_pi_inv * tan_theta_i * (-inv_erf * inv_erf).exp())
            - sample_x;
        let derivative = normalization * (1.0 - inv_erf * tan_theta_i);

        if value.abs() < 1e-5 {
            break;
        }

        // update bisection intervals
        if value > 0.0 {
            c = b;
        } else {
            a = b;
        }
        b -= value / derivative;
    }

    // now convert back into a slope value
    *slope_x = erf_inv(b);

    // simulate Y component
    *slope_y = erf_inv(2.0 * u2.max(1e-6) - 1.0);

    assert!(!(*slope_x).is_infinite());
    assert!(!(*slope_x).is_nan());
    assert!(!(*slope_y).is_infinite());
    assert!(!(*slope_y).is_nan());
}

fn beckmann_sample(wi: &Vector3f, alpha_x: f64, alpha_y: f64, u1: f64, u2: f64) -> Vector3f {
    // 1. stretch wi
    let wi_stretched: Vector3f = Vector3f {
        x: alpha_x * wi.x,
        y: alpha_y * wi.y,
        z: wi.z,
    }
    .normalize();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    let mut slope_x = 0.0;
    let mut slope_y = 0.0;
    beckmann_sample_11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    // 3. rotate
    let tmp = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x *= alpha_x;
    slope_y *= alpha_y;

    // 5. compute normal
    Vector3f {
        x: -slope_x,
        y: -slope_y,
        z: 1.0,
    }
    .normalize()
}

impl MicrofacetDistribution for BeckmannDistribution {
    fn d(&self, wh: &Vector3f) -> f64 {
        let tan2_theta: f64 = tan_2_theta(wh);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let cos4_theta: f64 = cos_2_theta(wh) * cos_2_theta(wh);
        (-tan2_theta
            * (cos_2_phi(wh) / (self.alpha_x * self.alpha_x)
                + sin_2_phi(wh) / (self.alpha_y * self.alpha_y)))
            .exp()
            / (PI * self.alpha_x * self.alpha_y * cos4_theta)
    }
    fn lambda(&self, w: &Vector3f) -> f64 {
        let abs_tan_theta = tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        // compute _alpha_ for direction _w_
        let alpha = (cos_2_phi(w) * self.alpha_x * self.alpha_x
            + sin_2_phi(w) * self.alpha_y * self.alpha_y)
            .sqrt();
        let a = 1.0 / (alpha * abs_tan_theta);
        if a >= 1.6 {
            return 0.0;
        }
        (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a)
    }
    fn sample_wh(&self, wo: &Vector3f, u: Point2f) -> Vector3f {
        if !self.sample_visible_area {
            // sample full distribution of normals for Beckmann
            // distribution

            // compute $\tan^2 \theta$ and $\phi$ for Beckmann
            // distribution sample
            let tan2_theta;
            let mut phi;
            if self.alpha_x == self.alpha_y {
                let log_sample = (1.0 - u[0]).ln();
                assert!(!log_sample.is_infinite());
                tan2_theta = -self.alpha_x * self.alpha_x * log_sample;
                phi = u[1] * 2.0 * PI;
            } else {
                // compute _tan_2_theta_ and _phi_ for anisotropic
                // Beckmann distribution
                let log_sample = (1.0 - u[0]).ln();
                assert!(!log_sample.is_infinite());
                phi = (self.alpha_y / self.alpha_x * (2.0 * PI * u[1] + 0.5 * PI).tan()).atan();
                if u[1] > 0.5 {
                    phi += PI;
                }
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let alpha_x2 = self.alpha_x * self.alpha_x;
                let alpha_y2 = self.alpha_y * self.alpha_y;
                tan2_theta =
                    -log_sample / (cos_phi * cos_phi / alpha_x2 + sin_phi * sin_phi / alpha_y2);
            }
            // map sampled Beckmann angles to normal direction _wh_
            let cos_theta = 1.0 / (1.0 + tan2_theta).sqrt();
            let sin_theta = ((0_f64).max(1.0 - cos_theta * cos_theta)).sqrt();
            let mut wh: Vector3f = spherical_direction(sin_theta, cos_theta, phi);
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
            wh
        } else {
            // sample visible area of normals for Beckmann distribution
            let mut wh: Vector3f;
            let flip: bool = wo.z < 0.0;
            if flip {
                wh = beckmann_sample(&-(*wo), self.alpha_x, self.alpha_y, u[0], u[1]);
            } else {
                wh = beckmann_sample(wo, self.alpha_x, self.alpha_y, u[0], u[1]);
            }
            if flip {
                wh = -wh;
            }
            wh
        }
    }
    fn get_sample_visible_area(&self) -> bool {
        self.sample_visible_area
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct TrowbridgeReitzDistribution {
    alpha_x: f64,
    alpha_y: f64,

    sample_visible_area: bool,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64, sample_visible_area: bool) -> Self {
        Self {
            alpha_x,
            alpha_y,
            sample_visible_area,
        }
    }
}

fn trowbridge_reitz_sample_11(
    cos_theta: f64,
    u1: f64,
    u2: f64,
    slope_x: &mut f64,
    slope_y: &mut f64,
) {
    // special case (normal incidence)
    if cos_theta > 0.9999 {
        let r: f64 = (u1 / (1.0 - u1)).sqrt();
        let phi: f64 = 6.283_185_307_18 * u2;
        *slope_x = r * phi.cos();
        *slope_y = r * phi.sin();
        return;
    }

    let sin_theta: f64 = (0.0 as f64).max(1.0 as f64 - cos_theta * cos_theta).sqrt();
    let tan_theta: f64 = sin_theta / cos_theta;
    let a: f64 = 1.0 / tan_theta;
    let g1: f64 = 2.0 / (1.0 + (1.0 + 1.0 / (a * a)).sqrt());

    // sample slope_x
    let a: f64 = 2.0 * u1 / g1 - 1.0;
    let mut tmp: f64 = 1.0 / (a * a - 1.0);
    if tmp > 1e10 {
        tmp = 1e10;
    }
    let b: f64 = tan_theta;
    let d: f64 = (b * b * tmp * tmp - (a * a - b * b) * tmp)
        .max(0.0 as f64)
        .sqrt();
    let slope_x_1: f64 = b * tmp - d;
    let slope_x_2: f64 = b * tmp + d;
    if a < 0.0 || slope_x_2 > 1.0 / tan_theta {
        *slope_x = slope_x_1;
    } else {
        *slope_x = slope_x_2;
    }

    // sample slope_y
    let s: f64;
    let new_u2 = if u2 > 0.5 {
        s = 1.0;
        2.0 * (u2 - 0.5)
    } else {
        s = -1.0;
        2.0 * (0.5 - u2)
    };
    let z: f64 = (new_u2 * (new_u2 * (new_u2 * 0.27385 - 0.73369) + 0.46341))
        / (new_u2 * (new_u2 * (new_u2 * 0.093_073 + 0.309_420) - 1.0) + 0.597_999);
    *slope_y = s * z * (1.0 + *slope_x * *slope_x).sqrt();

    assert!(!(*slope_y).is_infinite());
    assert!(!(*slope_y).is_nan());
}

fn trowbridge_reitz_sample(
    wi: &Vector3f,
    alpha_x: f64,
    alpha_y: f64,
    u1: f64,
    u2: f64,
) -> Vector3f {
    // 1. stretch wi
    let wi_stretched: Vector3f = Vector3f {
        x: alpha_x * wi.x,
        y: alpha_y * wi.y,
        z: wi.z,
    }
    .normalize();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    let mut slope_x: f64 = 0.0;
    let mut slope_y: f64 = 0.0;
    trowbridge_reitz_sample_11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    // 3. rotate
    let tmp: f64 = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x *= alpha_x;
    slope_y *= alpha_y;

    // 5. compute normal
    Vector3f {
        x: -slope_x,
        y: -slope_y,
        z: 1.0,
    }
    .normalize()
}

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: &Vector3f) -> f64 {
        let tan2_theta = tan_2_theta(wh);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let cos4_theta = cos_2_theta(wh).powi(2);
        let e = (cos_2_phi(wh) / (self.alpha_x.powi(2)) + sin_2_phi(wh) / self.alpha_y.powi(2))
            * tan2_theta;
        1.0 / (PI * self.alpha_x * self.alpha_y * cos4_theta * (1.0 + e) * (1.0 + e))
    }
    fn lambda(&self, w: &Vector3f) -> f64 {
        let abs_tan_theta = tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        // Compute _alpha_ for direction _w_
        let alpha =
            (cos_2_phi(w) * self.alpha_x.powi(2) + sin_2_phi(w) * self.alpha_y.powi(2)).sqrt();
        let alpha2_tan2_theta = (alpha * abs_tan_theta).powi(2);
        (-1.0 + (1.0 + alpha2_tan2_theta).sqrt()) / 2.0
    }
    fn sample_wh(&self, wo: &Vector3f, u: Point2f) -> Vector3f {
        let mut wh;
        if !self.get_sample_visible_area() {
            let cos_theta;
            let mut phi = (2.0 * PI) * u[1];
            if self.alpha_x == self.alpha_y {
                let tan_theta2 = self.alpha_x.powi(2) * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            } else {
                phi = (self.alpha_y / self.alpha_x * (2.0 * PI * u[1] + 0.5 * PI).tan()).atan();
                if u[1] > 0.5 {
                    phi += PI;
                }
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let alpha_x2 = self.alpha_x.powi(2);
                let alpha_y2 = self.alpha_y.powi(2);
                let alpha2 = 1.0 / (cos_phi.powi(2) / alpha_x2 + sin_phi.powi(2) / alpha_y2);
                let tan_theta2 = alpha2 * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            }
            let sin_theta = (1.0 - cos_theta.powi(2)).max(0.0).sqrt();
            wh = spherical_direction(sin_theta, cos_theta, phi);
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
        } else {
            let flip = wo.z < 0.0;
            if flip {
                wh = -trowbridge_reitz_sample(&(-(*wo)), self.alpha_x, self.alpha_y, u[0], u[1]);
            } else {
                wh = trowbridge_reitz_sample(wo, self.alpha_x, self.alpha_y, u[0], u[1]);
            };
        }
        wh
    }
    fn get_sample_visible_area(&self) -> bool {
        self.sample_visible_area
    }
}
