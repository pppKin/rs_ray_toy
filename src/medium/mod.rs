use std::{f64::consts::PI, fmt::Debug, sync::Arc};

use crate::{
    geometry::{dot3, spherical_direction_vec3, vec3_coordinate_system, Point2f, Ray, Vector3f},
    interaction::MediumInteraction,
    misc::copy_option_arc,
    rtoycore::SPECTRUM_N,
    samplers::Sampler,
    spectrum::Spectrum,
};

pub fn phase_hg(cos_theta: f64, g: f64) -> f64 {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    (1.0 / (PI * 4.0)) * (1.0 - g * g) / (denom * denom.sqrt())
}

pub trait PhaseFunction: std::fmt::Debug {
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> f64;
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: Point2f) -> f64;
}

#[derive(Debug)]
pub struct HenyeyGreenstein {
    g: f64,
}

impl HenyeyGreenstein {
    pub fn new(g: f64) -> Self {
        Self { g }
    }
}

impl PhaseFunction for HenyeyGreenstein {
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        // PhaseHG(Dot(wo, wi), g);
        phase_hg(dot3(wo, wi), self.g)
    }
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: Point2f) -> f64 {
        // Compute $\cos \theta$ for Henyey--Greenstein sample
        let cos_theta;
        if self.g.abs() < 1e-3 {
            cos_theta = 1.0 - 2.0 * u[0];
        } else {
            let sqr_term = (1.0 - self.g * self.g) / (1.0 + self.g - 2.0 * self.g * u[0]);
            cos_theta = -(1.0 + self.g * self.g - sqr_term * sqr_term) / (2.0 * self.g);
        }

        // Compute direction _wi_ for Henyey--Greenstein sample
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * u[0];
        let mut v1 = Vector3f::default();
        let mut v2 = Vector3f::default();
        vec3_coordinate_system(wo, &mut v1, &mut v2);
        *wi = spherical_direction_vec3(sin_theta, cos_theta, phi, &v1, &v2, wo);
        phase_hg(cos_theta, self.g)
    }
}

pub trait Medium: Debug {
    fn tr(&self, ray: &Ray, sampler: &mut dyn Sampler) -> Spectrum<SPECTRUM_N>;
    fn sample(
        &self,
        ray: &Ray,
        sampler: &mut dyn Sampler,
        mi: &mut MediumInteraction,
    ) -> Spectrum<SPECTRUM_N>;
}

pub type MediumOpArc = Option<Arc<dyn Medium>>;

#[derive(Debug)]
pub struct MediumInterface {
    inside: MediumOpArc,
    outside: MediumOpArc,
}

impl Clone for MediumInterface {
    fn clone(&self) -> MediumInterface {
        MediumInterface {
            inside: copy_option_arc(&self.inside),
            outside: copy_option_arc(&self.outside),
        }
    }
}

impl Default for MediumInterface {
    fn default() -> Self {
        MediumInterface {
            inside: None,
            outside: None,
        }
    }
}

pub mod grid;
