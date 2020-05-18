use std::{f64::consts::PI, fmt::Debug, sync::Arc};

use crate::{
    geometry::{Point2f, Ray, Vector3f},
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

pub trait IMedium: Debug {
    fn tr(&self, ray: &Ray, sampler: &dyn Sampler) -> Spectrum<SPECTRUM_N>;
    fn sample(
        &self,
        ray: &Ray,
        sampler: &dyn Sampler,
        mi: &MediumInterface,
    ) -> Spectrum<SPECTRUM_N>;
}

pub type Medium = Option<Arc<dyn IMedium>>;

pub struct MediumInterface {
    inside: Medium,
    outside: Medium,
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
