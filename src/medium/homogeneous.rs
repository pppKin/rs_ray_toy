use std::{f64::INFINITY, sync::Arc};

use crate::{
    geometry::{Normal3, Ray, Vector3f},
    interaction::{BaseInteraction, MediumInteraction},
    medium::{HenyeyGreenstein, Medium},
    samplers::Sampler,
    spectrum::Spectrum,
    SPECTRUM_N,
};

// HomogeneousMedium Declarations
#[derive(Debug)]
pub struct HomogeneousMedium {
    sigma_a: Spectrum<SPECTRUM_N>,
    sigma_s: Spectrum<SPECTRUM_N>,
    sigma_t: Spectrum<SPECTRUM_N>,

    g: f64,
}

impl HomogeneousMedium {
    pub fn new(
        sigma_a: Spectrum<SPECTRUM_N>,
        sigma_s: Spectrum<SPECTRUM_N>,
        sigma_t: Spectrum<SPECTRUM_N>,
        g: f64,
    ) -> Self {
        Self {
            sigma_a,
            sigma_s,
            sigma_t,
            g,
        }
    }
}

impl Medium for HomogeneousMedium {
    fn tr(&self, ray: &Ray, _sampler: &mut dyn Sampler) -> Spectrum<SPECTRUM_N> {
        (-self.sigma_t * (ray.t_max * ray.d.length()).min(INFINITY)).exp()
    }

    fn sample(
        &self,
        ray: &Ray,
        sampler: &mut dyn Sampler,
        mi: &mut MediumInteraction,
    ) -> Spectrum<SPECTRUM_N> {
        // Sample a channel and distance along the ray
        let channel = ((sampler.get_1d() * SPECTRUM_N as f64) as usize).min(SPECTRUM_N - 1);
        let dist = -((1.0 - sampler.get_1d()) / self.sigma_t[channel]).ln();
        let t = (dist / ray.d.length()).min(ray.t_max);
        let sampled_medium = t < ray.t_max;
        if sampled_medium {
            *mi = MediumInteraction::new(
                BaseInteraction::new(
                    ray.position(t),
                    ray.time,
                    Vector3f::default(),
                    -ray.d,
                    Normal3::default(),
                    None,
                ),
                Some(Arc::new(HenyeyGreenstein::new(self.g))),
            );
        }
        // Compute the transmittance and sampling density
        let tr = (-self.sigma_t * t.min(INFINITY) * ray.d.length()).exp();
        // Return weighting factor for scattering from homogeneous medium
        let density;
        if sampled_medium {
            density = self.sigma_t * tr;
        } else {
            density = tr;
        }
        let mut pdf = 0.0;
        for i in 0..SPECTRUM_N {
            pdf += density[i];
        }
        pdf *= 1.0 / SPECTRUM_N as f64;
        if pdf == 0.0 {
            assert!(tr.is_black());
            pdf = 1.0;
        }
        if sampled_medium {
            return tr * self.sigma_s / pdf;
        } else {
            return tr / pdf;
        }
    }
}
