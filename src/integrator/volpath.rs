use std::sync::{Arc, Mutex};

use crate::{
    geometry::RayDifferential, samplers::Sampler, sampling::Distribution1D, scene::Scene,
    spectrum::Spectrum, SPECTRUM_N,
};

use super::{Integrator, SamplerIntegrator, SamplerIntegratorData};

#[derive(Debug)]
pub struct VolPathIntegrator {
    max_depth: usize,
    rr_threshold: f64,
    //  we'll use UniformLightDistribution/PowerLightDistribution for now
    light_distr: Arc<Distribution1D>,

    i: Arc<SamplerIntegratorData>,
}

impl Integrator for VolPathIntegrator {
    fn render(&mut self, scene: &Scene) {
        self.si_render(scene)
    }
}

impl SamplerIntegrator for VolPathIntegrator {
    fn itgt(&self) -> Arc<SamplerIntegratorData> {
        Arc::clone(&self.i)
    }

    fn preprocess(&mut self, scene: &Scene, sampler: Arc<Mutex<dyn Sampler>>) {
        todo!()
    }

    fn li(
        &self,
        ray: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        todo!()
    }
}
