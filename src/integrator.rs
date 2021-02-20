use std::sync::Arc;

use crate::{
    camera::ICamera,
    geometry::{Point2i, RayDifferential},
    interaction::SurfaceInteraction,
    samplers::Sampler,
    scene::Scene,
    spectrum::Spectrum,
    SPECTRUM_N,
};
pub trait Integrator {
    fn render(&self, scene: &Scene);
}

#[derive(Debug, Clone)]
pub struct SamplerIntegratorData {
    cam: Arc<dyn ICamera>,
    sampler: Arc<dyn Sampler>,
    pixel_bounds: Point2i,
}

pub trait SamplerIntegrator: Integrator {
    fn itgt(&self) -> &SamplerIntegratorData;

    fn preprocess(&self, scene: &Scene, sampler: &dyn Sampler);
    fn li(
        &self,
        ray: &RayDifferential,
        scene: &Scene,
        sampler: &dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N>;
    fn specular_reflect(
        &self,
        ray: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &Scene,
        sampler: &dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N>;
    fn specular_transmit(
        &self,
        ray: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &Scene,
        sampler: &dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N>;
}
