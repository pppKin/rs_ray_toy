use std::sync::Arc;

use crate::{
    geometry::RayDifferential, interaction::SurfaceInteraction, samplers::Sampler, scene::Scene,
    spectrum::Spectrum, SPECTRUM_N,
};

use super::{Integrator, SamplerIntegrator, SamplerIntegratorData};

pub struct IntersectDebugIntegrator {
    max_depth: usize,
    i: Arc<SamplerIntegratorData>,
}

impl IntersectDebugIntegrator {
    pub fn new(max_depth: usize, i: Arc<SamplerIntegratorData>) -> Self {
        Self { max_depth, i }
    }
}

impl Integrator for IntersectDebugIntegrator {
    fn render(&mut self, scene: &Scene) {
        self.si_render(scene)
    }
}

impl SamplerIntegrator for IntersectDebugIntegrator {
    fn itgt(&self) -> Arc<SamplerIntegratorData> {
        Arc::clone(&self.i)
    }

    fn preprocess(&mut self, _scene: &Scene, _sampler: &mut Sampler) {}

    fn li(
        &self,
        ray: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let mut l;
        // Find closest ray intersection or return background radiance
        let mut isect = SurfaceInteraction::default();
        if !scene.intersect(&mut ray.ray, &mut isect) {
            return Spectrum::zero();
        } else {
            l = Spectrum::new([0.1, 0.0, 0.0]);
        }
        if (depth + 1) < self.max_depth {
            // Trace rays for specular reflection and refraction
            l += self.specular_reflect(ray, &isect, scene, sampler, depth);
            l += self.specular_transmit(ray, &isect, scene, sampler, depth);
        }
        l
    }
}
