use std::sync::Arc;

use crate::{
    geometry::RayDifferential,
    interaction::{Interaction, SurfaceInteraction},
    samplers::{ISampler, RoundCount, Sampler},
    scene::Scene,
    spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{uniform_sample_all_lights, Integrator, SamplerIntegrator, SamplerIntegratorData};

pub struct IntersectDebugIntegrator {
    max_depth: usize,
    n_light_samples: Vec<u32>,
    i: Arc<SamplerIntegratorData>,
}

impl IntersectDebugIntegrator {
    pub fn new(max_depth: usize, i: Arc<SamplerIntegratorData>) -> Self {
        Self {
            max_depth,
            n_light_samples: vec![],
            i,
        }
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

    fn preprocess(&mut self, scene: &Scene, sampler: &mut Sampler) {
        // Compute number of samples to use for each light
        for light in &scene.lights {
            self.n_light_samples
                .push(sampler.round_count(light.n_samples() as u32));
        }

        for _i in 0..self.max_depth {
            for j in 0..scene.lights.len() {
                sampler.request_2d_array(self.n_light_samples[j]);
                sampler.request_2d_array(self.n_light_samples[j]);
            }
        }
    }

    fn li(
        &self,
        ray: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let l;
        // Find closest ray intersection or return background radiance
        let mut isect = SurfaceInteraction::default();
        if !scene.intersect(&mut ray.ray, &mut isect) {
            return Spectrum::zero();
        } else {
            l = Spectrum::new([0.1, 0.1, 0.1]);
        }
        isect.compute_scattering_functions(ray, false, crate::material::TransportMode::Radiance);

        let mut s_l = Spectrum::zero();
        if scene.lights.len() > 0 {
            s_l += uniform_sample_all_lights(
                &Interaction::Surface(isect.clone()),
                scene,
                sampler,
                &self.n_light_samples,
                false,
            );
        }
        if (depth + 1) < self.max_depth {
            // Trace rays for specular reflection and refraction
            s_l += self.specular_reflect(ray, &isect, scene, sampler, depth);
            s_l += self.specular_transmit(ray, &isect, scene, sampler, depth);
        }
        l + s_l
    }
}
