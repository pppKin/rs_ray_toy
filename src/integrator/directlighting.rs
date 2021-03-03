use std::sync::{Arc, Mutex};

use crate::{
    geometry::RayDifferential,
    interaction::{Interaction, SurfaceInteraction},
    primitives::Primitive,
    samplers::Sampler,
    scene::Scene,
    spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{
    uniform_sample_all_lights, uniform_sample_one_light, Integrator, SamplerIntegrator,
    SamplerIntegratorData,
};

pub enum LightStrategy {
    UniformSampleAll,
    UniformSampleOne,
}

pub struct DirectLightingIntegrator {
    strategy: LightStrategy,
    max_depth: usize,
    n_light_samples: Vec<u32>,

    i: Arc<SamplerIntegratorData>,
}

impl Integrator for DirectLightingIntegrator {
    fn render(&mut self, scene: &Scene) {
        self.si_render(scene)
    }
}

// Spectrum DirectLightingIntegrator::Li(const RayDifferential &ray,
//                                       const Scene &scene, Sampler &sampler,
//                                       MemoryArena &arena, int depth) const {

// }

impl SamplerIntegrator for DirectLightingIntegrator {
    fn itgt(&self) -> Arc<SamplerIntegratorData> {
        Arc::clone(&self.i)
    }

    fn preprocess(&mut self, scene: &Scene, am_sampler: Arc<Mutex<dyn Sampler>>) {
        match self.strategy {
            LightStrategy::UniformSampleAll => {
                // Compute number of samples to use for each light
                let mut sampler = am_sampler.lock().unwrap();
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
            LightStrategy::UniformSampleOne => {}
        }
        todo!()
    }

    fn li(
        &self,
        ray: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let mut l = Spectrum::zero();
        let mut isect = SurfaceInteraction::default();

        // Find closest ray intersection or return background radiance
        if !scene.intersect(&mut ray.ray, &mut isect) {
            for light in &scene.lights {
                l += light.le(ray);
                return l;
            }
        }

        // Compute scattering functions for surface interaction
        isect.compute_scattering_functions(ray, false, crate::material::TransportMode::Radiance);
        if isect.bsdf.is_none() {
            return self.li(
                &mut isect.ist.spawn_ray(ray.ray.d).into(),
                scene,
                sampler,
                depth,
            );
        }
        // Compute emitted light if ray hit an area light source
        l += isect.le(&isect.ist.wo);

        if scene.lights.len() > 0 {
            // Compute direct lighting for _DirectLightingIntegrator_ integrator
            match self.strategy {
                LightStrategy::UniformSampleAll => {
                    l += uniform_sample_all_lights(
                        &Interaction::Surface(isect.clone()),
                        scene,
                        sampler,
                        &self.n_light_samples,
                        false,
                    );
                }
                LightStrategy::UniformSampleOne => {
                    l += uniform_sample_one_light(
                        &Interaction::Surface(isect.clone()),
                        scene,
                        sampler,
                        false,
                        None,
                    );
                }
            }
        }
        if (depth + 1) < self.max_depth {
            // Trace rays for specular reflection and refraction
            l += self.specular_reflect(ray, &isect, scene, sampler, depth);
            l += self.specular_transmit(ray, &isect, scene, sampler, depth);
        }
        l
    }
}
