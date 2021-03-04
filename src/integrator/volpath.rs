use std::sync::{Arc, Mutex};

use crate::{
    geometry::{abs_dot3, dot3, RayDifferential, Vector3f},
    interaction::{Interaction, MediumInteraction, SurfaceInteraction},
    primitives::Primitive,
    reflection::{BXDF_ALL, BXDF_SPECULAR, BXDF_TRANSMISSION},
    samplers::Sampler,
    sampling::Distribution1D,
    scene::Scene,
    spectrum::{ISpectrum, Spectrum},
    SPECTRUM_N,
};

use super::{uniform_sample_one_light, Integrator, SamplerIntegrator, SamplerIntegratorData};

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

    fn preprocess(&mut self, scene: &Scene, _sampler: Arc<Mutex<dyn Sampler>>) {
        if scene.lights.len() == 0 {
            self.light_distr = Arc::new(Distribution1D::default());
        } else {
            let mut light_power = vec![];
            for light in &scene.lights {
                light_power.push(light.power().y());
            }
            self.light_distr = Arc::new(Distribution1D::new(light_power));
        }
    }

    fn li(
        &self,
        r: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        _depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let mut l = Spectrum::zero();
        let mut beta = Spectrum::one();

        let mut ray = r.clone();
        let mut specular_bounce = false;
        let mut bounces = 0;
        // Added after book publication: etaScale tracks the accumulated effect
        // of radiance scaling due to rays passing through refractive
        // boundaries (see the derivation on p. 527 of the third edition). We
        // track this value in order to remove it from beta when we apply
        // Russian roulette; this is worthwhile, since it lets us sometimes
        // avoid terminating refracted rays that are about to be refracted back
        // out of a medium and thus have their beta value increased.
        let mut eta_scale = 1.0;

        loop {
            //  Intersect _ray_ with scene and store intersection in _isect_
            let mut isect = SurfaceInteraction::default();
            let found_intersection = scene.intersect(&mut ray.ray, &mut isect);

            //  Sample the participating medium, if present
            let mut mi = MediumInteraction::default();
            if let Some(mm) = &ray.ray.medium {
                beta *= mm.sample(&ray.ray, sampler, &mut mi);
            }

            if beta.is_black() {
                break;
            }

            // Handle an interaction with a medium or a surface
            if mi.is_valid() {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if bounces >= self.max_depth {
                    break;
                }
                l += beta
                    * uniform_sample_one_light(
                        &Interaction::Medium(mi.clone()),
                        scene,
                        sampler,
                        true,
                        Some(self.light_distr.clone()),
                    );
                let wo = -ray.ray.d;
                let mut wi = Vector3f::default();
                if let Some(phase) = &mi.phase {
                    phase.sample_p(&wo, &mut wi, sampler.get_2d());
                    ray = mi.ist.spawn_ray(wi).into();
                    specular_bounce = false;
                }
            } else {
                // Handle scattering at point on surface for volumetric path tracer

                // Possibly add emitted light at intersection
                if bounces == 0 || specular_bounce {
                    // Add emitted light at path vertex or from the environment
                    if found_intersection {
                        l += beta * isect.le(&-ray.ray.d);
                    } else {
                        for light in &scene.infinite_lights {
                            l += beta * light.le(&ray);
                        }
                    }
                }

                // Terminate path if ray escaped or _maxDepth_ was reached
                if !found_intersection || bounces >= self.max_depth {
                    break;
                }
                // Compute scattering functions and skip over medium boundaries
                isect.compute_scattering_functions(
                    &ray,
                    true,
                    crate::material::TransportMode::Radiance,
                );

                if isect.bsdf.is_none() {
                    ray = isect.ist.spawn_ray(ray.ray.d).into();
                    bounces -= 1;
                    continue;
                }

                // Sample illumination from lights to find attenuated path
                l += beta
                    * uniform_sample_one_light(
                        &Interaction::Surface(isect.clone()),
                        scene,
                        sampler,
                        true,
                        Some(self.light_distr.clone()),
                    );
                // Sample BSDF to get new path direction
                let wo = -ray.ray.d;
                let mut wi = Vector3f::default();
                let mut pdf = 0.0;
                let mut flags = 0;
                let mut f = Spectrum::zero();
                if let Some(bsdf) = &isect.bsdf {
                    f = bsdf.sample_f(
                        &wo,
                        &mut wi,
                        &sampler.get_2d(),
                        &mut pdf,
                        BXDF_ALL,
                        &mut flags,
                    );
                }
                if f.is_black() || pdf == 0.0 {
                    break;
                }
                beta *= f * abs_dot3(&wi, &isect.shading.n) / pdf;
                assert!(beta.y().is_finite());
                specular_bounce = (flags & BXDF_SPECULAR) != 0;
                if (flags & BXDF_SPECULAR) > 0 && (flags & BXDF_TRANSMISSION) > 0 {
                    if let Some(bsdf) = &isect.bsdf {
                        let eta = bsdf.eta;
                        // Update the term that tracks radiance scaling for refraction
                        // depending on whether the ray is entering or leaving the
                        // medium.

                        eta_scale *= if dot3(&wo, &isect.ist.n) > 0.0 {
                            eta * eta
                        } else {
                            1.0 / (eta * eta)
                        }
                    }
                }

                ray = isect.ist.spawn_ray(wi).into();
                // Account for attenuated subsurface scattering, if applicable
                if isect.bssrdf.is_some() && (flags & BXDF_TRANSMISSION) > 0 {
                    // Importance sample the BSSRDF
                    let mut pi = SurfaceInteraction::default();
                    if let Some(bssrdf) = &isect.bssrdf {
                        let s = bssrdf.sample_s(
                            scene,
                            sampler.get_1d(),
                            &sampler.get_2d(),
                            &mut pi,
                            &mut pdf,
                        );
                        if s.is_black() || pdf == 0.0 {
                            break;
                        }
                        beta *= s / pdf;
                        // Account for the direct subsurface scattering component
                        l += beta
                            * uniform_sample_one_light(
                                &Interaction::Surface(pi.clone()),
                                scene,
                                sampler,
                                false,
                                Some(self.light_distr.clone()),
                            );
                        // Account for the indirect subsurface scattering component
                        if let Some(bsdf) = &pi.bsdf {
                            let f = bsdf.sample_f(
                                &pi.ist.wo,
                                &mut wi,
                                &sampler.get_2d(),
                                &mut pdf,
                                BXDF_ALL,
                                &mut flags,
                            );

                            if f.is_black() || pdf == 0.0 {
                                break;
                            }
                            beta *= f * abs_dot3(&wi, &pi.shading.n) / pdf;
                            assert!(beta.y().is_finite());
                            specular_bounce = (flags & BXDF_SPECULAR) > 0;
                            ray = pi.ist.spawn_ray(wi).into();
                        }
                    }
                }
            }
            // Possibly terminate the path with Russian roulette
            // Factor out radiance scaling due to refraction in rrBeta.
            let rr_beta = beta * eta_scale;
            if rr_beta.max_component_value() < self.rr_threshold && bounces > 3 {
                let q = (1.0 - rr_beta.max_component_value()).max(0.05);
                if sampler.get_1d() < q {
                    break;
                }
                beta /= 1.0 - q;
                assert!(beta.y().is_finite());
            }
            bounces += 1;
        }
        l
    }
}
