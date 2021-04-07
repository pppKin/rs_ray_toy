use std::sync::Arc;

use crate::{
    geometry::{abs_dot3, dot3, RayDifferential, Vector3f},
    interaction::{Interaction, SurfaceInteraction},
    reflection::{BXDF_ALL, BXDF_SPECULAR, BXDF_TRANSMISSION},
    samplers::{ISampler, Sampler},
    sampling::Distribution1D,
    scene::Scene,
    spectrum::{ISpectrum, Spectrum},
    SPECTRUM_N,
};

use super::{uniform_sample_one_light, Integrator, SamplerIntegrator, SamplerIntegratorData};

pub struct PathIntegrator {
    max_depth: usize,
    rr_threshold: f64,
    //  we'll use UniformLightDistribution/PowerLightDistribution for now
    light_distrib: Arc<Distribution1D>,

    i: Arc<SamplerIntegratorData>,
}

impl PathIntegrator {
    pub fn new(max_depth: usize, rr_threshold: f64, i: Arc<SamplerIntegratorData>) -> Self {
        Self {
            max_depth,
            rr_threshold,
            light_distrib: Arc::new(Distribution1D::default()),
            i,
        }
    }
}

impl Integrator for PathIntegrator {
    fn render(&mut self, scene: &Scene) {
        self.si_render(scene)
    }
}

impl SamplerIntegrator for PathIntegrator {
    fn itgt(&self) -> Arc<SamplerIntegratorData> {
        Arc::clone(&self.i)
    }

    fn preprocess(&mut self, scene: &Scene, _sampler: &mut Sampler) {
        self.light_distrib = Arc::new(Distribution1D::new(vec![1.0; scene.lights.len()]));
    }

    fn li(
        &self,
        r: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut Sampler,
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
            // Find next path vertex and accumulate contribution
            // Intersect _ray_ with scene and store intersection in _isect_
            let mut isect = SurfaceInteraction::default();
            let found_intersection = scene.intersect(&mut ray.ray, &mut isect);
            // Possibly add emitted light at intersection
            if bounces == 0 || specular_bounce {
                // Add emitted light at path vertex or from the environment
                if found_intersection {
                    l += isect.le(&-ray.ray.d) * beta;
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
                //     VLOG(2) << "Skipping intersection due to null bsdf";
                ray = isect.ist.spawn_ray(ray.ray.d).into();
                bounces -= 1;
                continue;
            }

            let distrib = self.light_distrib.clone();
            // Sample illumination from lights to find path contribution.
            // (But skip this for perfectly specular BSDFs.)
            if let Some(bsdf) = &isect.bsdf {
                if bsdf.num_components(BXDF_ALL & !BXDF_SPECULAR) > 0 {
                    let ld = beta
                        * uniform_sample_one_light(
                            &Interaction::Surface(isect.clone()),
                            scene,
                            sampler,
                            false,
                            Some(distrib.clone()),
                        );
                    l += ld;
                }
            }

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
            assert!(beta.y() > 0.0);
            assert!(beta.y().is_finite());
            specular_bounce = (flags & BXDF_SPECULAR) != 0;

            if (flags & BXDF_SPECULAR) > 0 && (flags & BXDF_TRANSMISSION) > 0 {
                if let Some(bsdf) = isect.bsdf {
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

            // Account for subsurface scattering, if applicable
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
                            Some(distrib),
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

            // Possibly terminate the path with Russian roulette.
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
