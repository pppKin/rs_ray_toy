use std::sync::Arc;

use crate::{
    camera::RealisticCamera,
    filters::IFilter,
    geometry::{abs_dot3, Point2f, Point2i, RayDifferential, Vector3f},
    interaction::{Interaction, SurfaceInteraction},
    lights::{is_delta_light, Light, VisibilityTester},
    primitives::Primitive,
    reflection::{BXDF_ALL, BXDF_NONE, BXDF_SPECULAR},
    samplers::Sampler,
    sampling::{power_heuristic, Distribution1D},
    scene::Scene,
    spectrum::{ISpectrum, Spectrum},
    SPECTRUM_N,
};

pub trait Integrator {
    fn render(&self, scene: &Scene);
}

#[derive(Debug, Clone)]
pub struct SamplerIntegratorData<T: IFilter> {
    cam: Arc<RealisticCamera<T>>,
    sampler: Arc<dyn Sampler>,
    pixel_bounds: Point2i,
}

pub trait SamplerIntegrator<T: IFilter>: Integrator {
    fn print_helloworld(&self) {
        println!("hello world");
    }

    fn itgt(&self) -> Arc<SamplerIntegratorData<T>>;

    fn preprocess(&self, scene: &Scene, sampler: Arc<dyn Sampler>);
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

pub fn uniform_sample_all_lights(
    it: &Interaction,
    scene: &Scene,
    sampler: &mut dyn Sampler,
    n_light_samples: &[usize],
    handle_media: bool,
) -> Spectrum<SPECTRUM_N> {
    // ProfilePhase p(Prof::DirectLighting);
    let mut l = Spectrum::zero();
    for j in 0..scene.lights.len() {
        // Accumulate contribution of _j_th light to _L_
        let light = Arc::clone(&scene.lights[j]);
        let n_samples = n_light_samples[j];
        // TODO: this seems rather expensive and inefficient, is it really worth it?
        let u_light_array = Vec::from(sampler.get_2d_array(n_samples as u32));
        // TODO: this seems rather expensive and inefficient, is it really worth it?
        let u_scattering_array = Vec::from(sampler.get_2d_array(n_samples as u32));
        if u_light_array.len() <= 0 || u_scattering_array.len() <= 0 {
            // Use a single sample for illumination from _light_
            let u_light = sampler.get_2d();
            let u_scattering = sampler.get_2d();

            l += estimate_direct(
                it,
                &u_scattering,
                Arc::clone(&light),
                &u_light,
                scene,
                sampler,
                handle_media,
                false,
            );
        } else {
            // Estimate direct lighting using sample arrays
            let mut ld = Spectrum::zero();
            for k in 0..n_samples {
                ld += estimate_direct(
                    it,
                    &u_scattering_array[k],
                    Arc::clone(&light),
                    &u_light_array[k],
                    scene,
                    sampler,
                    handle_media,
                    false,
                )
            }
            l += ld / n_samples as f64;
        }
    }
    l
}

pub fn estimate_direct(
    it: &Interaction,
    u_scattering: &Point2f,
    light: Arc<dyn Light>,
    u_light: &Point2f,
    scene: &Scene,
    sampler: &mut dyn Sampler,
    handle_media: bool,
    specular: bool,
) -> Spectrum<SPECTRUM_N> {
    let bsdf_flags = if specular {
        BXDF_ALL
    } else {
        BXDF_ALL & !BXDF_SPECULAR
    };
    let mut ld = Spectrum::zero();
    // Sample light source with multiple importance sampling
    let mut wi = Vector3f::default();
    let mut light_pdf = 0.0;
    let mut scattering_pdf = 0.0;
    let mut visibility = VisibilityTester::default();
    let ref_ist;
    match it {
        Interaction::Surface(si) => {
            ref_ist = &si.ist;
        }
        Interaction::Medium(mi) => {
            ref_ist = &mi.ist;
        }
    }
    let mut li = light.sample_li(ref_ist, u_light, &mut wi, &mut light_pdf, &mut visibility);

    if light_pdf > 0.0 && !li.is_black() {
        // Compute BSDF or phase function's value for light sample
        let f: Spectrum<SPECTRUM_N>;
        match it {
            Interaction::Surface(si) => {
                if let Some(bsdf) = &si.bsdf {
                    f = bsdf.f(&si.ist.wo, &wi, bsdf_flags) * abs_dot3(&wi, &si.shading.n);
                    scattering_pdf = bsdf.pdf(&si.ist.wo, &wi, bsdf_flags);
                } else {
                    f = Spectrum::zero();
                    scattering_pdf = 0.0;
                }
            }
            Interaction::Medium(mi) => {
                // Evaluate phase function for light sampling strategy
                let p = mi.phase.p(&mi.ist.wo, &wi);
                f = Spectrum::from(p);
                scattering_pdf = p;
            }
        }
        if !f.is_black() {
            // Compute effect of visibility for light source sample
            if handle_media {
                li *= visibility.tr(scene, sampler);
            } else {
                if !visibility.unoccluded(scene) {
                    li = Spectrum::zero();
                } else {
                    // Log "  shadow ray unoccluded"
                }
            }

            // Add light's contribution to reflected radiance
            if !li.is_black() {
                if is_delta_light(light.flags()) {
                    ld += f * li / light_pdf;
                } else {
                    let weight = power_heuristic(1, light_pdf, 1, scattering_pdf);
                    ld += li * f * weight / light_pdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if !is_delta_light(light.flags()) {
        let mut f: Spectrum<SPECTRUM_N>;
        let mut sampled_specular = false;
        match it {
            Interaction::Surface(si) => {
                // Sample scattered direction for surface interactions
                if let Some(bsdf) = &si.bsdf {
                    let mut sampled_type = BXDF_NONE;
                    f = bsdf.sample_f(
                        &si.ist.wo,
                        &mut wi,
                        u_scattering,
                        &mut scattering_pdf,
                        bsdf_flags,
                        &mut sampled_type,
                    );
                    f *= abs_dot3(&wi, &si.shading.n);
                    sampled_specular = (sampled_type & BXDF_SPECULAR) != 0;
                } else {
                    f = Spectrum::zero();
                }
            }
            Interaction::Medium(mi) => {
                // Sample scattered direction for medium interactions
                let p = mi.phase.sample_p(&mi.ist.wo, &mut wi, *u_scattering);
                f = Spectrum::from(p);
                scattering_pdf = p;
            }
        }

        if !f.is_black() && scattering_pdf > 0.0 {
            // Account for light contributions along sampled direction _wi_
            let mut weight = 1.0;
            if !sampled_specular {
                light_pdf = light.pdf_li(ref_ist, &wi);
                if light_pdf == 0.0 {
                    return ld;
                }
                weight = power_heuristic(1, scattering_pdf, 1, light_pdf);
            }
            // Find intersection and compute transmittance
            let mut light_isect = SurfaceInteraction::default();
            let mut ray = ref_ist.spawn_ray(wi);
            let mut tr = Spectrum::one();
            let found_surface_interaction = if handle_media {
                scene.intersect_tr(&mut ray, sampler, &mut light_isect, &mut tr)
            } else {
                scene.intersect(&mut ray, &mut light_isect)
            };
            // Add light contribution from material sampling
            let mut li = Spectrum::zero();
            if found_surface_interaction {
                if let Some(pri) = &light_isect.primitive {
                    if let Some(area_light) = pri.get_arealight() {
                        let pa = &*area_light as *const _ as *const usize;
                        let pl = &*light as *const _ as *const usize;
                        if pa == pl {
                            li = light_isect.le(&-wi);
                        }
                    }
                }
            } else {
                li = light.le(&ray.into());
            }
            if !li.is_black() {
                ld += li * f * tr * weight / scattering_pdf;
            }
        }
    }
    ld
}

fn compute_light_power_distribution(scene: &Scene) -> Option<Arc<Distribution1D>> {
    if scene.lights.len() <= 0 {
        return None;
    }
    let mut light_power = vec![];
    let mut tmps = vec![];
    for light in &scene.lights {
        light_power.push(light.power().y());
        tmps.push(light);
    }
    Some(Arc::new(Distribution1D::new(light_power)))
}
