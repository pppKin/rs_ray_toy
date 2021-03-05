use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::{
    camera::{ICamera, RealisticCamera},
    geometry::{abs_dot3, dot3, Bounds2i, Normal3f, Point2f, Point2i, RayDifferential, Vector3f},
    interaction::{Interaction, SurfaceInteraction},
    lights::{is_delta_light, Light, VisibilityTester},
    primitives::Primitive,
    reflection::{BXDF_ALL, BXDF_NONE, BXDF_REFLECTION, BXDF_SPECULAR, BXDF_TRANSMISSION},
    samplers::Sampler,
    sampling::{power_heuristic, Distribution1D},
    scene::Scene,
    spectrum::{ISpectrum, Spectrum},
    SPECTRUM_N,
};

pub trait Integrator: Send + Sync {
    fn render(&mut self, scene: &Scene);
}

#[derive(Debug, Clone)]
pub struct SamplerIntegratorData {
    pub cam: Arc<RealisticCamera>,
    pub sampler: Arc<Mutex<dyn Sampler>>,
    pub pixel_bounds: Bounds2i,
}

trait SamplerIntegrator: Send + Sync {
    fn itgt(&self) -> Arc<SamplerIntegratorData>;
    fn si_render(&mut self, scene: &Scene) {
        let itgt = self.itgt();
        let am_tile_sampler = itgt.sampler.clone();
        self.preprocess(scene, am_tile_sampler.clone());

        let sample_bounds = self.itgt().cam.camera.film.get_sample_bounds();
        let sample_extent = sample_bounds.diagonal();
        let tile_size = 16;
        let n_tiles_x = (sample_extent.x + tile_size - 1) / tile_size;
        let n_tiles_y = (sample_extent.y + tile_size - 1) / tile_size;

        (0..n_tiles_x).into_par_iter().for_each(|tile_x| {
            (0..n_tiles_y).into_par_iter().for_each(|tile_y| {
                // Compute sample bounds for tile
                let x0 = sample_bounds.p_min.x + tile_x * tile_size;
                let x1 = (x0 + tile_size).min(sample_bounds.p_max.x);
                let y0 = sample_bounds.p_min.y + tile_y * tile_size;
                let y1 = (y0 + tile_size).min(sample_bounds.p_max.y);
                let tile_bounds = Bounds2i::new(Point2i::new(x0, y0), Point2i::new(x1, y1));

                let mut tile_sampler = am_tile_sampler.lock().unwrap();
                let mut film_tile = itgt.cam.camera.film.get_film_tile(&tile_bounds);
                // Loop over pixels in tile to render them
                for pixel in tile_bounds.into_iter() {
                    tile_sampler.start_pixel(pixel);
                    // Do this check after the StartPixel() call; this keeps
                    // the usage of RNG values from (most) Samplers that use
                    // RNGs consistent, which improves reproducability /
                    // debugging.
                    if !Bounds2i::inside_exclusive(&pixel, &itgt.pixel_bounds) {
                        continue;
                    }

                    while tile_sampler.start_next_sample() {
                        // Initialize _CameraSample_ for current sample

                        let camera_sample = tile_sampler.get_camerasample(&pixel);
                        // Generate camera ray for current sample
                        let mut ray = RayDifferential::default();
                        let ray_weight =
                            itgt.cam.generate_ray_differential(&camera_sample, &mut ray);
                        ray.scale_differentials(
                            1.0 / (tile_sampler.samples_per_pixel() as f64).sqrt(),
                        );

                        // Evaluate radiance along camera ray
                        let mut l = Spectrum::zero();
                        if ray_weight > 0.0 {
                            l = self.li(&mut ray, scene, &mut *tile_sampler, 1);
                        }
                        // Issue warning if unexpected radiance value returned
                        if l.has_nan() {
                            // TODO:     LOG(ERROR) << StringPrintf(
                            //         "Not-a-number radiance value returned "
                            //         "for pixel (%d, %d), sample %d. Setting to black.",
                            //         pixel.x, pixel.y,
                            //         (int)tileSampler->CurrentSampleNumber());
                            l = Spectrum::zero();
                        } else if l.y() < -1e-5 {
                            // TODO:      LOG(ERROR) << StringPrintf(
                            //         "Negative luminance value, %f, returned "
                            //         "for pixel (%d, %d), sample %d. Setting to black.",
                            //         L.y(), pixel.x, pixel.y,
                            //         (int)tileSampler->CurrentSampleNumber());
                            l = Spectrum::zero();
                        } else if l.y().is_infinite() {
                            //  TODO:         LOG(ERROR) << StringPrintf(
                            //         "Infinite luminance value returned "
                            //         "for pixel (%d, %d), sample %d. Setting to black.",
                            //         pixel.x, pixel.y,
                            //         (int)tileSampler->CurrentSampleNumber());
                            l = Spectrum::zero();
                        }
                        // TODO: VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " <<
                        //     ray << " -> L = " << L;

                        // Add camera ray's contribution to image
                        film_tile.add_sample(&camera_sample.p_film, &mut l, ray_weight);
                    }
                }
                // TODO: LOG(INFO) << "Finished image tile " << tileBounds;

                // Merge image tile into _Film_
                itgt.cam.camera.film.merge_film_tile(&mut film_tile);
            });
        });
    }
    fn preprocess(&mut self, _scene: &Scene, _sampler: Arc<Mutex<dyn Sampler>>) {
        // do absolutely nothing at all
    }
    fn li(
        &self,
        ray: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N>;
    fn specular_reflect(
        &self,
        ray: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        // Compute specular reflection direction _wi_ and BSDF value
        let wo = isect.ist.wo;
        let mut wi = Vector3f::default();
        let mut pdf = 0.0;
        let ty = BXDF_SPECULAR | BXDF_REFLECTION;
        let mut sampled_type = 0;
        let f;
        if let Some(bsdf) = &isect.bsdf {
            f = bsdf.sample_f(
                &wo,
                &mut wi,
                &sampler.get_2d(),
                &mut pdf,
                ty,
                &mut sampled_type,
            );
        } else {
            return Spectrum::zero();
        }

        // Return contribution of specular reflection
        let ns = isect.shading.n;
        if pdf > 0.0 && !f.is_black() && abs_dot3(&wi, &ns) != 0.0 {
            // Compute ray differential _rd_ for specular reflection
            let mut rd: RayDifferential = isect.ist.spawn_ray(wi).into();
            if ray.has_differentials {
                rd.has_differentials = true;
                rd.rx_origin = isect.ist.p + isect.dpdx;
                rd.ry_origin = isect.ist.p + isect.dpdy;

                // Compute differential reflected directions
                let dndx: Normal3f =
                    isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
                let dndy: Normal3f =
                    isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;
                let dwodx = -ray.rx_direction - wo;
                let dwody = -ray.ry_direction - wo;
                let ddndx = dot3(&dwodx, &ns) + dot3(&wo, &dndx);
                let ddndy = dot3(&dwody, &ns) + dot3(&wo, &dndy);
                rd.rx_direction =
                    wi - dwodx + Vector3f::from(dndx * dot3(&wo, &ns) + ns * ddndx) * 0.2;
                rd.ry_direction =
                    wi - dwody + Vector3f::from(dndy * dot3(&wo, &ns) + ns * ddndy) * 0.2;
            }
            return f * self.li(&mut rd, scene, sampler, depth + 1) * abs_dot3(&wi, &ns) / pdf;
        } else {
            return Spectrum::zero();
        }
    }
    fn specular_transmit(
        &self,
        ray: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let wo = isect.ist.wo;
        let mut wi = Vector3f::default();
        let mut pdf = 0.0;
        let ty = BXDF_SPECULAR | BXDF_TRANSMISSION;
        let mut sampled_type = 0;
        let f;
        if let Some(bsdf) = &isect.bsdf {
            f = bsdf.sample_f(
                &wo,
                &mut wi,
                &sampler.get_2d(),
                &mut pdf,
                ty,
                &mut sampled_type,
            );
        } else {
            return Spectrum::zero();
        }

        let mut ns = isect.shading.n;
        if pdf > 0.0 && !f.is_black() && abs_dot3(&wi, &ns) != 0.0 {
            // Compute ray differential _rd_ for specular transmission
            let mut rd: RayDifferential = isect.ist.spawn_ray(wi).into();
            if ray.has_differentials {
                rd.has_differentials = true;
                rd.rx_origin = isect.ist.p + isect.dpdx;
                rd.ry_origin = isect.ist.p + isect.dpdy;

                let mut dndx: Normal3f =
                    isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
                let mut dndy: Normal3f =
                    isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;

                if let Some(bsdf) = &isect.bsdf {
                    // The BSDF stores the IOR of the interior of the object being
                    // intersected.  Compute the relative IOR by first out by
                    // assuming that the ray is entering the object.
                    let mut eta = 1.0 / bsdf.eta;
                    if dot3(&wo, &ns) < 0.0 {
                        // If the ray isn't entering, then we need to invert the
                        // relative IOR and negate the normal and its derivatives.
                        eta = 1.0 / eta;
                        ns = -ns;
                        dndx = -dndx;
                        dndy = -dndy;
                    }

                    // /*
                    //   Notes on the derivation:
                    //   - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [ \eta (\wo \cdot \N) - \cos \theta_t ] \N
                    //     It flips the normal to lie in the same hemisphere as \wo, and then \eta is the relative IOR from
                    //     \wo's medium to \wi's medium.
                    //   - If we denote the term in brackets by \mu, then we have: \wi = -\eta \omega_o + \mu \N
                    //   - Now let's take the partial derivative. (We'll use "d" for \partial in the following for brevity.)
                    //     We get: -\eta d\omega_o / dx + \mu dN/dx + d\mu/dx N.
                    //   - We have the values of all of these except for d\mu/dx (using bits from the derivation of specularly
                    //     reflected ray deifferentials).
                    //   - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We already have d(\wo \cdot N)/dx.
                    //   - The second term takes a little more work. We have:
                    //      \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
                    //      Starting from (\wo \cdot N)^2 and reading outward, we have \cos^2 \theta_o, then \sin^2 \theta_o,
                    //      then \sin^2 \theta_i (via Snell's law), then \cos^2 \theta_i and then \cos \theta_i.
                    //   - Let's take the partial derivative of the sqrt expression. We get:
                    //     1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot N)^2)).
                    //   - That partial derivatve is equal to:
                    //     d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo \cdot N).
                    //   - Plugging it in, we have d\mu/dx =
                    //     \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot N))/(-\wi \cdot N).
                    //  */
                    let dwodx = -ray.rx_direction - wo;
                    let dwody = -ray.ry_direction - wo;
                    let ddndx = dot3(&dwodx, &ns) + dot3(&wo, &dndx);
                    let ddndy = dot3(&dwody, &ns) + dot3(&wo, &dndy);
                    let mu = eta * dot3(&wo, &ns) - abs_dot3(&wi, &ns);
                    let dmudx = ddndx * (eta - (eta * eta * dot3(&wo, &ns)) / abs_dot3(&wi, &ns));
                    let dmudy = ddndy * (eta - (eta * eta * dot3(&wo, &ns)) / abs_dot3(&wi, &ns));
                    rd.rx_direction = wi - dwodx * eta + (dndx * mu + ns * dmudx).into();
                    rd.ry_direction = wi - dwody * eta + (dndy * mu + ns * dmudy).into();
                } else {
                    return Spectrum::zero();
                }
            }
            return f * self.li(&mut rd, scene, sampler, depth + 1) * abs_dot3(&wi, &ns) / pdf;
        } else {
            return Spectrum::zero();
        }
    }
}

pub fn uniform_sample_all_lights(
    it: &Interaction,
    scene: &Scene,
    sampler: &mut dyn Sampler,
    n_light_samples: &[u32],
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
            for k in 0..n_samples as usize {
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

/// Estimate direct lighting for only one randomly chosen light and
/// multiply the result by the number of lights to compensate.
pub fn uniform_sample_one_light(
    it: &Interaction,
    scene: &Scene,
    sampler: &mut dyn Sampler,
    handle_media: bool,
    light_distrib: Option<Arc<Distribution1D>>,
) -> Spectrum<SPECTRUM_N> {
    // ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    let n_lights = scene.lights.len();
    if n_lights == 0 {
        return Spectrum::zero();
    }
    let light_num;
    let mut light_pdf = 0.0;

    match light_distrib {
        Some(dis) => {
            light_num = dis.sample_discrete(sampler.get_1d(), Some(&mut light_pdf));
            if light_pdf == 0.0 {
                return Spectrum::zero();
            }
        }
        None => {
            light_num = ((sampler.get_1d() * n_lights as f64) as usize).min(n_lights - 1);
            light_pdf = 1.0 / n_lights as f64;
        }
    }

    let light = (&scene.lights[light_num]).clone();
    let u_light = sampler.get_2d();
    let u_scattering = sampler.get_2d();
    estimate_direct(
        it,
        &u_scattering,
        light,
        &u_light,
        scene,
        sampler,
        handle_media,
        false,
    ) / light_pdf
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
                if let Some(phase) = &mi.phase {
                    let p = phase.p(&mi.ist.wo, &wi);
                    f = Spectrum::from(p);
                    scattering_pdf = p;
                } else {
                    f = Spectrum::zero();
                }
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
                if let Some(phase) = &mi.phase {
                    let p = phase.sample_p(&mi.ist.wo, &mut wi, *u_scattering);
                    f = Spectrum::from(p);
                    scattering_pdf = p;
                } else {
                    f = Spectrum::zero();
                }
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

pub mod ao;
pub mod directlighting;
pub mod path;
// pub mod sppm;
pub mod volpath;
