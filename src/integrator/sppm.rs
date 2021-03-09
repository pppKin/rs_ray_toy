use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

use super::*;
use crate::{
    geometry::{max_component, Bounds3f, Point3f, Point3i},
    lowdiscrepancy::radical_inverse,
    material::TransportMode,
    misc::{clamp_t, lerp},
    reflection::{Bsdf, BXDF_DIFFUSE, BXDF_GLOSSY},
    samplers::{halton::Halton, GlobalSampler, StartPixel},
};

#[derive(Debug)]
pub struct SPPMIntegrator {
    cam: Arc<RealisticCamera>,

    init_search_radius: f64,
    n_iters: usize,
    max_depth: usize,
    photons_per_iter: usize,
    write_freq: usize,

    light_distr: Arc<Distribution1D>,
}

#[derive(Debug, Default, Clone)]
pub struct VisiblePoint {
    pub p: Point3f,
    pub wo: Vector3f,
    pub bsdf: Option<Bsdf>,
    pub beta: Spectrum<SPECTRUM_N>,
}

impl VisiblePoint {
    pub fn new(p: Point3f, wo: Vector3f, bsdf: Option<Bsdf>, beta: Spectrum<SPECTRUM_N>) -> Self {
        Self { p, wo, bsdf, beta }
    }
}

#[derive(Debug, Default)]
struct SPPMPixel {
    pub radius: f64,

    pub ld: Spectrum<SPECTRUM_N>,
    pub vp: VisiblePoint,
    pub phi: [f64; SPECTRUM_N],
    pub m: i32,
    pub n: f64,
    pub tau: Spectrum<SPECTRUM_N>,
}

struct SPPMPixelListNode {
    pixel: Arc<Mutex<SPPMPixel>>,

    next: Box<Option<SPPMPixelListNode>>,
}

impl SPPMPixelListNode {
    fn new(pixel: Arc<Mutex<SPPMPixel>>, next: Box<Option<SPPMPixelListNode>>) -> Self {
        Self { pixel, next }
    }
}

pub fn to_grid(p: &Point3f, bounds: &Bounds3f, grid_res: [i64; 3], pi: &mut Point3i) -> bool {
    let mut in_bounds = true;
    let pg = bounds.offset(p);
    for i in 0..3 {
        pi[i] = grid_res[i as usize] * pg[i] as i64;
        in_bounds &= pi[i] >= 0 && pi[i] < grid_res[i as usize];
        pi[i] = clamp_t(pi[i], 0, grid_res[i as usize] - 1);
    }
    in_bounds
}

fn hash(p: &Point3f, hash_size: usize) -> usize {
    ((p.x * 73856093.0) as usize ^ (p.y * 19349663.0) as usize ^ (p.z * 83492791.0) as usize)
        % hash_size
}

impl Integrator for SPPMIntegrator {
    fn render(&mut self, scene: &Scene) {
        // Initialize _pixelBounds_ and _pixels_ array for SPPM
        let pixel_bounds = self.cam.camera.film.cropped_pixel_bounds;
        let n_pixels = pixel_bounds.area();
        let mut tmp_pixels = vec![];
        for _pidx in 0..n_pixels as usize {
            let mut tmp = SPPMPixel::default();
            tmp.radius = self.init_search_radius;
            tmp_pixels.push(Arc::new(Mutex::new(tmp)));
        }
        let am_pixels = tmp_pixels;
        let inv_sqrt_spp = 1.0 / (self.n_iters as f64).sqrt();

        if scene.lights.len() == 0 {
            self.light_distr = Arc::new(Distribution1D::default());
        } else {
            let mut light_power = vec![];
            for light in &scene.lights {
                light_power.push(light.power().y());
            }
            self.light_distr = Arc::new(Distribution1D::new(light_power));
        }

        // Perform _nIterations_ of SPPM integration
        let halton = Halton::new(&pixel_bounds, true);
        let halton_sampler = Arc::new(GlobalSampler::new(self.n_iters as u64, halton));

        // Compute number of tiles to use for SPPM camera pass
        let pixel_extent = pixel_bounds.diagonal();
        let tile_size = 16;
        let n_tiles_x = ((pixel_extent.x + tile_size - 1) / tile_size) as usize;
        let n_tiles_y = ((pixel_extent.y + tile_size - 1) / tile_size) as usize;

        for iter in 0..self.n_iters {
            // Generate SPPM visible points
            (0..n_tiles_x).into_par_iter().for_each(|tile_x| {
                (0..n_tiles_y).into_par_iter().for_each(|tile_y| {
                    // Follow camera paths for _tile_ in image for SPPM
                    let mut ht_sampler = (*halton_sampler).clone();

                    // Compute _tileBounds_ for SPPM tile
                    let x0 = pixel_bounds.p_min.x + tile_x as i64 * tile_size;
                    let x1 = (x0 + tile_size).min(pixel_bounds.p_max.x);
                    let y0 = pixel_bounds.p_min.y + tile_y as i64 * tile_size;
                    let y1 = (y0 + tile_size).min(pixel_bounds.p_max.y);
                    let tile_bounds = Bounds2i::new(Point2i::new(x0, y0), Point2i::new(x1, y1));
                    for p_pixel in tile_bounds.into_iter() {
                        // Prepare _tileSampler_ for _pPixel_
                        ht_sampler.start_pixel(p_pixel);
                        ht_sampler.set_sample_number(iter as u64);
                        // Generate camera ray for pixel for SPPM
                        let camera_sample = ht_sampler.get_camerasample(&p_pixel);
                        let mut ray = RayDifferential::default();
                        let mut beta = Spectrum::<SPECTRUM_N>::from(
                            self.cam.generate_ray_differential(&camera_sample, &mut ray),
                        );
                        if beta.is_black() {
                            continue;
                        }
                        ray.scale_differentials(inv_sqrt_spp);

                        // Follow camera ray path until a visible point is created

                        // Get _SPPMPixel_ for _pPixel_
                        let p_pixel_o = p_pixel - pixel_bounds.p_min;
                        let pixel_offset = p_pixel_o.x
                            + p_pixel_o.y * (pixel_bounds.p_max.x - pixel_bounds.p_min.x);
                        let mut pixel = am_pixels[pixel_offset as usize].lock().unwrap();
                        let mut specular_bounce = false;
                        for depth in 0..self.max_depth {
                            let mut isect = SurfaceInteraction::default();

                            if !scene.intersect(&mut ray.ray, &mut isect) {
                                // Accumulate light contributions for ray with no intersection
                                for light in &scene.lights {
                                    pixel.ld += beta * light.le(&ray);
                                }
                                break;
                            }
                            // Process SPPM camera ray intersection

                            // Compute BSDF at SPPM camera ray intersection
                            isect.compute_scattering_functions(
                                &ray,
                                true,
                                crate::material::TransportMode::Radiance,
                            );
                            if isect.bsdf.is_none() {
                                ray = isect.ist.spawn_ray(ray.ray.d).into();
                                // depth -= 1;
                                continue;
                            }
                            // Accumulate direct illumination at SPPM camera ray intersection
                            let wo = -ray.ray.d;
                            if depth == 0 || specular_bounce {
                                pixel.ld += beta * isect.le(&wo);
                            }
                            pixel.ld += beta
                                * uniform_sample_one_light(
                                    &Interaction::Surface(isect.clone()),
                                    scene,
                                    &mut ht_sampler,
                                    false,
                                    None,
                                );
                            // Possibly create visible point and end camera path
                            if let Some(bsdf) = &isect.bsdf {
                                let is_diffuse = bsdf.num_components(
                                    BXDF_DIFFUSE | BXDF_REFLECTION | BXDF_TRANSMISSION,
                                ) > 0;
                                let is_glossy = bsdf.num_components(
                                    BXDF_GLOSSY | BXDF_REFLECTION | BXDF_TRANSMISSION,
                                ) > 0;

                                if is_diffuse || (is_glossy && depth == self.max_depth - 1) {
                                    pixel.vp = VisiblePoint::new(
                                        isect.ist.p,
                                        wo,
                                        Some(bsdf.clone()),
                                        beta,
                                    );
                                    break;
                                }

                                // Spawn ray from SPPM camera path vertex
                                if depth < self.max_depth - 1 {
                                    let mut wi = Vector3f::default();
                                    let mut pdf = 0.0;
                                    let mut flags = 0;
                                    let f = bsdf.sample_f(
                                        &wo,
                                        &mut wi,
                                        &ht_sampler.get_2d(),
                                        &mut pdf,
                                        BXDF_ALL,
                                        &mut flags,
                                    );
                                    if pdf == 0.0 || f.is_black() {
                                        break;
                                    }
                                    specular_bounce = (flags & BXDF_SPECULAR) != 0;
                                    beta *= f * abs_dot3(&wi, &isect.shading.n) / pdf;
                                    if beta.y() < 0.25 {
                                        let continue_prob = beta.y().min(1.0);
                                        if ht_sampler.get_1d() > continue_prob {
                                            break;
                                        }
                                        beta /= continue_prob;
                                    }
                                    ray = isect.ist.spawn_ray(wi).into();
                                }
                            }
                        }
                    }
                })
            });

            // Create grid of all SPPM visible points
            let mut grid_res = [0_i64; 3];
            let mut grid_bounds = Bounds3f::default();
            // Allocate grid for SPPM visible points
            let hash_size = n_pixels as usize;
            let mut grid: Vec<Arc<Mutex<Option<SPPMPixelListNode>>>> =
                Vec::with_capacity(hash_size);
            for _g_i in 0..hash_size {
                grid.push(Arc::new(Mutex::new(None)));
            }
            // Compute grid bounds for SPPM visible points
            let mut max_radius: f64 = 0.0;
            for i in 0..hash_size {
                let pixel = am_pixels[i].lock().unwrap();
                if pixel.vp.beta.is_black() {
                    continue;
                }

                let vp_bound = Bounds3f::new(pixel.vp.p, pixel.vp.p).expand(pixel.radius);
                grid_bounds = Bounds3f::union_bnd(&grid_bounds, &vp_bound);
                max_radius = max_radius.max(pixel.radius);
            }

            // Compute resolution of SPPM grid in each dimension
            let diag = grid_bounds.diagonal();
            let max_diag = max_component(&diag);
            assert!(max_diag > 0.0);
            let base_grid_res = max_diag / max_radius;
            for i in 0..3 {
                grid_res[i] = ((base_grid_res * diag[i as u8] / max_diag) as i64).max(1);
            }

            // Add visible points to SPPM grid
            (0..am_pixels.len()).into_par_iter().for_each(|pixel_idx| {
                let pixel = am_pixels[pixel_idx].lock().unwrap();
                if !pixel.vp.beta.is_black() {
                    // Add pixel's visible point to applicable grid cells
                    let radius = pixel.radius;
                    let mut p_min = Point3i::default();
                    let mut p_max = Point3i::default();

                    to_grid(
                        &(pixel.vp.p - Vector3f::new(radius, radius, radius)),
                        &grid_bounds,
                        grid_res,
                        &mut p_min,
                    );
                    to_grid(
                        &(pixel.vp.p + Vector3f::new(radius, radius, radius)),
                        &grid_bounds,
                        grid_res,
                        &mut p_max,
                    );

                    for z in p_min.z..=p_max.z {
                        for y in p_min.y..=p_max.y {
                            for x in p_min.x..=p_max.x {
                                // Add visible point to grid cell $(x, y, z)$
                                let h =
                                    hash(&Point3f::new(x as f64, y as f64, z as f64), hash_size);

                                let mut node = SPPMPixelListNode::new(
                                    am_pixels[pixel_idx].clone(),
                                    Box::new(None),
                                );

                                // Atomically add _node_ to the start of _grid[h]_'s linked list
                                let mut cur_start_node_opt = grid[h].lock().unwrap();
                                let old = (*cur_start_node_opt).take();
                                match old {
                                    Some(old_node) => {
                                        node.next = Box::new(Some(old_node));
                                    }
                                    None => {
                                        node.next = Box::new(None);
                                    }
                                }
                                (*cur_start_node_opt).replace(node);
                            }
                        }
                    }
                }
            });

            // Trace photons and accumulate contributions
            (0..=self.photons_per_iter)
                .into_par_iter()
                .for_each(|photon_index| {
                    // Follow photon path for _photonIndex_
                    let halton_index = (iter * self.photons_per_iter + photon_index) as u64;
                    let mut halton_dim = 0;

                    // Choose light to shoot photon from
                    let mut light_pdf = 0.0;
                    let light_sample = radical_inverse(halton_dim, halton_index);
                    halton_dim += 1;
                    let light_num = self
                        .light_distr
                        .sample_discrete(light_sample, Some(&mut light_pdf));
                    let light = scene.lights[light_num].clone();
                    // Compute sample values for photon ray leaving light source
                    let u_light0 = Point2f::new(
                        radical_inverse(halton_dim, halton_index),
                        radical_inverse(halton_dim, halton_index),
                    );
                    let u_light1 = Point2f::new(
                        radical_inverse(halton_dim + 2, halton_index),
                        radical_inverse(halton_dim + 3, halton_index),
                    );
                    let u_light_time = lerp(
                        radical_inverse(halton_dim + 2, halton_index),
                        self.cam.camera.shutter_open,
                        self.cam.camera.shutter_close,
                    );
                    halton_dim += 5;

                    // Generate _photonRay_ from light source and initialize _beta_
                    let mut photon_ray = RayDifferential::default();
                    let mut n_light = Normal3f::default();
                    let mut pdf_pos = 0.0;
                    let mut pdf_dir = 0.0;

                    let le = light.sample_le(
                        &u_light0,
                        &u_light1,
                        u_light_time,
                        &mut photon_ray.ray,
                        &mut n_light,
                        &mut pdf_pos,
                        &mut pdf_dir,
                    );
                    if pdf_pos == 0.0 || pdf_dir == 0.0 || le.is_black() {
                        return;
                    }

                    let mut beta = (le * abs_dot3(&n_light, &photon_ray.ray.d))
                        / (light_pdf * pdf_pos * pdf_dir);
                    if beta.is_black() {
                        return;
                    }
                    // Follow photon path through scene and record intersections
                    let mut isect = SurfaceInteraction::default();
                    for depth in 0..self.max_depth {
                        if !scene.intersect(&mut photon_ray.ray, &mut isect) {
                            break;
                        }

                        if depth > 0 {
                            // Add photon contribution to nearby visible points
                            let mut photon_grid_index = Point3i::default();
                            if to_grid(&isect.ist.p, &grid_bounds, grid_res, &mut photon_grid_index)
                            {
                                let h = hash(
                                    &Point3f::new(
                                        photon_grid_index.x as f64,
                                        photon_grid_index.y as f64,
                                        photon_grid_index.z as f64,
                                    ),
                                    hash_size,
                                );
                                // Add photon contribution to visible points in _grid[h]_
                                // walk this linked list
                                let mut cur_list_node_opt = grid[h].lock().unwrap();
                                let mut cur_node_opt = Box::new(cur_list_node_opt.take());
                                loop {
                                    if cur_node_opt.is_none() {
                                        break;
                                    }
                                    let cur_node = cur_node_opt.unwrap();
                                    let mut pixel = cur_node.pixel.lock().unwrap();
                                    let radius = pixel.radius;

                                    if (pixel.vp.p - isect.ist.p).length_squared()
                                        > (radius * radius)
                                    {
                                        cur_node_opt = cur_node.next;
                                        continue;
                                    }

                                    // Update _pixel_ $\Phi$ and $M$ for nearby photon
                                    let wi = -photon_ray.ray.d;
                                    if let Some(pixel_vp_bsdf) = &pixel.vp.bsdf {
                                        let phi =
                                            beta * pixel_vp_bsdf.f(&pixel.vp.wo, &wi, BXDF_ALL);
                                        for i in 0..SPECTRUM_N {
                                            pixel.phi[i] += phi[i];
                                        }
                                        pixel.m += 1;
                                    }

                                    cur_node_opt = cur_node.next;
                                }
                            }
                        }
                        // Sample new photon ray direction

                        // Compute BSDF at photon intersection point
                        isect.compute_scattering_functions(
                            &photon_ray,
                            true,
                            TransportMode::Importance,
                        );

                        if isect.bsdf.is_none() {
                            photon_ray = isect.ist.spawn_ray(photon_ray.ray.d).into();
                            // depth -= 1;
                            continue;
                        }

                        let mut wi = Vector3f::default();
                        let wo = -photon_ray.ray.d;
                        let mut pdf = 0.0;
                        let mut flags = 0;

                        // Generate _bsdfSample_ for outgoing photon sample
                        let bsdf_sample = Point2f::new(
                            radical_inverse(halton_dim, halton_index),
                            radical_inverse(halton_dim + 1, halton_index),
                        );
                        halton_dim += 2;
                        if let Some(photon_bsdf) = &isect.bsdf {
                            let fr = photon_bsdf.sample_f(
                                &wo,
                                &mut wi,
                                &bsdf_sample,
                                &mut pdf,
                                BXDF_ALL,
                                &mut flags,
                            );
                            if fr.is_black() || pdf == 0.0 {
                                break;
                            }
                            let bnew = beta * fr * abs_dot3(&wi, &isect.shading.n) / pdf;
                            // Possibly terminate photon path with Russian roulette
                            let q = (1.0 - bnew.y() / beta.y()).max(0.0);
                            if radical_inverse(halton_dim, halton_index) < q {
                                break;
                            } else {
                                halton_dim += 1;
                            }
                            beta = bnew / (1.0 - q);
                            photon_ray = isect.ist.spawn_ray(wi).into();
                        }
                    }
                });

            // Update pixel values from this pass's photons
            (0..n_pixels as usize).into_par_iter().for_each(|i| {
                let mut p = am_pixels[i].lock().unwrap();
                if p.m > 0 {
                    // Update pixel photon count, search radius, and $\tau$ from photons
                    let gamma = 2.0 / 3.0;
                    let n_new = p.n + gamma * p.m as f64;
                    let r_new = p.radius * (n_new / (p.n + p.m as f64));
                    let mut phi = Spectrum::<SPECTRUM_N>::zero();
                    for j in 0..SPECTRUM_N {
                        phi[j] = p.phi[j];
                    }
                    p.tau = (p.tau + p.vp.beta * phi) * (r_new * r_new) / (p.radius * p.radius);

                    p.n = n_new;
                    p.radius = r_new;
                    p.m = 0;

                    for j in 0..SPECTRUM_N {
                        p.phi[j] = 0.0;
                    }
                }
                p.vp.beta = Spectrum::zero();
                p.vp.bsdf = None;
            });

            // Periodically store SPPM image in film and write image
            if (iter + 1) == self.n_iters || ((iter + 1) % self.write_freq) == 0 {
                let x0 = pixel_bounds.p_min.x as usize;
                let x1 = pixel_bounds.p_max.x as usize;
                let np = (iter + 1) * self.photons_per_iter;
                let mut image = Vec::with_capacity(pixel_bounds.area() as usize);

                for y in pixel_bounds.p_min.y as usize..pixel_bounds.p_max.y as usize {
                    for x in x0..x1 {
                        // Compute radiance _L_ for SPPM pixel _pixel_
                        let pixel = am_pixels
                            [(y - pixel_bounds.p_min.y as usize) * (x1 - x0) + (x - x0)]
                            .lock()
                            .unwrap();
                        let mut l = pixel.ld / (iter + 1) as f64;
                        l += pixel.tau / (np as f64 * PI * pixel.radius * pixel.radius);
                        image.push(l);
                    }
                }
                self.cam.camera.film.set_image(&image);
                self.cam.camera.film.write_image(1.0);
            }
        }
    }
}
