use std::{
    f64::INFINITY,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    film::Film,
    geometry::{faceforward, Bounds2f, Normal3f, Point2f, Point3f, Ray, RayDifferential, Vector3f},
    interaction::Interaction,
    lowdiscrepancy::radical_inverse,
    medium::MediumOpArc,
    misc::{copy_option_arc, lerp, quadratic},
    reflection::refract,
    spectrum::Spectrum,
    transform::{ToWorld, Transform},
    SPECTRUM_N,
};

#[derive(Debug, Default, Copy, Clone)]
pub struct CameraSample {
    pub p_film: Point2f, // the point on the film
    pub p_lens: Point2f, // the point on the lens that the ray pass through
    pub time: f64,
}

impl CameraSample {
    pub fn new(p_film: Point2f, p_lens: Point2f, time: f64) -> Self {
        Self {
            p_film,
            p_lens,
            time,
        }
    }
}

pub trait ICamera: ToWorld + Debug {
    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f64;
    fn generate_ray_differential(&self, sample: &CameraSample, rd: &mut RayDifferential) -> f64 {
        // Float wt = GenerateRay(sample, rd);
        let wt = self.generate_ray(sample, &mut rd.ray);
        // if (wt == 0) return 0;
        if wt == 0.0 {
            return 0.0;
        }
        // Find camera ray after shifting a fraction of a pixel in the $x$ direction
        let mut wtx = 0_f64;
        for eps in [0.05, -0.05].iter() {
            let mut sshift = *sample;
            sshift.p_film.x += eps;
            let mut rx = Ray::default();
            wtx = self.generate_ray(&sshift, &mut rx);
            rd.rx_origin = rd.ray.o + (rx.o - rd.ray.o) / (*eps);
            rd.rx_direction = rd.ray.d + (rx.d - rd.ray.d) / (*eps);
            if wtx != 0_f64 {
                break;
            }
        }
        if wtx == 0_f64 {
            return 0_f64;
        }

        // Find camera ray after shifting a fraction of a pixel in the $y$ direction
        let mut wty = 0_f64;
        for eps in [0.05, -0.05].iter() {
            let mut sshift = *sample;
            sshift.p_film.y += eps;
            let mut ry = Ray::default();
            wty = self.generate_ray(&sshift, &mut ry);
            rd.ry_origin = rd.ray.o + (ry.o - rd.ray.o) / (*eps);
            rd.ry_direction = rd.ray.d + (ry.d - rd.ray.d) / (*eps);
            if wty != 0_f64 {
                break;
            }
        }
        if wty == 0_f64 {
            return 0_f64;
        }

        rd.has_differentials = true;
        wt
    }
    fn we(&self, _ray: &Ray, _p_raster2: &Point2f) -> Spectrum<SPECTRUM_N> {
        unimplemented!()
    }
    fn pdf_we(&self, _ray: &Ray, _pdf_pos: &mut f64, _pdf_dir: &mut f64) {
        unimplemented!()
    }
    fn sample_wi(
        &self,
        _ref_int: &Interaction,
        _u: &Point2f,
        _wi: &Vector3f,
        _pdf: f64,
        _p_raster: Point2f,
    ) -> Spectrum<SPECTRUM_N> {
        unimplemented!()
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct LensElementInterface {
    curvature_radius: f64,
    thickness: f64,
    eta: f64,
    aperture_radius: f64,
}

impl LensElementInterface {
    pub fn new(curvature_radius: f64, thickness: f64, eta: f64, aperture_radius: f64) -> Self {
        Self {
            curvature_radius,
            thickness,
            eta,
            aperture_radius,
        }
    }
}

#[derive(Debug)]
pub struct RealisticCamera {
    camera_to_world: Transform,
    pub shutter_open: f64,
    pub shutter_close: f64,

    pub film: Arc<Film>,
    pub medium: MediumOpArc,

    element_interfaces: Vec<LensElementInterface>,
    exit_pupil_bounds: Vec<Bounds2f>,
    simple_weighting: bool,
}

impl RealisticCamera {
    pub fn new(
        camera_to_world: Transform,
        shutter_open: f64,
        shutter_close: f64,
        aperture_diameter: f64,
        focus_distance: f64,
        film: Arc<Film>,
        medium: MediumOpArc,
        lens_data: &[f64],
        simple_weighting: bool,
    ) -> Self {
        assert!(lens_data.len() % 4 == 0);
        let mut element_interfaces = Vec::with_capacity(lens_data.len() / 4);
        for idx in 0..lens_data.len() / 4 {
            let i = idx * 4;
            let mut aperture_radius = lens_data[i + 3];
            if lens_data[i] == 0.0 {
                if aperture_diameter > lens_data[i + 3] {
                    // Warning(
                    //     "Specified aperture diameter %f is greater than maximum "
                    //     "possible %f.  Clamping it.",
                    //     apertureDiameter, lensData[i + 3]);
                } else {
                    aperture_radius = aperture_diameter;
                }
            }
            element_interfaces.push(LensElementInterface::new(
                lens_data[i] * 0.001,
                lens_data[i + 1] * 0.001,
                lens_data[i + 2],
                aperture_radius * 0.001 / 2.0,
            ));
        }
        let mut cam = Self {
            camera_to_world,
            shutter_open,
            shutter_close,
            film,
            medium,
            element_interfaces,
            exit_pupil_bounds: vec![],
            simple_weighting,
        };
        // Compute lens--film distance for given focus distance
        let tmp_thickness = cam.focus_thick_lens(focus_distance);
        if let Some(e) = cam.element_interfaces.last_mut() {
            e.thickness = tmp_thickness;
        }

        let n_samples = 64_usize;
        let exbs = Arc::new(Mutex::new(Vec::with_capacity(n_samples)));
        (0..n_samples).into_par_iter().for_each(|i| {
            let r0 = i as f64 / n_samples as f64 * cam.film.diagonal / 2.0;
            let r1 = (i + 1) as f64 / n_samples as f64 * cam.film.diagonal / 2.0;
            let b = cam.bound_exit_pupil(r0, r1);
            exbs.lock().unwrap().push(b);
        });
        cam.exit_pupil_bounds.reserve(n_samples);
        for b in exbs.lock().unwrap().to_owned() {
            cam.exit_pupil_bounds.push(b);
        }
        return cam;
    }

    fn lens_rear_z(&self) -> f64 {
        self.element_interfaces
            .last()
            .expect("Error Getting Last Lens Element")
            .thickness
    }
    fn lens_front_z(&self) -> f64 {
        let mut z_sum = 0_f64;
        for element in &self.element_interfaces {
            z_sum += element.thickness;
        }
        z_sum
    }
    fn rear_element_radius(&self) -> f64 {
        self.element_interfaces
            .last()
            .expect("Error Getting Last Lens Element")
            .aperture_radius
    }
    fn trace_lenses_from_film(&self, r_camera: &Ray) -> Option<Ray> {
        let r_out;
        let mut element_z = 0_f64;

        // Transform _rCamera_ from camera to lens system space
        let camera_to_lens = Transform::scale(1.0, 1.0, -1.0);
        let mut r_lens = camera_to_lens.t(r_camera);
        for i in (0..(self.element_interfaces.len())).rev() {
            let element = self.element_interfaces[i];

            // Update ray from film accounting for interaction with _element_
            element_z -= element.thickness;
            // Compute intersection of ray with lens element
            let mut t: f64 = 0_f64;
            let mut n: Normal3f = Normal3f::default();
            let is_stop = element.curvature_radius == 0.0;

            if is_stop {
                // The refracted ray computed in the previous lens element
                // interface may be pointed towards film plane(+z) in some
                // extreme situations; in such cases, 't' becomes negative.
                if r_lens.d.z >= 0.0 {
                    return None;
                }
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;

                if !self.intersect_spherical_element(radius, z_center, &r_lens, &mut t, &mut n) {
                    return None;
                }
            }
            assert!(t >= 0_f64);

            // Test intersection point against element aperture
            let p_hit = r_lens.position(t);
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 >= element.aperture_radius * element.aperture_radius {
                return None;
            }
            r_lens.o = p_hit;
            // Update ray path for element interface interaction
            if !is_stop {
                let mut w = Vector3f::default();
                let eta_i = element.eta;
                let eta_t = if i > 0 && self.element_interfaces[i - 1].eta != 0.0 {
                    self.element_interfaces[i - 1].eta
                } else {
                    1.0
                };
                if !refract(&-r_lens.d.normalize(), &n, eta_i / eta_t, &mut w) {
                    return None;
                }
                r_lens.d = w;
            }
        }

        // Transform _rLens_ from lens system space back to camera space
        let lens_to_camera = Transform::scale(1.0, 1.0, -1.0);
        r_out = lens_to_camera.t(&r_lens);

        return Some(r_out);
    }
    fn intersect_spherical_element(
        &self,
        radius: f64,
        z_center: f64,
        ray: &Ray,
        t: &mut f64,
        n: &mut Normal3f,
    ) -> bool {
        // Compute _t0_ and _t1_ for ray--element intersection
        let o = ray.o - Vector3f::new(0.0, 0.0, z_center);
        let a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        let b = 2.0 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        let c = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        let mut t0 = 0_f64;
        let mut t1 = 0_f64;
        if !quadratic(a, b, c, &mut t0, &mut t1) {
            return false;
        }
        // Select intersection $t$ based on ray direction and element curvature
        let use_closer_t = (ray.d.z > 0.0) ^ (radius < 0.0);
        *t = if use_closer_t {
            f64::min(t0, t1)
        } else {
            f64::max(t0, t1)
        };
        if *t < 0.0 {
            return false;
        }

        // Compute surface normal of element at ray intersection point
        *n = Normal3f::from(Vector3f::from(o + ray.d * *t));
        *n = faceforward(&(*n).normalize(), &-ray.d);
        return true;
    }
    fn trace_lenses_from_scene(&self, r_camera: &Ray) -> Option<Ray> {
        let mut element_z = -self.lens_front_z();
        // Transform _rCamera_ from camera to lens system space
        let camera_to_lens = Transform::scale(1.0, 1.0, -1.0);
        let mut r_lens = camera_to_lens.t(r_camera);

        for i in (0..(self.element_interfaces.len())).rev() {
            let element = self.element_interfaces[i];
            // Compute intersection of ray with lens element
            let mut t: f64 = 0_f64;
            let mut n: Normal3f = Normal3f::default();
            let is_stop = element.curvature_radius == 0.0;
            if is_stop {
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;

                if !self.intersect_spherical_element(radius, z_center, &r_lens, &mut t, &mut n) {
                    return None;
                }
            }
            assert!(t >= 0_f64);

            // Test intersection point against element aperture
            let p_hit = r_lens.position(t);
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 >= element.aperture_radius * element.aperture_radius {
                return None;
            }
            r_lens.o = p_hit;

            // Update ray path for from-scene element interface interaction
            if !is_stop {
                let mut wt = Vector3f::default();
                let eta_i = if i == 0 || self.element_interfaces[i - 1].eta == 0_f64 {
                    1_f64
                } else {
                    self.element_interfaces[i - 1].eta
                };
                let eta_t = if element.eta != 0_f64 {
                    element.eta
                } else {
                    1_f64
                };
                if !refract(&-r_lens.d.normalize(), &n, eta_i / eta_t, &mut wt) {
                    return None;
                }
                r_lens.d = wt;
            }
            element_z += element.thickness;
        }

        // Transform _rLens_ from lens system space back to camera space
        let lens_to_camera = Transform::scale(1.0, 1.0, -1.0);
        let r_out = lens_to_camera.t(&r_lens);

        return Some(r_out);
    }
    fn draw_lens_system(&self) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn draw_ray_path_from_film(&self, _r: &Ray, _arrow: bool, _to_optical_intercept: bool) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn draw_ray_path_from_scene(&self, _r: &Ray, _arrow: bool, _to_optical_intercept: bool) {
        unimplemented!("cause I'm a lazy pig");
    }
    // THE THICK LENS APPROXIMATION
    fn compute_cardinal_points(r_in: &Ray, r_out: &Ray) -> (f64, f64) {
        let tf = -r_out.o.x / r_out.d.x;
        let fz = -r_out.position(tf).z;
        let tp = (r_in.o.x - r_out.o.x) / r_out.d.x;

        let pz = -r_out.position(tp).z;
        (pz, fz)
    }
    fn compute_thick_lens_approximation(&self) -> ([f64; 2], [f64; 2]) {
        // Find height $x$ from optical axis for parallel rays
        let x = 0.001 * self.film.diagonal;
        // Compute cardinal points for film side of lens system
        let r_scene = Ray::new_od(
            Point3f::new(x, 0.0, self.lens_front_z() + 1.0),
            Vector3f::new(0.0, 0.0, -1.0),
        );

        let r_film = self.trace_lenses_from_scene(&r_scene).expect("Unable to trace ray from scene to film for thick lens approximation. Is aperture stop extremely small?");
        let (pz0, fz0) = Self::compute_cardinal_points(&r_scene, &r_film);

        // Compute cardinal points for scene side of lens system
        let r_film = Ray::new_od(
            Point3f::new(x, 0.0, self.lens_rear_z() - 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
        );
        let r_scene = self.trace_lenses_from_film(&r_film).expect("Unable to trace ray from film to scene for thick lens approximation. Is aperture stop extremely small?");
        let (pz1, fz1) = Self::compute_cardinal_points(&r_film, &r_scene);
        ([pz0, pz1], [fz0, fz1])
    }
    fn focus_thick_lens(&self, focus_distance: f64) -> f64 {
        let (pz, fz) = self.compute_thick_lens_approximation();
        println!(
            "Secondary focal point f\' = {}, Secondary principal plane p\' = {}",
            fz[0], pz[0]
        );
        println!(
            "Primary focal point f = {}, Primary principal plane p = {}",
            fz[1], pz[1]
        );
        println!(
            "effective focal length = {}, lens thickness = {}",
            fz[0] - pz[0],
            pz[0] - pz[1]
        );
        // Compute translation of lens, _delta_, to focus at _focusDistance_
        let f = fz[0] - pz[0];
        let z = -focus_distance;
        let c = (pz[1] - z - pz[0]) * (pz[1] - z - 4.0 * f - pz[0]);

        assert!(c>0.0, "Coefficient must be positive. It looks focusDistance: {} is too short for a given lenses configuration", focus_distance);
        let delta = 0.5 * (pz[1] - z + pz[0] - c.sqrt());
        self.element_interfaces
            .last()
            .expect("Error Getting Last Lens Element")
            .thickness
            + delta
    }
    fn focus_binary_search(&self, focus_distance: f64) -> f64 {
        let mut film_distance_lower: f64;
        let mut film_distance_upper: f64;
        // Find _filmDistanceLower_, _filmDistanceUpper_ that bound focus distance
        let tmp_focal_distance = self.focus_thick_lens(focus_distance);
        film_distance_lower = tmp_focal_distance;
        film_distance_upper = tmp_focal_distance;
        while self.focus_distance(film_distance_lower) > focus_distance {
            film_distance_lower *= 1.005;
        }
        while self.focus_distance(film_distance_upper) < focus_distance {
            film_distance_upper /= 1.005;
        }

        for _i in 0..20 {
            let fmid = 0.5 * (film_distance_lower + film_distance_upper);
            let mid_focus = self.focus_distance(fmid);
            if mid_focus < focus_distance {
                film_distance_lower = fmid;
            } else {
                film_distance_upper = fmid;
            }
        }
        0.5 * (film_distance_lower + film_distance_upper)
    }
    fn focus_distance(&self, film_dist: f64) -> f64 {
        // Find offset ray from film center through lens
        let bounds = self.bound_exit_pupil(0.0, 0.001 * self.film.diagonal);
        let scale_factors = [0.1, 0.01, 0.001];
        let mut lu = 0.0;
        let mut ray: Option<Ray> = None;
        // Try some different and decreasing scaling factor to find focus ray
        // more quickly when `aperturediameter` is too small.
        // (e.g. 2 [mm] for `aperturediameter` with wide.22mm.dat),
        for scale in scale_factors.iter() {
            lu = scale * bounds.p_max[0];
            ray = self.trace_lenses_from_film(&Ray::new_od(
                Point3f::new(0.0, 0.0, self.lens_rear_z() - film_dist),
                Vector3f::new(lu, 0.0, film_dist),
            ));
            if ray.is_some() {
                break;
            }
        }

        match ray {
            Some(r) => {
                // Compute distance _zFocus_ where ray intersects the principal axis
                let t_focus = -r.o.x / r.d.x;
                let mut z_focus = r.position(t_focus).z;
                // if (zFocus < 0) zFocus = Infinity;
                if z_focus < 0.0 {
                    z_focus = INFINITY;
                }
                return z_focus;
            }
            None => {
                eprintln!("Focus ray at lens pos({},0) didn't make it through the lenses with film distance {}", lu, film_dist);
                return INFINITY;
            }
        }
    }
    fn bound_exit_pupil(&self, p_film_x0: f64, p_film_x1: f64) -> Bounds2f {
        let mut pupil_bounds = Bounds2f::default();
        // Sample a collection of points on the rear lens to find exit pupil
        let n_samples = 1024 * 1024;
        let mut n_exiting_rays = 0;

        // Compute bounding box of projection of rear element on sampling plane
        let rear_radius = self.rear_element_radius();
        let proj_rear_bounds = Bounds2f::new(
            Point2f::new(-1.5 * rear_radius, -1.5 * rear_radius),
            Point2f::new(-1.5 * rear_radius, -1.5 * rear_radius),
        );

        for i in 0..n_samples {
            // Find location of sample points on $x$ segment and rear lens element
            let p_film = Point3f::new(
                lerp((i as f64 + 0.5) / n_samples as f64, p_film_x0, p_film_x1),
                0.0,
                0.0,
            );
            let u = [radical_inverse(0, i), radical_inverse(1, i)];

            let p_rear = Point3f::new(
                lerp(u[0], proj_rear_bounds.p_min.x, proj_rear_bounds.p_min.x),
                lerp(u[1], proj_rear_bounds.p_min.y, proj_rear_bounds.p_min.y),
                self.lens_rear_z(),
            );
            // Expand pupil bounds if ray makes it through the lens system
            if Bounds2f::inside(&Point2f::new(p_rear.x, p_rear.y), &pupil_bounds)
                || self
                    .trace_lenses_from_film(&Ray::new_od(p_film, p_rear - p_film))
                    .is_some()
            {
                pupil_bounds = Bounds2f::union(&pupil_bounds, &Point2f::new(p_rear.x, p_rear.y));
                n_exiting_rays += 1;
            }
        }
        // Return entire element bounds if no rays made it through the lens system
        if n_exiting_rays == 0 {
            return proj_rear_bounds;
        }

        // Expand bounds to account for sample spacing
        pupil_bounds = pupil_bounds
            .expand(2.0 * proj_rear_bounds.diagonal().length() / (n_samples as f64).sqrt());
        pupil_bounds
    }
    fn render_exit_pupil(&self, _sx: f64, _sy: f64, _filename: &str) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn sample_exit_pupil(&self, p_film: &Point2f, lens_sample: &Point2f) -> (Point3f, f64) {
        // Find exit pupil bound for sample distance from film center
        let r_film = (p_film.x * p_film.x + p_film.y * p_film.y).sqrt();
        let mut r_index =
            (r_film / (self.film.diagonal / 2.0)) as usize * self.exit_pupil_bounds.len();
        r_index = r_index.min(self.exit_pupil_bounds.len() - 1);
        let pupil_bounds = self.exit_pupil_bounds[r_index];

        // Generate sample point inside exit pupil bound
        let p_lens = pupil_bounds.lerp(lens_sample);
        // Return sample point rotated by angle of _pFilm_ with $+x$ axis
        let sin_theta = if r_film != 0.0 {
            p_film.y / r_film
        } else {
            0.0
        };
        let cos_theta = if r_film != 0.0 {
            p_film.x / r_film
        } else {
            1.0
        };
        (
            Point3f::new(
                cos_theta * p_lens.x - sin_theta * p_lens.y,
                sin_theta * p_lens.x + cos_theta * p_lens.y,
                self.lens_rear_z(),
            ),
            pupil_bounds.area(),
        )
    }
    fn test_exit_pupil_bounds(&self) {
        unimplemented!("cause I'm a lazy pig");
    }
}

impl ToWorld for RealisticCamera {
    fn to_world(&self) -> &Transform {
        &self.camera_to_world
    }
}

impl RealisticCamera {
    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f64 {
        // Find point on film, _pFilm_, corresponding to _sample.pFilm_
        let s = Point2f::new(
            sample.p_film.x / self.film.full_resolution.x as f64,
            sample.p_film.y / self.film.full_resolution.y as f64,
        );

        let p_film2 = self.film.get_physical_extent().lerp(&s);
        let p_film = Point3f::new(-p_film2.x, p_film2.y, 0.0);

        // Trace ray from _pFilm_ through lens system
        let (p_rear, exit_pupil_bounds_area) =
            self.sample_exit_pupil(&Point2f::new(p_film.x, p_film.y), &sample.p_lens);
        let r_film = Ray::new(
            p_film,
            p_rear - p_film,
            f64::INFINITY,
            lerp(sample.time, self.shutter_open, self.shutter_close),
            None,
        );

        match self.trace_lenses_from_film(&r_film) {
            Some(r) => {
                *ray = r;
            }
            None => {
                return 0.0;
            }
        }

        // Finish initialization of _RealisticCamera_ ray
        *ray = self.to_world().t(ray);
        ray.d = ray.d.normalize();
        ray.medium = copy_option_arc(&self.medium);

        // Return weighting for _RealisticCamera_ ray
        let cos_theta = r_film.d.normalize().z;
        let cos4_theta = (cos_theta * cos_theta) * (cos_theta * cos_theta);
        if self.simple_weighting {
            return cos4_theta * exit_pupil_bounds_area / self.exit_pupil_bounds[0].area();
        } else {
            return (self.shutter_close - self.shutter_open)
                * (cos4_theta * exit_pupil_bounds_area)
                / self.lens_rear_z()
                * self.lens_rear_z();
        }
    }

    pub fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        rd: &mut RayDifferential,
    ) -> f64 {
        let wt = self.generate_ray(sample, &mut rd.ray);
        if wt == 0.0 {
            return 0.0;
        }
        // Find camera ray after shifting a fraction of a pixel in the $x$ direction
        let mut wtx = 0_f64;
        for eps in [0.05, -0.05].iter() {
            let mut sshift = *sample;
            sshift.p_film.x += eps;
            let mut rx = Ray::default();
            wtx = self.generate_ray(&sshift, &mut rx);
            rd.rx_origin = rd.ray.o + (rx.o - rd.ray.o) / (*eps);
            rd.rx_direction = rd.ray.d + (rx.d - rd.ray.d) / (*eps);
            if wtx != 0_f64 {
                break;
            }
        }
        if wtx == 0_f64 {
            return 0_f64;
        }

        // Find camera ray after shifting a fraction of a pixel in the $y$ direction
        let mut wty = 0_f64;
        for eps in [0.05, -0.05].iter() {
            let mut sshift = *sample;
            sshift.p_film.y += eps;
            let mut ry = Ray::default();
            wty = self.generate_ray(&sshift, &mut ry);
            rd.ry_origin = rd.ray.o + (ry.o - rd.ray.o) / (*eps);
            rd.ry_direction = rd.ray.d + (ry.d - rd.ray.d) / (*eps);
            if wty != 0_f64 {
                break;
            }
        }
        if wty == 0_f64 {
            return 0_f64;
        }

        rd.has_differentials = true;
        wt
    }
}
