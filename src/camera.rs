use std::{f64::INFINITY, sync::Arc};

use crate::{
    film::Film,
    filters::IFilter,
    geometry::{
        faceforward, Bounds2f, Normal3f, Point2f, Point2i, Point3f, Ray, RayDifferential, Vector3f,
    },
    interaction::Interaction,
    lowdiscrepancy::radical_inverse,
    medium::Medium,
    misc::{concentric_sample_disk, copy_option_arc, lerp, quadratic},
    reflection::refract,
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    transform::Transform,
};

pub struct CameraData<T>
where
    T: IFilter,
{
    pub camera_to_world: Transform,
    pub shutter_open: f64,
    pub shutter_close: f64,

    pub film: Arc<Film<T>>,
    pub medium: Medium,
}

impl<T: IFilter> CameraData<T> {
    pub fn new(
        camera_to_world: Transform,
        shutter_open: f64,
        shutter_close: f64,
        film: Arc<Film<T>>,
        medium: Medium,
    ) -> Self {
        Self {
            camera_to_world,
            shutter_open,
            shutter_close,
            film,
            medium,
        }
    }
}

pub struct PerspectiveCamera {
    pub camera_to_world: Transform,
    pub resolution: Point2i,
    pub shutter_open: f64,
    pub shutter_close: f64,
    pub raster_to_camera: Transform,
    pub lens_radius: f64,
    pub focal_distance: f64,

    pub dx_camera: Vector3f,
    pub dy_camera: Vector3f,

    pub a: f64,
}

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

pub trait ICamera {
    fn generate_ray(&mut self, sample: &CameraSample, ray: &mut Ray) -> f64;
    fn generate_ray_differential(
        &mut self,
        sample: &CameraSample,
        rd: &mut RayDifferential,
    ) -> f64 {
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
    fn we(ray: &Ray, p_raster2: &Point2f) -> Spectrum<SPECTRUM_N> {
        unimplemented!()
    }
    fn pdf_we(ray: &Ray, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        unimplemented!()
    }
    fn sample__wi(
        ref_int: &Interaction,
        u: &Point2f,
        wi: &Vector3f,
        pdf: f64,
        p_raster: Point2f,
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

pub struct RealisticCamera<T>
where
    T: IFilter,
{
    camera: CameraData<T>,

    element_interfaces: Vec<LensElementInterface>,
    exit_pupil_bounds: Vec<Bounds2f>,
    simple_weighting: bool,
}

impl<T> RealisticCamera<T>
where
    T: IFilter,
{
    pub fn new(
        camera: CameraData<T>,
        element_interfaces: Vec<LensElementInterface>,
        exit_pupil_bounds: Vec<Bounds2f>,
        simple_weighting: bool,
    ) -> Self {
        Self {
            camera,
            element_interfaces,
            exit_pupil_bounds,
            simple_weighting,
        }
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
        let mut r_lens = camera_to_lens.transform_ray(r_camera);
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
        r_out = lens_to_camera.transform_ray(&r_lens);

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
        let mut r_lens = camera_to_lens.transform_ray(r_camera);

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
        let r_out = lens_to_camera.transform_ray(&r_lens);

        return Some(r_out);
    }
    fn draw_lens_system(&self) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn draw_ray_path_from_film(&self, r: &Ray, arrow: bool, to_optical_intercept: bool) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn draw_ray_path_from_scene(&self, r: &Ray, arrow: bool, to_optical_intercept: bool) {
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
        let x = 0.001 * self.camera.film.diagonal;
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

        assert!(c>0.0, format!("Coefficient must be positive. It looks focusDistance: {} is too short for a given lenses configuration", focus_distance));
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
        let bounds = self.bound_exit_pupil(0.0, 0.001 * self.camera.film.diagonal);
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
    fn render_exit_pupil(&self, sx: f64, sy: f64, filename: &str) {
        unimplemented!("cause I'm a lazy pig");
    }
    fn sample_exit_pupil(&self, p_film: &Point2f, lens_sample: &Point2f) -> (Point3f, f64) {
        // Find exit pupil bound for sample distance from film center
        let r_film = (p_film.x * p_film.x + p_film.y * p_film.y).sqrt();
        let mut r_index =
            (r_film / (self.camera.film.diagonal / 2.0)) as usize * self.exit_pupil_bounds.len();
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

impl<T> ICamera for RealisticCamera<T>
where
    T: IFilter,
{
    fn generate_ray(&mut self, sample: &CameraSample, ray: &mut Ray) -> f64 {
        // Find point on film, _pFilm_, corresponding to _sample.pFilm_
        let s = Point2f::new(
            sample.p_film.x / self.camera.film.full_resolution.x as f64,
            sample.p_film.y / self.camera.film.full_resolution.y as f64,
        );

        let p_film2 = self.camera.film.get_physical_extent().lerp(&s);
        let p_film = Point3f::new(-p_film2.x, p_film2.y, 0.0);

        // Trace ray from _pFilm_ through lens system
        let (p_rear, exit_pupil_bounds_area) =
            self.sample_exit_pupil(&Point2f::new(p_film.x, p_film.y), &sample.p_lens);
        let r_film = Ray::new(
            p_film,
            p_rear - p_film,
            f64::INFINITY,
            lerp(
                sample.time,
                self.camera.shutter_open,
                self.camera.shutter_close,
            ),
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
        *ray = self.camera.camera_to_world.transform_ray(ray);
        ray.d = ray.d.normalize();
        ray.medium = copy_option_arc(&self.camera.medium);

        // Return weighting for _RealisticCamera_ ray
        let cos_theta = r_film.d.normalize().z;
        let cos4_theta = (cos_theta * cos_theta) * (cos_theta * cos_theta);
        if self.simple_weighting {
            return cos4_theta * exit_pupil_bounds_area / self.exit_pupil_bounds[0].area();
        } else {
            return (self.camera.shutter_close - self.camera.shutter_open)
                * (cos4_theta * exit_pupil_bounds_area)
                / self.lens_rear_z()
                * self.lens_rear_z();
        }
    }
}

impl PerspectiveCamera {
    pub fn new(
        camera_to_world: Transform,
        screen_window: Bounds2f,
        resolution: Point2i,
        fov: f64,
        shutter_open: f64,
        shutter_close: f64,
        lens_radius: f64,
        focal_distance: f64,
    ) -> Self {
        let camera_to_screen: Transform = Transform::perspective(fov, 1e-2, 1000.0);

        // compute projective camera screen transformations
        let scale1 = Transform::scale(resolution.x as f64, resolution.y as f64, 1.0);
        let scale2 = Transform::scale(
            1.0 / (screen_window.p_max.x - screen_window.p_min.x),
            1.0 / (screen_window.p_min.y - screen_window.p_max.y),
            1.0,
        );
        let translate = Transform::translate(&Vector3f {
            x: -screen_window.p_min.x,
            y: -screen_window.p_max.y,
            z: 0.0,
        });
        let screen_to_raster = scale1 * scale2 * translate;
        let raster_to_screen = Transform::inverse(&screen_to_raster);
        let raster_to_camera = Transform::inverse(&camera_to_screen) * raster_to_screen;
        // see perspective.cpp
        // compute differential changes in origin for perspective camera rays
        let dx_camera: Vector3f = raster_to_camera.transform_point(&Point3f {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }) - raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        let dy_camera: Vector3f = raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }) - raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        // compute image plane bounds at $z=1$ for _PerspectiveCamera_
        let mut p_min: Point3f = raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        // Point3f p_max = RasterToCamera(Point3f(res.x, res.y, 0));
        let mut p_max: Point3f = raster_to_camera.transform_point(&Point3f {
            x: resolution.x as f64,
            y: resolution.y as f64,
            z: 0.0,
        });
        p_min /= p_min.z;
        p_max /= p_max.z;
        let a: f64 = ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs();

        PerspectiveCamera {
            camera_to_world,
            resolution,
            shutter_open,
            shutter_close,
            raster_to_camera,
            lens_radius,
            focal_distance,
            dx_camera,
            dy_camera,
            a,
        }
    }
    pub fn create(
        cam2world: Transform,
        resolution: Point2i,
        fov: f64,
        shutteropen: f64,
        shutterclose: f64,
        lensradius: f64,
        focaldistance: f64,
    ) -> PerspectiveCamera {
        assert!(shutterclose >= shutteropen);
        let frame = resolution.x as f64 / resolution.y as f64;

        let mut screen: Bounds2f = Bounds2f::default();
        if frame > 1.0 {
            screen.p_min.x = -frame;
            screen.p_max.x = frame;
            screen.p_min.y = -1.0;
            screen.p_max.y = 1.0;
        } else {
            screen.p_min.x = -1.0;
            screen.p_max.x = 1.0;
            screen.p_min.y = -1.0 / frame;
            screen.p_max.y = 1.0 / frame;
        }

        PerspectiveCamera::new(
            cam2world,
            screen,
            resolution,
            fov,
            shutteropen,
            shutterclose,
            lensradius,
            focaldistance,
        )
    }

    // Camera
    pub fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f64 {
        // We will use a simplified version here
        let p_film: Point3f = Point3f {
            x: sample.p_film.x,
            y: sample.p_film.y,
            z: 0.0,
        };
        let p_camera = self.raster_to_camera.transform_point(&p_film);
        ray.d = Vector3f::from(p_camera).normalize();
        // 1/z` - 1/z = 1/f where z is object depth in scene; z` is focusing film distance from lens; f is focal distance
        if self.lens_radius > 0.0 {
            // sample point on lens
            let p_lens: Point2f = concentric_sample_disk(sample.p_lens) * self.lens_radius;
            // // compute point on plane of focus
            let ft = self.focal_distance / ray.d.z;
            let p_focus: Point3f = ray.position(ft);

            // // update ray for effect of lens
            ray.o = Point3f {
                x: p_lens.x,
                y: p_lens.y,
                z: 0.0,
            };
            ray.d = (p_focus - ray.o).normalize();
        }

        *ray = self.camera_to_world.transform_ray(&ray);
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perspective() {
        let t: Transform = Transform::look_at(
            &Point3f::new(0.0, 0.0, 0.0),
            &Point3f::new(0.0, 0.0, 1.0),
            &Vector3f::new(0.0, 1.0, 0.0),
        );

        let it: Transform = Transform {
            m: t.m_inv.clone(),
            m_inv: t.m.clone(),
        };
        let cam = PerspectiveCamera::create(it, Point2i::new(640, 640), 75.0, 0.0, 0.0, 0.05, 5.0);
        let mut r = Ray::default();
        cam.generate_ray(
            &CameraSample::new(Point2f::new(0.0, 1.0), Point2f::new(0.0, 0.01), 0.0),
            &mut r,
        );
        println!("{:?}", r);
    }
}
