use image::io::Reader as ImageReader;

use std::{
    f64::{consts::PI, INFINITY},
    sync::Arc,
};

use crate::{
    geometry::{spherical_phi, spherical_theta, vec3_coordinate_system, Bounds3f, Point2},
    mipmap::{ImageWrap, MIPMap},
    misc::{INV_2_PI, INV_PI},
    sampling::{concentric_sample_disk, Distribution2D},
    transform::Transform,
};

use super::*;

// InfiniteAreaLight Private Data
#[derive(Debug)]
pub struct InfiniteAreaLight {
    l_map: MIPMap,
    distribution: Arc<Distribution2D>,
    l: Spectrum<SPECTRUM_N>,
    w_light: Vector3f,
    world_center: Point3f,
    world_radius: f64,

    to_world: Transform,
    to_local: Transform,
    n_samples: usize,
    medium_interface: MediumInterface,
}

impl InfiniteAreaLight {
    pub fn new(
        to_world: Transform,
        l: Spectrum<SPECTRUM_N>,
        texmap: &str,
        n_samples: usize,
        medium_interface: MediumInterface,
        scene_world_bound: Bounds3f,
    ) -> Self {
        // Read texel data from _texmap_ and initialize _Lmap_
        let resolution;
        let texels;

        let decoded = ImageReader::open(texmap)
            .expect((format!("Failed to open {}", texmap)).as_str())
            .decode()
            .expect((format!("Failed to decode {}", texmap)).as_str());
        let img = decoded.into_rgb8();
        let width = img.width();
        let height = img.height();
        resolution = Point2::<usize>::new(width as usize, height as usize);

        let mut rgb_vec: Vec<Spectrum<SPECTRUM_N>> = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let tmp_pixel = img.get_pixel(x, y);
                let tmp = [
                    tmp_pixel[0] as f64 / 255.0,
                    tmp_pixel[1] as f64 / 255.0,
                    tmp_pixel[2] as f64 / 255.0,
                ];
                rgb_vec.push(Spectrum::from_rgb(tmp, SpectrumType::Reflectance));
            }
        }
        for y in 0..resolution.y / 2 {
            for x in 0..resolution.x {
                let o1 = (y * resolution.x + x) as usize;
                let o2 = ((resolution.y - 1 - y) * resolution.x + x) as usize;
                rgb_vec.swap(o1, o2);
            }
        }
        texels = rgb_vec;

        let l_map = MIPMap::create(resolution, &texels, false, 8.0, ImageWrap::Repeat);
        // Initialize sampling PDFs for infinite area light

        // Compute scalar-valued image _img_ from environment map
        let width = 2 * l_map.width();
        let height = 2 * l_map.height();
        let mut img = vec![];
        let fwidth: f64 = 0.5 as f64 / (width as f64).min(height as f64);
        for v in 0..height {
            let vp: f64 = (v as f64 + 0.5 as f64) / height as f64;
            let sin_theta: f64 = (PI * (v as f64 + 0.5 as f64) / height as f64).sin();
            for u in 0..width {
                let up: f64 = (u as f64 + 0.5 as f64) / width as f64;
                let st: Point2f = Point2f { x: up, y: vp };
                img.push(l_map.lookup_w(&st, fwidth).y() * sin_theta);
            }
        }
        // Compute sampling distributions for rows and columns of image
        let distribution = Distribution2D::new(img, width, height);

        let mut world_center = Point3f::default();
        let mut world_radius = 0.0;
        Bounds3f::bounding_sphere(&scene_world_bound, &mut world_center, &mut world_radius);
        Self {
            l_map: l_map,
            distribution: Arc::new(distribution),
            l: l,
            w_light: Vector3f::default(),
            world_center,
            world_radius,
            to_world,
            to_local: Transform::inverse(&to_world),
            n_samples,
            medium_interface,
        }
    }
}

impl ToWorld for InfiniteAreaLight {
    fn to_world(&self) -> &Transform {
        &self.to_world
    }
}

impl Light for InfiniteAreaLight {
    fn flags(&self) -> LightFlag {
        LIGHT_INFINITE
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn medium_interface(&self) -> &MediumInterface {
        &self.medium_interface
    }

    fn le(&self, r: &RayDifferential) -> Spectrum<SPECTRUM_N> {
        let w = self.to_local.t(&r.ray.d).normalize();
        let st = Point2f::new(spherical_phi(&w) * INV_2_PI, spherical_theta(&w) * INV_PI);
        self.l_map.lookup_w(&st, 0.0)
    }

    fn sample_li(
        &self,
        ref_ist: &BaseInteraction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f64,
        vis: &mut VisibilityTester,
    ) -> Spectrum<SPECTRUM_N> {
        // Find $(u,v)$ sample coordinates in infinite light texture
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample_continuous(*u, &mut map_pdf);
        if map_pdf == 0.0 {
            return (0.0).into();
        }

        // Convert infinite light sample point to direction
        let theta = uv[1] * PI;
        let phi = uv[0] * 2.0 * PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        *wi = self.to_world.t(&Vector3f::new(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
        ));
        // Compute PDF for sampled infinite light direction
        *pdf = map_pdf / (2.0 * PI * PI * sin_theta);
        if sin_theta == 0.0 {
            *pdf = 0.0;
        }

        // Return radiance value for infinite light direction
        *vis = VisibilityTester::new(
            ref_ist.clone(),
            BaseInteraction::new(
                ref_ist.p + *wi * (2.0 * self.world_radius),
                ref_ist.time,
                Vector3f::default(),
                Vector3f::default(),
                Normal3f::default(),
                Some(self.medium_interface().clone()),
            ),
        );
        self.l_map.lookup_w(&uv, 0.0)
    }

    fn power(&self) -> Spectrum<SPECTRUM_N> {
        self.l_map.lookup_w(&Point2f::new(0.5, 0.5), 0.5)
            * (PI * self.world_radius * self.world_radius)
    }

    fn pdf_li(&self, _ref_ist: &BaseInteraction, w: &Vector3f) -> f64 {
        // ProfilePhase _(Prof::LightPdf);
        let wi = self.to_world().t(w);
        let theta = spherical_theta(&wi);
        let phi = spherical_phi(&wi);
        let sin_theta = theta.sin();

        if sin_theta == 0.0 {
            return 0.0;
        }
        self.distribution
            .pdf(Point2f::new(phi * INV_2_PI, theta * INV_PI) / (2.0 * PI * PI * sin_theta))
    }

    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        time: f64,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f64,
        pdf_dir: &mut f64,
    ) -> Spectrum<SPECTRUM_N> {
        // Compute direction for infinite light sample ray
        let u = *u1;

        // Find $(u,v)$ sample coordinates in infinite light texture
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample_continuous(u, &mut map_pdf);
        if map_pdf == 0.0 {
            return (0.0).into();
        }

        let theta = uv[1] * PI;
        let phi = uv[0] * 2.0 * PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let d = -self.to_world().t(&Vector3f::new(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
        ));
        *n_light = d.into();

        // Compute origin for infinite light sample ray
        let mut v1 = Vector3f::default();
        let mut v2 = Vector3f::default();
        vec3_coordinate_system(&-d, &mut v1, &mut v2);
        let cd = concentric_sample_disk(*u2);
        let p_disk = self.world_center + (v1 * cd.x + v2 * cd.y) * self.world_radius;
        *ray = Ray::new(p_disk + (-d) * self.world_radius, d, INFINITY, time, None);

        // Compute _InfiniteAreaLight_ ray PDFs
        *pdf_dir = if sin_theta == 0.0 {
            0.0
        } else {
            map_pdf / (2.0 * PI * PI * sin_theta)
        };
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
        self.l_map.lookup_w(&uv, 0.0)
    }

    fn pdf_le(&self, ray: &Ray, _n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        // ProfilePhase _(Prof::LightPdf);
        let d = -self.to_world().t(&ray.d);
        let theta = spherical_theta(&d);
        let phi = spherical_phi(&d);
        let uv = Point2f::new(phi * INV_2_PI, theta * INV_PI);
        let map_pdf = self.distribution.pdf(uv);
        *pdf_dir = map_pdf / (2.0 * PI * PI * theta.sin());
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
    }
}
