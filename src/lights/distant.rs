use std::f64::{consts::PI, INFINITY};

use crate::{
    geometry::{vec3_coordinate_system, Bounds3f},
    misc::concentric_sample_disk,
    primitives::Primitive,
    transform::Transform,
};

use super::*;

#[derive(Debug)]
pub struct DistantLight {
    light_to_world: Transform,
    medium_interface: MediumInterface,

    // DistantLight Private Data
    l: Spectrum<SPECTRUM_N>,
    w_light: Vector3f,
    world_center: Point3f,
    world_radius: f64,
}

impl DistantLight {
    pub fn new(
        light_to_world: Transform,
        medium_interface: MediumInterface,
        l: Spectrum<SPECTRUM_N>,
        w_light: Vector3f,
    ) -> Self {
        let w_light = light_to_world.t(&w_light).normalize();
        Self {
            light_to_world,
            medium_interface,
            l,
            w_light,
            world_center: Point3f::default(),
            world_radius: 0.0,
        }
    }
}

impl ToWorld for DistantLight {
    fn to_world(&self) -> &Transform {
        &self.light_to_world
    }
}

impl Light for DistantLight {
    #[inline]
    fn flags(&self) -> LightFlags {
        LIGHT_DELTADIRECTION
    }
    #[inline]
    fn n_samples(&self) -> usize {
        1
    }

    fn medium_interface(&self) -> &MediumInterface {
        &self.medium_interface
    }

    fn sample_li(
        &self,
        ref_ist: &BaseInteraction,
        _u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f64,
        vis: &mut VisibilityTester,
    ) -> Spectrum<SPECTRUM_N> {
        *wi = self.w_light;
        *pdf = 1.0;
        let p_outside = ref_ist.p + self.w_light * (2.0 * self.world_radius);

        *vis = VisibilityTester::new(
            ref_ist.clone(),
            BaseInteraction::new(
                p_outside,
                ref_ist.time,
                Vector3f::default(),
                Vector3f::default(),
                Normal3f::default(),
                Some(self.medium_interface().clone()),
            ),
        );
        self.l
    }

    fn preprocess(&mut self, scene: &Scene) {
        Bounds3f::bounding_sphere(
            &scene.world_bound(),
            &mut self.world_center,
            &mut self.world_radius,
        );
    }
    fn power(&self) -> Spectrum<SPECTRUM_N> {
        self.l * PI * self.world_radius * self.world_radius
    }

    fn pdf_li(&self, _ref_ist: &BaseInteraction, _wi: &Vector3f) -> f64 {
        0.0
    }

    fn sample_le(
        &self,
        u1: &Point2f,
        _u2: &Point2f,
        time: f64,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f64,
        pdf_dir: &mut f64,
    ) -> Spectrum<SPECTRUM_N> {
        // Choose point on disk oriented toward infinite light direction
        let mut v1 = Vector3f::default();
        let mut v2 = Vector3f::default();
        vec3_coordinate_system(&self.w_light, &mut v1, &mut v2);
        let cd = concentric_sample_disk(*u1);
        let p_disk = self.world_center + (v1 * cd.x + v2 * cd.y) * self.world_radius;

        // Set ray origin and direction for infinite light ray
        *ray = Ray::new(
            p_disk + self.w_light * self.world_radius,
            -self.w_light,
            INFINITY,
            time,
            None,
        );
        *n_light = ray.d.into();
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
        *pdf_dir = 1.0;
        self.l
    }

    fn pdf_le(&self, _ray: &Ray, _n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
        *pdf_dir = 0.0;
    }
}
