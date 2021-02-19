use std::f64::{consts::PI, INFINITY};

use crate::{
    sampling::{uniform_hemisphere_pdf, uniform_sample_sphere},
    transform::Transform,
};

use super::*;

#[derive(Debug)]
pub struct PointLight {
    light_to_world: Transform,
    medium_interface: MediumInterface,

    p_light: Point3f,
    i: Spectrum<SPECTRUM_N>,
}

impl PointLight {
    pub fn new(
        light_to_world: Transform,
        medium_interface: MediumInterface,
        p_light: Point3f,
        i: Spectrum<SPECTRUM_N>,
    ) -> Self {
        Self {
            light_to_world,
            medium_interface,
            p_light,
            i,
        }
    }
}

impl ToWorld for PointLight {
    fn to_world(&self) -> &Transform {
        &self.light_to_world
    }
}

impl Light for PointLight {
    #[inline]
    fn flags(&self) -> LightFlags {
        LIGHT_DELTAPOSITION
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
        *wi = (self.p_light - ref_ist.p).normalize();
        *pdf = 1.0;
        *vis = VisibilityTester::new(
            ref_ist.clone(),
            BaseInteraction::new(
                self.p_light,
                ref_ist.time,
                Vector3f::default(),
                Vector3f::default(),
                Normal3f::default(),
                Some(self.medium_interface().clone()),
            ),
        );
        self.i / (self.p_light - ref_ist.p).length_squared()
    }

    fn power(&self) -> Spectrum<SPECTRUM_N> {
        self.i * 4.0 * PI
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
        *ray = Ray::new(
            self.p_light,
            uniform_sample_sphere(*u1),
            INFINITY,
            time,
            self.medium_interface().inside.clone(),
        );
        *n_light = ray.d.into();
        *pdf_pos = 1.0;
        *pdf_dir = uniform_hemisphere_pdf();
        self.i
    }

    fn pdf_le(&self, _ray: &Ray, _n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        *pdf_pos = 0.0;
        *pdf_dir = uniform_hemisphere_pdf();
    }
}
