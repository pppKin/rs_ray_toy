use std::{f64::consts::PI, sync::Arc};

use crate::{
    geometry::{dot3, vec3_coordinate_system},
    misc::{cosine_hemisphere_pdf, cosine_sample_hemisphere},
    shape::Shape,
    transform::Transform,
};

use super::*;

#[derive(Debug)]
pub struct DiffuseAreaLight {
    light_to_world: Transform,
    medium_interface: MediumInterface,
    n_samples: usize,

    lemit: Spectrum<SPECTRUM_N>,
    shape: Arc<dyn Shape>,

    area: f64,
}

impl DiffuseAreaLight {
    pub fn new(
        light_to_world: Transform,
        medium_interface: MediumInterface,
        n_samples: usize,
        lemit: Spectrum<SPECTRUM_N>,
        shape: Arc<dyn Shape>,
        area: f64,
    ) -> Self {
        Self {
            light_to_world,
            medium_interface,
            n_samples,
            lemit,
            shape,
            area,
        }
    }
}

impl ToWorld for DiffuseAreaLight {
    fn to_world(&self) -> &Transform {
        &self.light_to_world
    }
}

impl Light for DiffuseAreaLight {
    fn flags(&self) -> LightFlags {
        LIGHT_AREA
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn medium_interface(&self) -> &MediumInterface {
        &self.medium_interface
    }

    fn sample_li(
        &self,
        ref_ist: &BaseInteraction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f64,
        vis: &mut VisibilityTester,
    ) -> Spectrum<SPECTRUM_N> {
        let mut p_shape = self.shape.sample_ref(ref_ist, u, pdf);
        p_shape.mi = Some(self.medium_interface().clone());
        if *pdf == 0.0 || (p_shape.p - ref_ist.p).length_squared() == 0.0 {
            *pdf = 0.0;
            return Spectrum::zero();
        }
        *wi = (p_shape.p - ref_ist.p).normalize();
        *vis = VisibilityTester::new(ref_ist.clone(), p_shape.clone());
        self.l(&p_shape, &-*wi)
    }

    fn power(&self) -> Spectrum<SPECTRUM_N> {
        self.lemit * self.area * PI
    }

    fn pdf_li(&self, ref_ist: &BaseInteraction, wi: &Vector3f) -> f64 {
        self.shape.pdf_ref(ref_ist, wi)
    }

    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        _time: f64,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f64,
        pdf_dir: &mut f64,
    ) -> Spectrum<SPECTRUM_N> {
        // ProfilePhase _(Prof::LightSample);
        // Sample a point on the area light's _Shape_, _pShape_
        let mut p_shape = self.shape.sample(u1, pdf_pos);
        p_shape.mi = Some(self.medium_interface().clone());
        *n_light = p_shape.n;

        // Sample a cosine-weighted outgoing direction _w_ for area light
        let mut w = cosine_sample_hemisphere(*u2);
        *pdf_dir = cosine_hemisphere_pdf(w.z);

        let mut v1 = Vector3f::default();
        let mut v2 = Vector3f::default();
        let n = p_shape.n.into();
        vec3_coordinate_system(&n, &mut v1, &mut v2);
        w = v1 * w.x + v2 * w.y + n * w.z;
        *ray = p_shape.spawn_ray(w);
        self.l(&p_shape, &w)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        let ist = BaseInteraction::new(
            ray.o,
            ray.time,
            Vector3f::default(),
            (*n_light).into(),
            *n_light,
            Some(self.medium_interface().clone()),
        );
        *pdf_pos = self.shape.pdf(&ist);
        *pdf_dir = cosine_hemisphere_pdf(dot3(n_light, &ray.d));
    }
}

impl AreaLight for DiffuseAreaLight {
    fn l(&self, ist: &BaseInteraction, w: &Vector3f) -> Spectrum<SPECTRUM_N> {
        if dot3(&ist.n, w) > 0.0 {
            self.lemit
        } else {
            Spectrum::zero()
        }
    }
}
