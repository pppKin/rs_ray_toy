use std::sync::Arc;

use crate::{geometry::dot3, shape::Shape, transform::Transform};

use super::*;

#[derive(Debug)]
pub struct DiffuseAreaLight {
    light_to_world: Transform,
    medium_interface: MediumInterface,

    lemit: Spectrum<SPECTRUM_N>,
    shape: Arc<dyn Shape>,

    areea: f64,
}

impl ToWorld for DiffuseAreaLight {
    fn to_world(&self) -> &Transform {
        &self.light_to_world
    }
}

impl Light for DiffuseAreaLight {
    fn flags(&self) -> LightFlags {
        todo!()
    }

    fn n_samples(&self) -> usize {
        todo!()
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
        todo!()
    }

    fn power(&self) -> Spectrum<SPECTRUM_N> {
        todo!()
    }

    fn pdf_li(&self, ref_ist: &BaseInteraction, wi: &Vector3f) -> f64 {
        todo!()
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
        todo!()
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64) {
        todo!()
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
