use std::fmt::Debug;

use crate::spectrum::*;
use crate::transform::Transform;
use crate::{color::Color, rtoycore::SPECTRUM_N};
use crate::{
    geometry::{Normal3f, Ray},
    interaction::BaseInteraction,
};
use crate::{
    geometry::{Point2f, Point3f, RayDifferential, Vector3f},
    medium::MediumInterface,
    scene::Scene,
};
#[derive(Debug, Default, Clone)]
pub struct DeprecatedLight {
    pub position: Point3f,
    pub color: Color,
    pub kind: String,
}

pub const LIGHT_DELTAPOSITION: u8 = 1 << 0;
pub const LIGHT_DELTADIRECTION: u8 = 1 << 1;
pub const LIGHT_AREA: u8 = 1 << 2;
pub const LIGHT_INFINITE: u8 = 1 << 3;
pub type LightFlags = u8;
pub struct LightData {
    pub flag: LightFlags,
    pub n_samples: u64,
    pub medium_interface: MediumInterface,
    // light_to_world: Transform,
    // world_to_light: Transform,
}

pub trait Light: Debug {
    fn Sample_Li(
        &self,
        ref_ist: &BaseInteraction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f64,
        vis: &mut VisibilityTester,
    ) -> Spectrum<SPECTRUM_N>;
    fn power(&self) -> Spectrum<SPECTRUM_N>;
    fn preprocess(&self, scene: &Scene);
    fn le(&self, r: &RayDifferential) -> Spectrum<SPECTRUM_N>;
    fn pdf_li(&self, ref_ist: &BaseInteraction, wi: &Vector3f) -> f64;
    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        time: f64,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f64,
        pdf_dir: &mut f64,
    ) -> Spectrum<SPECTRUM_N>;
    fn pdf_le(&self, ray: &Ray, n_light: &Normal3f, pdf_pos: &mut f64, pdf_dir: &mut f64);
}

#[derive(Debug)]
pub struct VisibilityTester {
    p0: BaseInteraction,
    p1: BaseInteraction,
}

impl VisibilityTester {
    pub fn new(p0: BaseInteraction, p1: BaseInteraction) -> Self {
        Self { p0, p1 }
    }
    pub fn unoccluded(&self, scene: &Scene) -> bool {
        todo!();
    }
    pub fn tr(&self, scene: &Scene) -> Spectrum<SPECTRUM_N> {
        todo!();
    }
    pub fn get_p0(&self) -> &BaseInteraction {
        &self.p0
    }
    pub fn get_p1(&self) -> &BaseInteraction {
        &self.p1
    }
}

pub trait AreaLight: Debug {
    fn L(&self, ist: &BaseInteraction, w: &Vector3f) -> Spectrum<SPECTRUM_N>;
}
