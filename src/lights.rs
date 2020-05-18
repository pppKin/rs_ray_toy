use crate::geometry::{Point3f, Vector3f};
use crate::interaction::BaseInteraction;
use crate::spectrum::*;
use crate::transform::Transform;
use crate::{color::Color, rtoycore::SPECTRUM_N};
#[derive(Debug, Default, Clone)]
pub struct Light {
    pub position: Point3f,
    pub color: Color,
    pub kind: String,
}

pub enum LightFlags {
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
}

pub struct LightData {
    pub flag: LightFlags,
    pub n_samples: u64,
    // pub medium_interface: MediumInterface
    light_to_world: Transform,
    world_to_light: Transform,
}

pub trait AreaLight {
    fn L(&self, ist: &BaseInteraction, w: &Vector3f) -> Spectrum<SPECTRUM_N>;
}
