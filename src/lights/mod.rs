use std::{fmt::Debug, usize};

use crate::{
    geometry::{IntersectP, Normal3f, Point2f, Point3f, Ray, RayDifferential, Vector3f},
    interaction::{BaseInteraction, SurfaceInteraction},
    medium::MediumInterface,
    samplers::Sampler,
    scene::Scene,
    spectrum::*,
    transform::ToWorld,
    SPECTRUM_N,
};

pub const LIGHT_DELTAPOSITION: u8 = 1 << 0;
pub const LIGHT_DELTADIRECTION: u8 = 1 << 1;
pub const LIGHT_AREA: u8 = 1 << 2;
pub const LIGHT_INFINITE: u8 = 1 << 3;
pub type LightFlag = u8;

pub fn is_delta_light(flag: LightFlag) -> bool {
    (flag & LIGHT_DELTAPOSITION) > 0 || (flag & LIGHT_DELTADIRECTION) > 0
}

pub trait Light: Debug + ToWorld + Send + Sync {
    fn flags(&self) -> LightFlag;
    fn n_samples(&self) -> usize;
    fn medium_interface(&self) -> &MediumInterface;
    fn sample_li(
        &self,
        ref_ist: &BaseInteraction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f64,
        vis: &mut VisibilityTester,
    ) -> Spectrum<SPECTRUM_N>;
    fn power(&self) -> Spectrum<SPECTRUM_N>;
    fn le(&self, _r: &RayDifferential) -> Spectrum<SPECTRUM_N> {
        Spectrum::zero()
    }
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

#[derive(Debug, Default)]
pub struct VisibilityTester {
    p0: BaseInteraction,
    p1: BaseInteraction,
}

impl VisibilityTester {
    pub fn new(p0: BaseInteraction, p1: BaseInteraction) -> Self {
        Self { p0, p1 }
    }
    pub fn unoccluded(&self, scene: &Scene) -> bool {
        !scene.intersect_p(&self.p0.spawn_ray_to_si(&self.p1))
    }
    pub fn tr(&self, scene: &Scene, sampler: &mut Sampler) -> Spectrum<SPECTRUM_N> {
        let mut ray = self.p0.spawn_ray_to_si(&self.p1);
        let mut tr = Spectrum::one();

        loop {
            let mut si = SurfaceInteraction::default();
            let hit_surface = scene.intersect(&mut ray, &mut si);

            // Handle opaque surface along ray's path
            if hit_surface && si.primitive.is_some() {
                return Spectrum::zero();
            }

            // Update transmittance for current ray segment
            match &ray.medium {
                Some(md) => {
                    // tr.md
                    let tmp = md.clone();
                    tr *= tmp.tr(&ray, sampler);
                }
                None => {}
            }

            // Generate next ray segment or return final transmittance
            if !hit_surface {
                break;
            }
            ray = si.ist.spawn_ray_to_si(&self.p1);
        }
        tr
    }
    pub fn get_p0(&self) -> &BaseInteraction {
        &self.p0
    }
    pub fn get_p1(&self) -> &BaseInteraction {
        &self.p1
    }
}

pub trait AreaLight: Light {
    fn l(&self, ist: &BaseInteraction, w: &Vector3f) -> Spectrum<SPECTRUM_N>;
}

pub mod diffuse;
pub mod distant;
pub mod infinite;
pub mod point;
