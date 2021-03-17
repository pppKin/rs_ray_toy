use std::sync::Arc;

use crate::{
    geometry::{cross, Normal3f, Vector2f, Vector3f},
    interaction::SurfaceInteraction,
    texture::Texture,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

pub trait Material: std::fmt::Debug + Send + Sync {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
    fn bump(&self, d: Arc<dyn Texture<f64>>, si: &mut SurfaceInteraction) {
        // p′(u,v)=p(u,v)+d(u,v)n(u,v),
        let mut si_eval = si.clone();
        // Shift _siEval_ _du_ in the $u$ direction
        let mut du = si.dudx.abs() * 0.5 + si.dudy.abs();
        // The most common reason for du to be zero is for ray that start from
        // light sources, where no differentials are available. In this case,
        // we try to choose a small enough du so that we still get a decently
        // accurate bump value.
        if du == 0.0 {
            du = 0.0005;
        }
        si_eval.ist.p = si.ist.p + si.shading.dpdu * du;
        si_eval.uv = si.uv + Vector2f::new(du, 0.0);
        si_eval.ist.n =
            (Normal3f::from(cross(&si.shading.dpdu, &si.shading.dpdv)) + si.dndu * du).normalize();
        let u_displace = d.evaluate(&si_eval);
        // Shift _siEval_ _dv_ in the $v$ direction
        let mut dv = (si.dvdx.abs() + si.dvdy.abs()) * 0.5;
        if dv == 0.0 {
            dv = 0.0005;
        }

        si_eval.ist.p = si.ist.p + si.shading.dpdv * dv;
        si_eval.uv = si.uv + Vector2f::new(0.0, dv);
        si_eval.ist.n =
            (Normal3f::from(cross(&si.shading.dpdu, &si.shading.dpdv)) + si.dndv * dv).normalize();
        let v_displace = d.evaluate(&si_eval);
        let displace = d.evaluate(si);
        // Compute bump-mapped differential geometry
        let dpdu = si.shading.dpdu
            + Vector3f::from(si.shading.n) * (u_displace - displace) / du
            + Vector3f::from(si.shading.dndu) * displace;
        let dpdv = si.shading.dpdv
            + Vector3f::from(si.shading.n) * (v_displace - displace) / dv
            + Vector3f::from(si.shading.dndv) * displace;

        let dndus = si.shading.dndu;
        let dndvs = si.shading.dndv;
        si.set_shading_geometry(&dpdu, &dpdv, &dndus, &dndvs, false);
    }
}

pub mod disney;
pub mod glass;
pub mod matte;
pub mod metal;
pub mod mirror;
pub mod mixmat;
pub mod plastic;
pub mod translucent;
