use std::sync::Arc;

use crate::{
    interaction::SurfaceInteraction,
    reflection::{Bsdf, BxDF},
    spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{Material, TransportMode};

#[derive(Debug)]
pub struct DebugBsdf {}

#[derive(Debug)]
pub struct DebugMaterial {}

impl BxDF for DebugBsdf {
    fn f(
        &self,
        _wo: &crate::geometry::Vector3f,
        _wi: &crate::geometry::Vector3f,
    ) -> Spectrum<SPECTRUM_N> {
        return Spectrum::from(0.5);
    }

    fn bxdf_type(&self) -> crate::reflection::BxDFType {
        return crate::reflection::BXDF_ALL;
    }
}

impl Material for DebugMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        let mut bsdf = Bsdf::new(si, 1.0);
        bsdf.add(Arc::new(DebugBsdf {}));
        si.bsdf = Some(bsdf);
    }
}
