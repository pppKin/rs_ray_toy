use std::sync::Arc;

use crate::{
    geometry::Vector3f, interaction::SurfaceInteraction, reflection::*, spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{Material, TransportMode};

#[derive(Debug)]
pub struct DebugDiffuseBxdf {}

impl BxDF for DebugDiffuseBxdf {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        return Spectrum::new([0.0, 1.0, 0.0]);
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

#[derive(Debug)]
pub struct DebugSpecularBxdf {}

impl BxDF for DebugSpecularBxdf {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        return Spectrum::new([0.0, 0.0, 1.0]);
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_SPECULAR | BXDF_REFLECTION
    }
}

#[derive(Debug)]
pub struct DebugMaterial {}

impl Material for DebugMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        let mut bsdf = Bsdf::new(si, 1.0);
        bsdf.add(Arc::new(DebugDiffuseBxdf {}));
        bsdf.add(Arc::new(DebugSpecularBxdf {}));
        si.bsdf = Some(bsdf);
    }
}
