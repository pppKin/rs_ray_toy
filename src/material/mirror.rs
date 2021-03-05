use std::{f64::INFINITY, sync::Arc};

use crate::{
    reflection::{Bsdf, FresnelNoOp, SpecularReflection},
    spectrum::Spectrum,
    texture::Texture,
    SPECTRUM_N,
};

use super::Material;

#[derive(Debug)]
pub struct MirrorMaterial {
    kr: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    bump_map: Option<Arc<dyn Texture<f64>>>,
}

impl MirrorMaterial {
    pub fn new(
        kr: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        bump_map: Option<Arc<dyn Texture<f64>>>,
    ) -> Self {
        Self { kr, bump_map }
    }
}

impl Material for MirrorMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut crate::interaction::SurfaceInteraction,
        _mode: super::TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(ref bm) = self.bump_map {
            self.bump(bm.clone(), si);
        }
        let r = self.kr.evaluate(si).clamp(0.0, INFINITY);
        let mut bsdf = Bsdf::new(si, 1.0);
        if !r.is_black() {
            bsdf.add(Arc::new(SpecularReflection::new(
                r,
                Arc::new(FresnelNoOp::default()),
            )));
        }
        si.bsdf = Some(bsdf);
    }
}
