use std::{f64::INFINITY, rc::Rc, sync::Arc};

use crate::{
    interaction::SurfaceInteraction,
    misc::clamp_t,
    reflection::{Bsdf, LambertianReflection, OrenNayar},
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    texture::Texture,
};

use super::{Material, TransportMode};

#[derive(Debug)]
pub struct MatteMaterial {
    kd: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    sigma: Arc<dyn Texture<f64>>,
    bump_map: Option<Arc<dyn Texture<f64>>>,
}

impl MatteMaterial {
    pub fn new(
        kd: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        sigma: Arc<dyn Texture<f64>>,
        bump_map: Option<Arc<dyn Texture<f64>>>,
    ) -> Self {
        Self {
            kd,
            sigma,
            bump_map,
        }
    }
}

impl Material for MatteMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(bump_map) = &self.bump_map {
            self.bump(Arc::clone(bump_map), si)
        }
        // Evaluate textures for _MatteMaterial_ material and allocate BRDF
        let r = self.kd.evaluate(si).clamp(0.0, INFINITY);

        let sig = clamp_t(self.sigma.evaluate(si), 0.0, 90.0);

        let mut bsdf = Bsdf::new(si, 1.0);
        if !r.is_black() {
            if sig == 0.0 {
                bsdf.add(Rc::new(LambertianReflection::new(r)));
            } else {
                bsdf.add(Rc::new(OrenNayar::new(r, sig)));
            }
        }
        si.bsdf = Some(bsdf);
    }
}
