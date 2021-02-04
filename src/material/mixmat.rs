use std::{f64::INFINITY, rc::Rc, sync::Arc};

use crate::{
    interaction::SurfaceInteraction,
    reflection::{Bsdf, ScaledBxdf},
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    texture::Texture,
};

use super::{Material, TransportMode};

#[derive(Debug)]
pub struct MixMaterial {
    m1: Arc<dyn Material>,
    m2: Arc<dyn Material>,

    scale: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
}

impl MixMaterial {
    pub fn new(
        m1: Arc<dyn Material>,
        m2: Arc<dyn Material>,
        scale: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    ) -> Self {
        Self { m1, m2, scale }
    }
}

impl Material for MixMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        // Compute weights and original _BxDF_s for mix material
        let s1 = self.scale.evaluate(si).clamp(0.0, INFINITY);
        let s2 = (Spectrum::from(1.0) - s1).clamp(0.0, INFINITY);
        let mut si2 = si.clone();
        self.m1
            .compute_scattering_functions(si, mode, allow_multiple_lobes);
        self.m2
            .compute_scattering_functions(&mut si2, mode, allow_multiple_lobes);

        let mut result_bsdfs = Bsdf::new(si, 1.0);
        if let Some(bsdf) = si.bsdf.take() {
            for b in &bsdf.bxdfs {
                let tmp_b = Rc::clone(b);
                result_bsdfs.add(Rc::new(ScaledBxdf::new(tmp_b, s1)));
            }
        }
        if let Some(bsdf) = si2.bsdf.take() {
            for b in &bsdf.bxdfs {
                let tmp_b = Rc::clone(b);
                result_bsdfs.add(Rc::new(ScaledBxdf::new(tmp_b, s2)));
            }
        }
        si.bsdf = Some(result_bsdfs);
    }
}
