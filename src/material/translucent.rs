use std::{f64::INFINITY, sync::Arc};

use crate::{
    microfacet::{roughness_to_alpha, TrowbridgeReitzDistribution},
    reflection::{
        Bsdf, FresnelDielectric, LambertianReflection, LambertianTransmission,
        MicrofacetReflection, MicrofacetTransmission,
    },
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    texture::Texture,
};

use super::Material;

#[derive(Debug)]
pub struct TranslucentMaterial {
    kd: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    ks: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    roughness: Arc<dyn Texture<f64>>,
    reflect: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    transmit: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,

    bump_map: Option<Arc<dyn Texture<f64>>>,
    remap_roughness: bool,
}

impl Material for TranslucentMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut crate::interaction::SurfaceInteraction,
        mode: super::TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(ref bm) = self.bump_map {
            self.bump(bm.clone(), si);
        }
        let eta = 1.5;
        let mut bsdf = Bsdf::new(si, eta);
        let r = self.reflect.evaluate(si).clamp(0.0, INFINITY);
        let t = self.transmit.evaluate(si).clamp(0.0, INFINITY);
        if r.is_black() && t.is_black() {
            si.bsdf = None;
            return;
        }

        let kd = self.kd.evaluate(si).clamp(0.0, INFINITY);
        if !kd.is_black() {
            if !r.is_black() {
                bsdf.add(Box::new(LambertianReflection::new(r * kd)));
            }
            if !t.is_black() {
                bsdf.add(Box::new(LambertianTransmission::new(t * kd)));
            }
        }

        let ks = self.ks.evaluate(si).clamp(0.0, INFINITY);
        if !ks.is_black() && (!r.is_black() || !t.is_black()) {
            let mut rough = self.roughness.evaluate(si);
            if self.remap_roughness {
                rough = roughness_to_alpha(rough);
            }
            if !r.is_black() {
                let distrib = TrowbridgeReitzDistribution::new(rough, rough, true);
                let fresnel = FresnelDielectric::new(1.0, eta);
                bsdf.add(Box::new(MicrofacetReflection::new(
                    r * ks,
                    Box::new(distrib),
                    Box::new(fresnel),
                )));
            }
            if !t.is_black() {
                let distrib = TrowbridgeReitzDistribution::new(rough, rough, true);
                bsdf.add(Box::new(MicrofacetTransmission::new(
                    t * ks,
                    Box::new(distrib),
                    1.0,
                    eta,
                    mode,
                )));
            }
        }
        si.bsdf = Some(bsdf);
    }
}
