use std::{f64::INFINITY, rc::Rc, sync::Arc};

use crate::{
    microfacet::{roughness_to_alpha, TrowbridgeReitzDistribution},
    reflection::{
        Bsdf, FresnelDielectric, FresnelSpecular, MicrofacetReflection, MicrofacetTransmission,
        SpecularReflection, SpecularTransmission,
    },
    spectrum::Spectrum,
    texture::Texture,
    SPECTRUM_N,
};

use super::Material;

#[derive(Debug)]
pub struct GlassMaterial {
    kr: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    kt: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    u_roughness: Arc<dyn Texture<f64>>,
    v_roughness: Arc<dyn Texture<f64>>,

    index: Arc<dyn Texture<f64>>,
    bump_map: Option<Arc<dyn Texture<f64>>>,

    remap_roughness: bool,
}

impl GlassMaterial {
    pub fn new(
        kr: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        kt: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        u_roughness: Arc<dyn Texture<f64>>,
        v_roughness: Arc<dyn Texture<f64>>,
        index: Arc<dyn Texture<f64>>,
        bump_map: Option<Arc<dyn Texture<f64>>>,
        remap_roughness: bool,
    ) -> Self {
        Self {
            kr,
            kt,
            u_roughness,
            v_roughness,
            index,
            bump_map,
            remap_roughness,
        }
    }
}

impl Material for GlassMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut crate::interaction::SurfaceInteraction,
        mode: super::TransportMode,
        allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(ref bm) = self.bump_map {
            self.bump(bm.clone(), si);
        }
        let eta = self.index.evaluate(si);
        let mut urough = self.u_roughness.evaluate(si);
        let mut vrough = self.v_roughness.evaluate(si);
        let r = self.kr.evaluate(si).clamp(0.0, INFINITY);
        let t = self.kt.evaluate(si).clamp(0.0, INFINITY);

        // Initialize _bsdf_ for smooth or rough dielectric
        let mut bsdf = Bsdf::new(si, eta);
        if r.is_black() && t.is_black() {
            si.bsdf = None;
            return;
        }
        let is_specular = urough == 0.0 && vrough == 0.0;

        if is_specular && allow_multiple_lobes {
            bsdf.add(Rc::new(FresnelSpecular::new(r, t, 1.0, eta, mode)));
        } else {
            if self.remap_roughness {
                urough = roughness_to_alpha(urough);
                vrough = roughness_to_alpha(vrough);
            }
            if !r.is_black() {
                let fresnel = FresnelDielectric::new(1.0, eta);
                if is_specular {
                    bsdf.add(Rc::new(SpecularReflection::new(r, Box::new(fresnel))));
                } else {
                    let distrib = TrowbridgeReitzDistribution::new(urough, vrough, true);
                    bsdf.add(Rc::new(MicrofacetReflection::new(
                        r,
                        Box::new(distrib),
                        Box::new(fresnel),
                    )));
                }
            }
            if !t.is_black() {
                if is_specular {
                    bsdf.add(Rc::new(SpecularTransmission::new(t, 1.0, eta, mode)));
                } else {
                    let distrib = TrowbridgeReitzDistribution::new(urough, vrough, true);
                    bsdf.add(Rc::new(MicrofacetTransmission::new(
                        t,
                        Box::new(distrib),
                        1.0,
                        eta,
                        mode,
                    )));
                }
            }
        }
        si.bsdf = Some(bsdf);
    }
}
