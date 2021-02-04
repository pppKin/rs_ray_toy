use std::{f64::INFINITY, rc::Rc, sync::Arc};

use crate::{
    interaction::SurfaceInteraction,
    microfacet::{roughness_to_alpha, TrowbridgeReitzDistribution},
    reflection::{Bsdf, FresnelDielectric, LambertianReflection, MicrofacetReflection},
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    texture::Texture,
};

use super::{Material, TransportMode};

#[derive(Debug)]
pub struct PlasticMaterial {
    kd: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    ks: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    roughness: Arc<dyn Texture<f64>>,
    bump_map: Option<Arc<dyn Texture<f64>>>,
    remap_roughness: bool,
}

impl PlasticMaterial {
    pub fn new(
        kd: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        ks: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
        roughness: Arc<dyn Texture<f64>>,
        bump_map: Option<Arc<dyn Texture<f64>>>,
        remap_roughness: bool,
    ) -> Self {
        Self {
            kd,
            ks,
            roughness,
            bump_map,
            remap_roughness,
        }
    }
}

impl Material for PlasticMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(bm) = &self.bump_map {
            self.bump(Arc::clone(bm), si);
        }
        let mut bsdf = Bsdf::new(si, 1.0);
        // Initialize diffuse component of plastic material
        let kd = self.kd.evaluate(si).clamp(0.0, INFINITY);
        if !kd.is_black() {
            bsdf.add(Rc::new(LambertianReflection::new(kd)));
        }
        // Initialize specular component of plastic material
        let ks = self.ks.evaluate(si).clamp(0.0, INFINITY);
        if !kd.is_black() {
            let fresnel = FresnelDielectric::new(1.5, 1.0);
            // Create microfacet distribution _distrib_ for plastic material
            let mut rough = self.roughness.evaluate(si);
            if self.remap_roughness {
                rough = roughness_to_alpha(rough);
            }
            let distrib = TrowbridgeReitzDistribution::new(rough, rough, true);
            let spec = MicrofacetReflection::new(ks, Box::new(distrib), Box::new(fresnel));
            bsdf.add(Rc::new(spec));
        }
        si.bsdf = Some(bsdf);
    }
}
