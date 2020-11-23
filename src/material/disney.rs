use std::{f64::consts::PI, f64::INFINITY, sync::Arc};

use crate::{
    geometry::{dot3, spherical_direction, Point2f, Vector3f},
    microfacet::{MicrofacetDistribution, TrowbridgeReitzDistribution},
    misc::lerp,
    reflection::{
        abs_cos_theta, fr_dielectric, fr_schlick, fr_schlick_spectrum, reflect, same_hemisphere,
        schlick_r0_from_eta, schlick_weight, Bsdf, BxDF, BxDFType, Fresnel, FresnelDielectric,
        LambertianTransmission, MicrofacetReflection, MicrofacetTransmission, BXDF_DIFFUSE,
        BXDF_GLOSSY, BXDF_REFLECTION,
    },
    rtoycore::SPECTRUM_N,
    spectrum::{ISpectrum, Spectrum},
    texture::Texture,
};

use super::Material;

fn gtr1(cos_theta: f64, alpha: f64) -> f64 {
    let alpha2 = alpha * alpha;

    (alpha2 - 1.0) / (PI * f64::log10(alpha2) * (1.0 + (alpha2 - 1.0) * cos_theta * cos_theta))
}

fn smith_g_ggx(cos_theta: f64, alpha: f64) -> f64 {
    let alpha2 = alpha * alpha;
    let cos_theta2 = cos_theta * cos_theta;

    1.0 / (cos_theta + f64::sqrt(alpha2 + cos_theta2 - alpha2 * cos_theta2))
}

// DisneyDiffuse
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyDiffuse {
    r: Spectrum<SPECTRUM_N>,
}

impl DisneyDiffuse {
    pub fn new(r: Spectrum<SPECTRUM_N>) -> Self {
        Self { r }
    }
}

impl BxDF for DisneyDiffuse {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        //     f64 Fo = SchlickWeight(AbsCosTheta(wo)),
        //           Fi = SchlickWeight(AbsCosTheta(wi));
        let fo = schlick_weight(abs_cos_theta(wo));
        let fi = schlick_weight(abs_cos_theta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        self.r / PI * (1.0 - fo / 2.0) * (1.0 - fi / 2.0)
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

/// DisneyFakeSS
/// "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
/// approximation of the BSSRDF.
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyFakeSS {
    r: Spectrum<SPECTRUM_N>,
    roughness: f64,
}

impl DisneyFakeSS {
    pub fn new(r: Spectrum<SPECTRUM_N>, roughness: f64) -> Self {
        Self { r, roughness }
    }
}

impl BxDF for DisneyFakeSS {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let mut wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::zero();
        }
        wh = wh.normalize();
        let cos_theta_d = dot3(wi, &wh);

        // Fss90 used to "flatten" retroreflection based on roughness
        let fss_90 = cos_theta_d * cos_theta_d * self.roughness;
        let fo = schlick_weight(abs_cos_theta(wo));
        let fi = schlick_weight(abs_cos_theta(wi));

        let fss = lerp(fo, 1.0, fss_90) * lerp(fi, 1.0, fss_90);
        // 1.25 scale is used to (roughly) preserve albedo
        let ss = 1.25 * (fss * (1.0 / (abs_cos_theta(wo) + abs_cos_theta(wi)) - 0.5) + 0.5);

        self.r / PI * ss
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

// DisneyRetro
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyRetro {
    r: Spectrum<SPECTRUM_N>,
    roughness: f64,
}

impl DisneyRetro {
    pub fn new(r: Spectrum<SPECTRUM_N>, roughness: f64) -> Self {
        Self { r, roughness }
    }
}

impl BxDF for DisneyRetro {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let mut wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::zero();
        }
        wh = wh.normalize();
        let cos_theta_d = dot3(wi, &wh);
        let fo = schlick_weight(abs_cos_theta(wo));
        let fi = schlick_weight(abs_cos_theta(wi));
        let r_r = 2.0 * self.roughness * cos_theta_d * cos_theta_d;
        // Burley 2015, eq (4).
        self.r / PI * r_r * (fo + fi + fo * fi * (r_r - 1.0))
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

// DisneySheen
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneySheen {
    r: Spectrum<SPECTRUM_N>,
}

impl DisneySheen {
    pub fn new(r: Spectrum<SPECTRUM_N>) -> Self {
        Self { r }
    }
}

impl BxDF for DisneySheen {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let mut wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::zero();
        }
        wh = wh.normalize();
        let cos_theta_d = dot3(wi, &wh);
        self.r * schlick_weight(cos_theta_d)
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.r
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

// DisneyClearcoat
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyClearcoat {
    weight: f64,
    gloss: f64,
}

impl DisneyClearcoat {
    pub fn new(weight: f64, gloss: f64) -> Self {
        Self { weight, gloss }
    }
}

impl BxDF for DisneyClearcoat {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let mut wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::zero();
        }
        wh = wh.normalize();
        // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
        // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
        // (which is GTR2).
        let dr = gtr1(abs_cos_theta(&wh), self.gloss);
        let fr = fr_schlick(0.04, dot3(wo, &wh));
        // The geometric term always based on alpha = 0.25.

        let gr = smith_g_ggx(abs_cos_theta(wo), 0.25) * smith_g_ggx(abs_cos_theta(wi), 0.25);
        Spectrum::from(self.weight * gr * fr * dr / 4.0)
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        // TODO: double check all this: there still seem to be some very
        // occasional fireflies with clearcoat; presumably there is a bug
        // somewhere.
        if wo.z == 0.0 {
            return Spectrum::zero();
        }
        let alpha2 = self.gloss * self.gloss;
        let cos_theta = (1.0 - alpha2.powf(1.0 - u[0])) / (1.0 - alpha2).max(0.0).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * u[1];
        let mut wh = spherical_direction(sin_theta, cos_theta, phi);
        if !same_hemisphere(wo, &wh) {
            wh = -wh;
        }
        *wi = reflect(wo, &wh);
        if !same_hemisphere(wo, wi) {
            return Spectrum::zero();
        }

        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        let mut wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return 0.0;
        }
        wh = wh.normalize();
        // The sampling routine samples wh exactly from the GTR1 distribution.
        // Thus, the final value of the PDF is just the value of the
        // distribution for wh converted to a mesure with respect to the
        // surface normal.
        let dr = gtr1(abs_cos_theta(&wh), self.gloss);
        dr * abs_cos_theta(&wh) / (4.0 * dot3(wo, &wh))
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_GLOSSY
    }
}

// DisneyFresnel
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyFresnel {
    r0: Spectrum<SPECTRUM_N>,
    metallic: f64,
    eta: f64,
}

impl DisneyFresnel {
    pub fn new(r0: Spectrum<SPECTRUM_N>, metallic: f64, eta: f64) -> Self {
        Self { r0, metallic, eta }
    }
}

impl Fresnel for DisneyFresnel {
    fn evaluate(&self, cos_i: f64) -> Spectrum<SPECTRUM_N> {
        Spectrum::lerp(
            self.metallic,
            Spectrum::from(fr_dielectric(cos_i, 1.0, self.eta)),
            fr_schlick_spectrum(self.r0, cos_i),
        )
    }
}

// DisneyMicrofacetDistribution
#[derive(Debug, Default, Copy, Clone)]
pub struct DisneyMicrofacetDistribution {
    trd: TrowbridgeReitzDistribution,
}

impl DisneyMicrofacetDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64, sample_visible_area: bool) -> Self {
        let trd = TrowbridgeReitzDistribution::new(alpha_x, alpha_y, sample_visible_area);
        Self { trd }
    }
}

impl MicrofacetDistribution for DisneyMicrofacetDistribution {
    fn d(&self, wh: &Vector3f) -> f64 {
        self.trd.d(wh)
    }
    fn lambda(&self, w: &Vector3f) -> f64 {
        self.trd.lambda(w)
    }
    fn sample_wh(&self, wo: &Vector3f, u: Point2f) -> Vector3f {
        self.trd.sample_wh(wo, u)
    }
    fn get_sample_visible_area(&self) -> bool {
        self.trd.get_sample_visible_area()
    }

    fn g(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        // Disney uses the separable masking-shadowing model.
        self.g1(wo) * self.g1(wi)
    }
}

// TODO: DisneyBSSRDF
#[derive(Debug, Clone)]
pub struct DisneyMaterial {
    // DisneyMaterial Private Data
    color: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,

    metallic: Arc<dyn Texture<f64>>,
    eta: Arc<dyn Texture<f64>>,

    roughness: Arc<dyn Texture<f64>>,
    specular_tint: Arc<dyn Texture<f64>>,
    anisotropic: Arc<dyn Texture<f64>>,
    sheen: Arc<dyn Texture<f64>>,

    sheen_tint: Arc<dyn Texture<f64>>,
    clearcoat: Arc<dyn Texture<f64>>,
    clearcoat_gloss: Arc<dyn Texture<f64>>,
    spec_trans: Arc<dyn Texture<f64>>,

    scatter_distance: Arc<dyn Texture<Spectrum<SPECTRUM_N>>>,
    thin: bool,

    flatness: Arc<dyn Texture<f64>>,
    diff_trans: Arc<dyn Texture<f64>>,
    bump_map: Option<Arc<dyn Texture<f64>>>,
}

impl Material for DisneyMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut crate::interaction::SurfaceInteraction,
        mode: super::TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        // Perform bump mapping with _bumpMap_, if present
        if let Some(ref bump_map) = self.bump_map {
            self.bump(bump_map.clone(), si);
        }

        // Evaluate textures for _DisneyMaterial_ material and allocate BRDF
        let mut bsdf = Bsdf::new(si, 1.0);
        // Diffuse
        let c = self.color.evaluate(si).clamp(0.0, INFINITY);
        let metallic_weight = self.metallic.evaluate(si);
        let e = self.eta.evaluate(si);
        let strans = self.spec_trans.evaluate(si);
        let diffuse_weight = (1.0 - metallic_weight) * (1.0 - strans);
        let dt = self.diff_trans.evaluate(si);
        let rough = self.roughness.evaluate(si);
        let lum = c.y();
        // normalize lum. to isolate hue+sat
        let c_tint = if lum > 0.0 {
            Spectrum::from(c / lum)
        } else {
            Spectrum::one()
        };

        let sheen_weight = self.sheen.evaluate(si);
        let mut c_sheen = Spectrum::default();
        if sheen_weight > 0.0 {
            let s_tint = self.sheen_tint.evaluate(si);
            c_sheen = Spectrum::lerp(s_tint, Spectrum::one(), c_tint);
        }

        if diffuse_weight > 0.0 {
            if self.thin {
                let flat = self.flatness.evaluate(si);
                // Blend between DisneyDiffuse and fake subsurface based on
                // flatness.  Additionally, weight using diffTrans.
                bsdf.add(Box::new(DisneyDiffuse::new(
                    c * diffuse_weight * (1.0 - flat) * (1.0 - dt),
                )));
                bsdf.add(Box::new(DisneyFakeSS::new(
                    c * diffuse_weight * flat * (1.0 - dt),
                    rough,
                )));
            } else {
                let sd = self.scatter_distance.evaluate(si);
                if sd.is_black() {
                    // No subsurface scattering; use regular (Fresnel modified)
                    // diffuse.
                    bsdf.add(Box::new(DisneyDiffuse::new(c * diffuse_weight)));
                } else {
                    // SpecularTransmission::new(1.0, 1.0, e, mode)
                    // DisneyBSSRDF::new(c* diffuse_weight, sd, si, e, mode)
                    todo!()
                }
            }
            // Retro-reflection.
            bsdf.add(Box::new(DisneyRetro::new(c * diffuse_weight, rough)));

            // Sheen (if enabled)
            if sheen_weight > 0.0 {
                bsdf.add(Box::new(DisneySheen::new(
                    c_sheen * sheen_weight * diffuse_weight,
                )));
            }
        }

        // Create the microfacet distribution for metallic and/or specular
        // transmission.
        let aspect = (1.0 - self.anisotropic.evaluate(si) * 0.9).sqrt();
        let ax = (rough.powi(2) / aspect).max(0.001);
        let ay = (rough.powi(2) * aspect).max(0.001);
        let distrib = DisneyMicrofacetDistribution::new(ax, ay, true);

        // Specular is Trowbridge-Reitz with a modified Fresnel function.
        let spec_tint = self.specular_tint.evaluate(si);
        let c_spec_0 = Spectrum::lerp(
            metallic_weight,
            Spectrum::lerp(spec_tint, Spectrum::one(), c_tint) * schlick_r0_from_eta(e),
            c,
        );

        let fresnel = DisneyFresnel::new(c_spec_0, metallic_weight, e);
        bsdf.add(Box::new(MicrofacetReflection::new(
            Spectrum::one(),
            Box::new(distrib),
            Box::new(fresnel),
        )));
        // Clearcoat
        let cc = self.clearcoat.evaluate(si);
        if cc > 0.0 {
            bsdf.add(Box::new(DisneyClearcoat::new(
                cc,
                lerp(self.clearcoat_gloss.evaluate(si), 0.1, 0.001),
            )))
        }

        // BTDF
        if strans > 0.0 {
            // Walter et al's model, with the provided transmissive term scaled
            // by sqrt(color), so that after two refractions, we're back to the
            // provided color.
            let t: Spectrum<SPECTRUM_N> = c.sqrt() * strans;
            if self.thin {
                // Scale roughness based on IOR (Burley 2015, Figure 15).
                let r_scaled = (0.65 * e - 0.35) * rough;
                let ax = (r_scaled.powi(2) / aspect).max(0.001);
                let ay = (r_scaled.powi(2) * aspect).max(0.001);
                let scaled_distrib = TrowbridgeReitzDistribution::new(ax, ay, true);
                bsdf.add(Box::new(MicrofacetTransmission::new(
                    t,
                    Box::new(scaled_distrib),
                    1.0,
                    e,
                    FresnelDielectric::new(1.0, e),
                    mode,
                )));
            } else {
                bsdf.add(Box::new(MicrofacetTransmission::new(
                    t,
                    Box::new(distrib),
                    1.0,
                    e,
                    FresnelDielectric::new(1.0, e),
                    mode,
                )));
            }
        }

        if self.thin {
            bsdf.add(Box::new(LambertianTransmission::new(c * dt)));
        }
    }
}
