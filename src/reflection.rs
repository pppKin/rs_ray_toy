use crate::{
    geometry::{abs_dot3, cross, dot3, faceforward, Cxyz, Normal3f, Point2f, Vector3f},
    interaction::SurfaceInteraction,
    material::TransportMode,
    microfacet::MicrofacetDistribution,
    misc::{clamp_t, lerp, radians, ONE_MINUS_EPSILON},
    sampling::{cosine_sample_hemisphere, uniform_hemisphere_pdf, uniform_sample_hemisphere},
    spectrum::Spectrum,
    SPECTRUM_N,
};
use std::{f64::consts::PI, rc::Rc};
#[inline]
pub fn schlick_weight(cos_theta: f64) -> f64 {
    let m = clamp_t(1.0 - cos_theta, 0.0, 1.0);
    (m * m) * (m * m) * m
}

pub fn fr_schlick(r0: f64, cos_theta: f64) -> f64 {
    lerp(schlick_weight(cos_theta), r0, 1.0)
}

pub fn fr_schlick_spectrum(r0: Spectrum<SPECTRUM_N>, cos_theta: f64) -> Spectrum<SPECTRUM_N> {
    lerp(schlick_weight(cos_theta), r0, Spectrum::one())
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
pub fn schlick_r0_from_eta(eta: f64) -> f64 {
    ((eta - 1.0) / (eta + 1.0)).powi(2)
}

/// Utility function to calculate cosine via spherical coordinates.
pub fn cos_theta(w: &Vector3f) -> f64 {
    w.z
}

/// Utility function to calculate the square cosine via spherical
/// coordinates.
pub fn cos_2_theta(w: &Vector3f) -> f64 {
    w.z * w.z
}

/// Utility function to calculate the absolute value of the cosine via
/// spherical coordinates.
pub fn abs_cos_theta(w: &Vector3f) -> f64 {
    w.z.abs()
}

/// Utility function to calculate the square sine via spherical
/// coordinates.
pub fn sin_2_theta(w: &Vector3f) -> f64 {
    (0.0 as f64).max(1.0 as f64 - cos_2_theta(w))
}

/// Utility function to calculate sine via spherical coordinates.
pub fn sin_theta(w: &Vector3f) -> f64 {
    sin_2_theta(w).sqrt()
}

/// Utility function to calculate the tangent via spherical
/// coordinates.
pub fn tan_theta(w: &Vector3f) -> f64 {
    sin_theta(w) / cos_theta(w)
}

/// Utility function to calculate the square tangent via spherical
/// coordinates.
pub fn tan_2_theta(w: &Vector3f) -> f64 {
    sin_2_theta(w) / cos_2_theta(w)
}

/// Utility function to calculate cosine via spherical coordinates.
pub fn cos_phi(w: &Vector3f) -> f64 {
    let sin_theta: f64 = sin_theta(w);
    if sin_theta == 0.0 as f64 {
        1.0 as f64
    } else {
        clamp_t(w.x / sin_theta, -1.0, 1.0)
    }
}

/// Utility function to calculate sine via spherical coordinates.
pub fn sin_phi(w: &Vector3f) -> f64 {
    let sin_theta: f64 = sin_theta(w);
    if sin_theta == 0.0 as f64 {
        0.0 as f64
    } else {
        clamp_t(w.y / sin_theta, -1.0, 1.0)
    }
}

/// Utility function to calculate square cosine via spherical coordinates.
pub fn cos_2_phi(w: &Vector3f) -> f64 {
    cos_phi(w) * cos_phi(w)
}

/// Utility function to calculate square sine via spherical coordinates.
pub fn sin_2_phi(w: &Vector3f) -> f64 {
    sin_phi(w) * sin_phi(w)
}

/// Utility function to calculate the cosine of the angle between two
/// vectors in the shading coordinate system.
pub fn cos_d_phi(wa: &Vector3f, wb: &Vector3f) -> f64 {
    clamp_t(
        (wa.x * wb.x + wa.y * wb.y)
            / ((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)).sqrt(),
        -1.0 as f64,
        1.0 as f64,
    )
}

/// Computes the reflection direction given an incident direction and
/// a surface normal.
pub fn reflect(wo: &Vector3f, n: &Vector3f) -> Vector3f {
    -(*wo) + *n * 2.0 as f64 * dot3(wo, n)
}

/// Computes the refraction direction given an incident direction, a
/// surface normal, and the ratio of indices of refraction (incident
/// and transmitted).
pub fn refract(wi: &Vector3f, n: &Normal3f, eta: f64, wt: &mut Vector3f) -> bool {
    // compute $\cos \theta_\roman{t}$ using Snell's law
    let cos_theta_i: f64 = dot3(n, wi);
    let sin2_theta_i: f64 = (0.0 as f64).max(1.0 as f64 - cos_theta_i * cos_theta_i);
    let sin2_theta_t: f64 = eta * eta * sin2_theta_i;
    // handle total internal reflection for transmission
    if sin2_theta_t >= 1.0 as f64 {
        return false;
    }
    let cos_theta_t: f64 = (1.0 as f64 - sin2_theta_t).sqrt();
    *wt = -(*wi) * eta + Vector3f::from(*n) * (eta * cos_theta_i - cos_theta_t);
    true
}

/// Check that two vectors lie on the same side of of the surface.
pub fn same_hemisphere(w: &impl Cxyz<f64>, wp: &impl Cxyz<f64>) -> bool {
    let (_, _, wz) = w.to_xyz();
    let (_, _, wpz) = wp.to_xyz();
    wz * wpz > 0.0
}

/// The Fresnel equations describe the amount of light reflected from a surface;
/// they are the solution to Maxwell’s equations at smooth surfaces.
pub fn fr_dielectric(cos_theta_i: f64, eta_i: f64, eta_t: f64) -> f64 {
    let mut cos_theta_i = clamp_t(cos_theta_i, -1.0, 1.0);
    let entering: bool = cos_theta_i > 0.0;
    let mut local_eta_i = eta_i;
    let mut local_eta_t = eta_t;
    if !entering {
        std::mem::swap(&mut local_eta_i, &mut local_eta_t);
        cos_theta_i = cos_theta_i.abs();
    }
    // compute _cos_theta_t_ using Snell's law
    // Snell's law : eta_i * sin_phi_i = eta_t * sin_phi_t
    let sin_theta_i: f64 = (0.0 as f64).max(1.0 - cos_theta_i * cos_theta_i).sqrt();
    let sin_theta_t: f64 = local_eta_i / local_eta_t * sin_theta_i;
    // handle total internal reflection
    if sin_theta_t >= 1.0 {
        return 1.0;
    }
    let cos_theta_t: f64 = (0.0 as f64).max(1.0 - sin_theta_t * sin_theta_t).sqrt();
    let r_parl: f64 = ((local_eta_t * cos_theta_i) - (local_eta_i * cos_theta_t))
        / ((local_eta_t * cos_theta_i) + (local_eta_i * cos_theta_t));
    let r_perp: f64 = ((local_eta_i * cos_theta_i) - (local_eta_t * cos_theta_t))
        / ((local_eta_i * cos_theta_i) + (local_eta_t * cos_theta_t));
    (r_parl * r_parl + r_perp * r_perp) / 2.0
}

pub fn fr_conductor(
    cos_theta_i: f64,
    eta_i: Spectrum<SPECTRUM_N>,
    eta_t: Spectrum<SPECTRUM_N>,
    k: Spectrum<SPECTRUM_N>,
) -> Spectrum<SPECTRUM_N> {
    let not_clamped: f64 = cos_theta_i;
    let cos_theta_i: f64 = clamp_t(not_clamped, -1.0, 1.0);
    let eta: Spectrum<SPECTRUM_N> = eta_t / eta_i;
    let eta_k: Spectrum<SPECTRUM_N> = k / eta_i;
    let cos_theta_i2: f64 = cos_theta_i * cos_theta_i;
    let sin_theta_i2: f64 = 1.0 - cos_theta_i2;
    let eta_2: Spectrum<SPECTRUM_N> = eta * eta;
    let eta_k2: Spectrum<SPECTRUM_N> = eta_k * eta_k;
    let t0: Spectrum<SPECTRUM_N> = eta_2 - eta_k2 - Spectrum::from(sin_theta_i2);
    let a2_plus_b2: Spectrum<SPECTRUM_N> = (t0 * t0 + eta_2 * eta_k2 * Spectrum::from(4.0)).sqrt();
    let t1: Spectrum<SPECTRUM_N> = a2_plus_b2 + Spectrum::from(cos_theta_i2);
    let a: Spectrum<SPECTRUM_N> = ((a2_plus_b2 + t0) * 0.5).sqrt();
    let t2: Spectrum<SPECTRUM_N> = a * 2.0 * cos_theta_i;
    let rs: Spectrum<SPECTRUM_N> = (t1 - t2) / (t1 + t2);
    let t3: Spectrum<SPECTRUM_N> =
        a2_plus_b2 * cos_theta_i2 + Spectrum::from(sin_theta_i2 * sin_theta_i2);
    let t4: Spectrum<SPECTRUM_N> = t2 * sin_theta_i2;
    let rp: Spectrum<SPECTRUM_N> = rs * (t3 - t4) / (t3 + t4);
    (rp + rs) * Spectrum::from(0.5)
}

#[inline]
fn pow5(v: f64) -> f64 {
    v.powi(5)
}

const MAX_BXDFS: usize = 8;

#[derive(Debug, Clone)]
pub struct Bsdf {
    pub eta: f64,

    ns: Normal3f,
    ng: Normal3f,
    ss: Vector3f,
    ts: Vector3f,
    pub bxdfs: Vec<Rc<dyn BxDF>>,
}

impl Bsdf {
    pub fn new(si: &SurfaceInteraction, eta: f64) -> Self {
        let ns = si.shading.n;
        let ss = si.shading.dpdu.normalize();
        Self {
            eta,
            ns,
            ng: si.ist.n,
            ss,
            ts: cross(&ns, &ss),
            bxdfs: vec![],
        }
    }
    pub fn add(&mut self, b: Rc<dyn BxDF>) {
        assert!(self.bxdfs.len() < MAX_BXDFS);
        self.bxdfs.push(b);
    }
    pub fn num_components(&self, flags: BxDFType) -> usize {
        let mut n: usize = 0;
        for b in &self.bxdfs {
            if b.match_flags(flags) {
                n += 1;
            }
        }
        n
    }
    pub fn world_to_local(&self, v: &Vector3f) -> Vector3f {
        Vector3f::new(dot3(v, &self.ss), dot3(v, &self.ts), dot3(v, &self.ns))
    }
    pub fn local_to_world(&self, v: &Vector3f) -> Vector3f {
        Vector3f::new(
            self.ss.x * v.x + self.ts.x * v.y + self.ns.x * v.z,
            self.ss.y * v.x + self.ts.y * v.y + self.ns.y * v.z,
            self.ss.z * v.x + self.ts.z * v.y + self.ns.z * v.z,
        )
    }
    // BSDF Method Definitions
    pub fn f(&self, wo_w: &Vector3f, wi_w: &Vector3f, flags: BxDFType) -> Spectrum<SPECTRUM_N> {
        let wi = self.world_to_local(wi_w);
        let wo = self.world_to_local(wo_w);
        if wo.z == 0.0 {
            return Spectrum::zero();
        }
        let reflect = dot3(wi_w, &self.ng) * dot3(wo_w, &self.ng) > 0.0;
        let mut f = Spectrum::zero();
        // It starts by transforming the world space direction vectors to local BSDF space
        // and then determines whether it should use the BRDFs or the BTDFs.
        for b in &self.bxdfs {
            if b.match_flags(flags) && ((reflect && b.is_refl()) || (!reflect && b.is_trans())) {
                f += b.f(&wo, &wi);
            }
        }
        f
    }
    /// hemispherical_hemispherical reflectance
    fn rho_hh(
        &self,
        n_samples: usize,
        samples1: &[Point2f],
        samples2: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        let mut ret = Spectrum::zero();
        for b in &self.bxdfs {
            if b.match_flags(flags) {
                ret += b.rho_hh(n_samples, samples1, samples2);
            }
        }
        ret
    }
    /// hemispherical_directional reflectance
    fn rho_hd(
        &self,
        wo_world: &Vector3f,
        n_samples: usize,
        samples: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        let wo = self.world_to_local(wo_world);
        let mut ret = Spectrum::zero();
        for b in &self.bxdfs {
            if b.match_flags(flags) {
                ret += b.rho_hd(&wo, n_samples, samples);
            }
        }
        ret
    }
    pub fn sample_f(
        &self,
        wo_world: &Vector3f,
        wi_world: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f64,
        flags: BxDFType,
        sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        // Choose which _BxDF_ to sample
        let matching_comps = self.num_components(flags);
        if matching_comps == 0 {
            *pdf = 0.0;
            *sampled_type = BXDF_NONE;
            return Spectrum::zero();
        }

        let comp = ((u[0] * matching_comps as f64).floor() as usize).min(matching_comps);
        // Get _BxDF_ pointer for chosen component
        let mut count = comp;
        let mut chosen_idx = 0;
        let mut chosen_bxdf = None;
        for i in 0..self.bxdfs.len() {
            if self.bxdfs[i].match_flags(flags) {
                if count == 0 {
                    chosen_idx = i;
                    chosen_bxdf = Some(&self.bxdfs[i]);
                    break;
                }
                count -= 1;
            }
        }
        // assert!(chosen_bxdf.is_some());
        let bxdf = chosen_bxdf.expect("Did not Choose Any BxDF");

        // Remap _BxDF_ sample _u_ to $[0,1)^2$
        let u_remapped = Point2f::new(
            (u[0] * matching_comps as f64 - comp as f64).min(ONE_MINUS_EPSILON),
            u[1],
        );
        // Sample chosen _BxDF_
        let mut wi: Vector3f = Vector3f::default();
        let wo: Vector3f = self.world_to_local(wo_world);

        if wo.z == 0.0 {
            return Spectrum::zero();
        }
        *pdf = 0.0;
        *sampled_type = bxdf.bxdf_type();
        let f = bxdf.sample_f(&wo, &mut wi, &u_remapped, pdf, sampled_type);
        if *pdf == 0.0 {
            *sampled_type = BXDF_NONE;
            return Spectrum::zero();
        }
        *wi_world = self.local_to_world(&wi);
        // Compute overall PDF with all matching _BxDF_s
        if !bxdf.is_refl() && matching_comps > 1 {
            for i in 0..self.bxdfs.len() {
                if i != chosen_idx && self.bxdfs[i].match_flags(flags) {
                    *pdf += self.bxdfs[i].pdf(&wo, &wi);
                }
            }
        }
        if matching_comps > 1 {
            *pdf /= matching_comps as f64;
        }
        // Compute value of BSDF for sampled direction
        if !bxdf_is_spec(bxdf.bxdf_type()) {
            let reflect = dot3(wi_world, &self.ng) * dot3(wo_world, &self.ng) > 0.0;
            let mut f = Spectrum::zero();
            for bx in &self.bxdfs {
                if bx.match_flags(flags)
                    && ((reflect && bx.is_refl()) || (!reflect && bx.is_trans()))
                {
                    f += bx.f(&wo, &wi);
                }
            }
        }
        f
    }
    fn pdf(&self, wo_world: &Vector3f, wi_world: &Vector3f, flags: BxDFType) -> f64 {
        if self.bxdfs.len() == 0 {
            return 0.0;
        }
        let wo = self.world_to_local(wo_world);
        let wi = self.world_to_local(wi_world);
        if wo.z == 0.0 {
            return 0.0;
        }
        let mut pdf = 0.0;
        let mut matching_comps = 0;
        for bx in &self.bxdfs {
            if bx.match_flags(flags) {
                matching_comps += 1;
                pdf += bx.pdf(&wo, &wi);
            }
        }
        if matching_comps > 0 {
            pdf / matching_comps as f64
        } else {
            0.0
        }
    }
}

/// The BSDF class, which will be introduced in Section 9.1, holds a collection of BxDF objects
/// that together describe the scattering at a point on a surface.
/// Although we are hiding the implementation details of the BxDF behind a common interface for reflective and transmissive materials,
/// some of the light transport algorithms in Chapters 14 through 16 will need to distinguish between these two types.
/// Therefore, allBxDFs have a BxDF::type member that holds flags from BxDFType.
/// For each BxDF, the flags should have at least one of
/// BSDF_REFLECTION or BSDF_TRANSMISSION set and exactly one of the BXDF_DIFFUSE, BXDF_GLOSSY, and BXDF_SPECULAR flags.
/// Note that there is no retro-reflective flag; retro-reflection is treated as glossy reflection in this categorization
pub const BXDF_REFLECTION: u8 = 1 << 0;
pub const BXDF_TRANSMISSION: u8 = 1 << 1;
pub const BXDF_DIFFUSE: u8 = 1 << 2;
pub const BXDF_GLOSSY: u8 = 1 << 3;
pub const BXDF_SPECULAR: u8 = 1 << 4;
pub const BXDF_ALL: u8 =
    BXDF_REFLECTION | BXDF_TRANSMISSION | BXDF_DIFFUSE | BXDF_GLOSSY | BXDF_SPECULAR;
pub const BXDF_NONE: u8 = 0;
pub type BxDFType = u8;

pub fn bxdf_is_refl(t: u8) -> bool {
    t & BXDF_REFLECTION > 0
}

pub fn bxdf_is_trans(t: u8) -> bool {
    t & BXDF_TRANSMISSION > 0
}

pub fn bxdf_is_diff(t: u8) -> bool {
    t & BXDF_DIFFUSE > 0
}

pub fn bxdf_is_glos(t: u8) -> bool {
    t & BXDF_GLOSSY > 0
}

pub fn bxdf_is_spec(t: u8) -> bool {
    t & BXDF_SPECULAR > 0
}

pub trait BxDF: std::fmt::Debug {
    // BxDF Interface
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N>;
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        *wi = cosine_sample_hemisphere(*sample);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }
        *pdf = self.pdf(wo, wi);
        return self.f(wo, wi);
    }
    /// hemispherical_directional reflectance
    fn rho_hd(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum<SPECTRUM_N> {
        let mut r = Spectrum::zero();
        for i in 0..n_samples {
            // Estimate one term of $\rho_\roman{hd}$
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let mut temp = 0;
            let f = self.sample_f(wo, &mut wi, &samples[i], &mut pdf, &mut temp);
            if pdf > 0.0 {
                r += f * abs_cos_theta(&wi) / pdf;
            }
        }
        r / n_samples as f64
    }
    /// hemispherical_hemispherical reflectance
    fn rho_hh(
        &self,
        n_samples: usize,
        samples1: &[Point2f],
        samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        let mut r = Spectrum::zero();
        for i in 0..n_samples {
            // Estimate one term of $\rho_\roman{hh}$
            let wo;
            let mut wi = Vector3f::default();
            wo = uniform_sample_hemisphere(&samples1[i]);
            let pdfo = uniform_hemisphere_pdf();
            let mut pdfi = 0.0;
            let mut temp = 0;
            let f = self.sample_f(&wo, &mut wi, &samples2[i], &mut pdfi, &mut temp);
            if pdfi > 0.0 {
                r += f * abs_cos_theta(&wi) * abs_cos_theta(&wo) / (pdfo * pdfi);
            }
        }
        r / (PI * n_samples as f64)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if same_hemisphere(wo, wi) {
            return abs_cos_theta(wi) / PI;
        } else {
            return 0.0;
        }
    }
    fn bxdf_type(&self) -> BxDFType;
    fn match_flags(&self, flags: BxDFType) -> bool {
        (self.bxdf_type() & flags) == self.bxdf_type()
    }
    fn is_refl(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_REFLECTION > 0
    }
    fn is_trans(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_TRANSMISSION > 0
    }
    fn is_diff(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_DIFFUSE > 0
    }
    fn is_glos(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_GLOSSY > 0
    }
    fn is_spec(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_SPECULAR > 0
    }
}

#[derive(Debug)]
pub struct ScaledBxdf {
    bxdf: Rc<dyn BxDF>,
    scale: Spectrum<SPECTRUM_N>,
}

impl ScaledBxdf {
    pub fn new(bxdf: Rc<dyn BxDF>, scale: Spectrum<SPECTRUM_N>) -> Self {
        Self { bxdf, scale }
    }
}

impl BxDF for ScaledBxdf {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        self.scale * self.bxdf.f(wo, wi)
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        self.scale * self.bxdf.sample_f(wo, wi, sample, pdf, sampled_type)
    }

    fn rho_hd(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum<SPECTRUM_N> {
        self.scale * self.bxdf.rho_hd(wo, n_samples, samples)
    }

    fn rho_hh(
        &self,
        n_samples: usize,
        samples1: &[Point2f],
        samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        self.scale * self.bxdf.rho_hh(n_samples, samples1, samples2)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        self.bxdf.pdf(wo, wi)
    }

    fn bxdf_type(&self) -> BxDFType {
        self.bxdf.bxdf_type()
    }
}

pub trait Fresnel: std::fmt::Debug {
    fn evaluate(&self, cos_i: f64) -> Spectrum<SPECTRUM_N>;
}

#[derive(Debug, Default, Copy, Clone)]
pub struct FresnelDielectric {
    eta_i: f64,
    eta_t: f64,
}

impl FresnelDielectric {
    pub fn new(eta_i: f64, eta_t: f64) -> Self {
        Self { eta_i, eta_t }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct FresnelConductor {
    eta_i: Spectrum<SPECTRUM_N>,
    eta_t: Spectrum<SPECTRUM_N>,
    k: Spectrum<SPECTRUM_N>,
}

impl FresnelConductor {
    pub fn new(
        eta_i: Spectrum<SPECTRUM_N>,
        eta_t: Spectrum<SPECTRUM_N>,
        k: Spectrum<SPECTRUM_N>,
    ) -> Self {
        Self { eta_i, eta_t, k }
    }
}

// The FresnelNoOp implementation of the Fresnel interface returns 100% reflection for all incoming directions.
// Although this is physically implausible, it is a convenient capabilityto have available.
#[derive(Debug, Default, Copy, Clone)]
pub struct FresnelNoOp {}

impl Fresnel for FresnelDielectric {
    fn evaluate(&self, cos_i: f64) -> Spectrum<SPECTRUM_N> {
        Spectrum::from(fr_dielectric(cos_i, self.eta_i, self.eta_t))
    }
}

impl Fresnel for FresnelConductor {
    fn evaluate(&self, cos_i: f64) -> Spectrum<SPECTRUM_N> {
        fr_conductor(cos_i.abs(), self.eta_i, self.eta_t, self.k)
    }
}

impl Fresnel for FresnelNoOp {
    fn evaluate(&self, _cos_i: f64) -> Spectrum<SPECTRUM_N> {
        Spectrum::one()
    }
}

#[derive(Debug)]
pub struct SpecularReflection {
    r: Spectrum<SPECTRUM_N>,
    fresnel: Box<dyn Fresnel>,
}

impl SpecularReflection {
    pub fn new(r: Spectrum<SPECTRUM_N>, fresnel: Box<dyn Fresnel>) -> Self {
        Self { r, fresnel }
    }
}

impl BxDF for SpecularReflection {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        Spectrum::zero()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        _sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        *wi = Vector3f::new(-wo.x, -wo.y, wo.z);
        *pdf = 1.0;
        self.fresnel.evaluate(cos_theta(wi)) * self.r / abs_cos_theta(wi)
    }

    fn pdf(&self, _wo: &Vector3f, _wi: &Vector3f) -> f64 {
        0.0
    }

    fn bxdf_type(&self) -> BxDFType {
        BXDF_REFLECTION | BXDF_SPECULAR
    }
}

#[derive(Debug)]
pub struct SpecularTransmission {
    t: Spectrum<SPECTRUM_N>,
    eta_a: f64,
    eta_b: f64,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl SpecularTransmission {
    pub fn new(t: Spectrum<SPECTRUM_N>, eta_a: f64, eta_b: f64, mode: TransportMode) -> Self {
        let fresnel = FresnelDielectric::new(eta_a, eta_b);
        Self {
            t,
            eta_a,
            eta_b,
            fresnel,
            mode,
        }
    }
}

impl BxDF for SpecularTransmission {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        Spectrum::zero()
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        _sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        // Figure out which $\eta$ is incident and which is transmitted
        let entering = cos_theta(wo) > 0.0;
        let eta_i = if entering { self.eta_a } else { self.eta_b };
        let eta_t = if entering { self.eta_b } else { self.eta_a };

        // Compute ray direction for specular transmission
        if !refract(
            wo,
            &faceforward(&Normal3f::new(0.0, 0.0, 1.0), wo),
            eta_i / eta_t,
            wi,
        ) {
            return Spectrum::zero();
        }
        *pdf = 1.0;
        let mut ft = self.t * (Spectrum::one() - self.fresnel.evaluate(cos_theta(wi)));
        // Account for non-symmetry with transmission to different medium
        if self.mode == TransportMode::Radiance {
            ft *= (eta_i * eta_i) / (eta_t * eta_t);
        }
        ft / abs_cos_theta(wi)
    }
    fn pdf(&self, _wo: &Vector3f, _wi: &Vector3f) -> f64 {
        0.0
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_SPECULAR | BXDF_TRANSMISSION
    }
}

#[derive(Debug)]
pub struct FresnelSpecular {
    r: Spectrum<SPECTRUM_N>,
    t: Spectrum<SPECTRUM_N>,
    eta_a: f64,
    eta_b: f64,
    mode: TransportMode,
}

impl FresnelSpecular {
    pub fn new(
        r: Spectrum<SPECTRUM_N>,
        t: Spectrum<SPECTRUM_N>,
        eta_a: f64,
        eta_b: f64,
        mode: TransportMode,
    ) -> Self {
        Self {
            r,
            t,
            eta_a,
            eta_b,
            mode,
        }
    }
}

impl BxDF for FresnelSpecular {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        return Spectrum::zero();
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        let f = fr_dielectric(cos_theta(wo), self.eta_a, self.eta_b);
        if sample[0] < f {
            // Compute specular reflection for _FresnelSpecular_
            // Compute perfect specular reflection direction
            *wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            *sampled_type = BXDF_SPECULAR | BXDF_REFLECTION;
            *pdf = f;
            return self.r * f / abs_cos_theta(wi);
        } else {
            // Compute specular transmission for _FresnelSpecular_
            // Figure out which $\eta$ is incident and which is transmitted
            let entering = cos_theta(wo) > 0.0;
            let eta_i = if entering { self.eta_a } else { self.eta_b };
            let eta_t = if entering { self.eta_b } else { self.eta_a };

            // Compute ray direction for specular transmission
            if !refract(
                wo,
                &faceforward(&Normal3f::new(0.0, 0.0, 1.0), wo),
                eta_i / eta_t,
                wi,
            ) {
                return Spectrum::zero();
            }

            let mut ft = self.t * (1.0 - f);
            // Account for non-symmetry with transmission to different medium
            if self.mode == TransportMode::Radiance {
                ft *= (eta_i * eta_i) / (eta_t * eta_t);
            }
            *sampled_type = BXDF_SPECULAR | BXDF_TRANSMISSION;
            *pdf = 1.0 - f;
            return ft / abs_cos_theta(wi);
        }
    }
    fn pdf(&self, _wo: &Vector3f, _wi: &Vector3f) -> f64 {
        0.0
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_SPECULAR | BXDF_ALL
    }
}

#[derive(Debug)]
pub struct LambertianReflection {
    r: Spectrum<SPECTRUM_N>,
}

impl LambertianReflection {
    pub fn new(r: Spectrum<SPECTRUM_N>) -> Self {
        Self { r }
    }
}

impl BxDF for LambertianReflection {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        self.r / PI
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        return self.r;
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        return self.r;
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

#[derive(Debug)]
pub struct LambertianTransmission {
    t: Spectrum<SPECTRUM_N>,
}

impl LambertianTransmission {
    pub fn new(t: Spectrum<SPECTRUM_N>) -> Self {
        Self { t }
    }
}

impl BxDF for LambertianTransmission {
    fn f(&self, _wo: &Vector3f, _wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        self.t / PI
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        *wi = cosine_sample_hemisphere(*sample);
        if wo.z > 0.0 {
            wi.z *= -1.0;
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }
    fn rho_hd(
        &self,
        _wo: &Vector3f,
        _n_samples: usize,
        _samples: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        return self.t;
    }
    fn rho_hh(
        &self,
        _n_samples: usize,
        _samples1: &[Point2f],
        _samples2: &[Point2f],
    ) -> Spectrum<SPECTRUM_N> {
        return self.t;
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if !same_hemisphere(wo, wi) {
            return abs_cos_theta(wi) / PI;
        } else {
            return 0.0;
        }
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_TRANSMISSION
    }

    fn match_flags(&self, flags: BxDFType) -> bool {
        (self.bxdf_type() & flags) == self.bxdf_type()
    }

    fn is_refl(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_REFLECTION > 0
    }

    fn is_trans(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_TRANSMISSION > 0
    }

    fn is_diff(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_DIFFUSE > 0
    }

    fn is_glos(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_GLOSSY > 0
    }

    fn is_spec(&self) -> bool {
        let t = self.bxdf_type();
        t & BXDF_SPECULAR > 0
    }
}

#[derive(Debug)]
pub struct OrenNayar {
    r: Spectrum<SPECTRUM_N>,
    a: f64,
    b: f64,
}

impl OrenNayar {
    pub fn new(r: Spectrum<SPECTRUM_N>, sigma: f64) -> Self {
        let sigma2 = radians(sigma).powi(2);
        let a = 1. - (sigma2 / (2. * (sigma2 + 0.33)));
        let b = 0.45 * sigma2 / (sigma2 + 0.09);
        Self { r, a, b }
    }
}

impl BxDF for OrenNayar {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let sin_theta_i = sin_theta(wi);
        let sin_theta_o = sin_theta(wo);
        // Compute cosine term of Oren-Nayar model
        let mut max_cos = 0.0;
        if sin_theta_i > 1e-4 && sin_theta_o > 1e-4 {
            let sin_phi_i = sin_phi(wi);
            let cos_phi_i = cos_phi(wi);
            let sin_phi_o = sin_phi(wo);
            let cos_phi_o = cos_phi(wo);
            let d_cos = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o;
            max_cos = d_cos.max(0.0);
        }
        // Compute sine and tangent terms of Oren-Nayar model
        let sin_alpha: f64;
        let tan_beta: f64;
        if abs_cos_theta(wi) > abs_cos_theta(wo) {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / abs_cos_theta(wi);
        } else {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / abs_cos_theta(wo);
        }
        self.r / PI * (self.a + self.b * max_cos * sin_alpha * tan_beta)
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_DIFFUSE | BXDF_REFLECTION
    }
}

#[derive(Debug)]
pub struct MicrofacetReflection {
    r: Spectrum<SPECTRUM_N>,
    distribution: Box<dyn MicrofacetDistribution>,
    fresnel: Box<dyn Fresnel>,
}

impl MicrofacetReflection {
    pub fn new(
        r: Spectrum<SPECTRUM_N>,
        distribution: Box<dyn MicrofacetDistribution>,
        fresnel: Box<dyn Fresnel>,
    ) -> Self {
        Self {
            r,
            distribution,
            fresnel,
        }
    }
}

impl BxDF for MicrofacetReflection {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        let mut wh = *wi + *wo;
        // Handle degenerate cases for microfacet reflection
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return Spectrum::zero();
        }
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::zero();
        }
        wh = wh.normalize();
        // For the Fresnel call, make sure that wh is in the same hemisphere
        // as the surface normal, so that TIR is handled correctly.
        let f = self.fresnel.evaluate(dot3(
            wi,
            &faceforward(&wh.into(), &Vector3f::new(0.0, 0.0, 1.0)),
        ));
        self.r * self.distribution.d(&wh) * self.distribution.g(wo, wi) * f
            / (4.0 * cos_theta_i * cos_theta_o)
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        if wo.z == 0.0 {
            return Spectrum::zero();
        }
        let wh = self.distribution.sample_wh(wo, *sample);
        // Should be rare
        if dot3(wo, &wh) < 0.0 {
            return Spectrum::zero();
        }
        *wi = reflect(wo, &wh);
        if !same_hemisphere(wo, wi) {
            return Spectrum::zero();
        }

        // Compute PDF of _wi_ for microfacet reflection
        *pdf = self.distribution.pdf(wo, &wh) / (4.0 * dot3(wo, &wh));
        self.f(wo, wi)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        let wh = (*wo + *wi).normalize();
        self.distribution.pdf(wo, &wh) / (4.0 * dot3(wo, &wh))
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_GLOSSY | BXDF_REFLECTION
    }
}

#[derive(Debug)]
pub struct MicrofacetTransmission {
    t: Spectrum<SPECTRUM_N>,
    distribution: Box<dyn MicrofacetDistribution>,
    eta_a: f64,
    eta_b: f64,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl MicrofacetTransmission {
    pub fn new(
        t: Spectrum<SPECTRUM_N>,
        distribution: Box<dyn MicrofacetDistribution>,
        eta_a: f64,
        eta_b: f64,
        mode: TransportMode,
    ) -> Self {
        let fresnel = FresnelDielectric::new(eta_a, eta_b);
        Self {
            t,
            distribution,
            eta_a,
            eta_b,
            fresnel,
            mode,
        }
    }
}

impl BxDF for MicrofacetTransmission {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        // transmission only
        if same_hemisphere(wo, wi) {
            return Spectrum::zero();
        }
        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return Spectrum::zero();
        }

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        let eta = if cos_theta(wo) > 0.0 {
            self.eta_b / self.eta_a
        } else {
            self.eta_a / self.eta_b
        };
        let mut wh = (*wo + *wi * eta).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        let f = self.fresnel.evaluate(dot3(wo, &wh));
        let sqrt_denom = dot3(wo, &wh) + eta * dot3(wi, &wh);
        let factor = if self.mode == TransportMode::Radiance {
            1.0 / eta
        } else {
            1.0
        };

        (Spectrum::one() - f)
            * self.t
            * (self.distribution.d(&wh)
                * self.distribution.g(wo, wi)
                * eta
                * eta
                * abs_dot3(wi, &wh)
                * abs_dot3(wo, &wh)
                * factor
                * factor
                / (cos_theta_i * cos_theta_o * sqrt_denom * sqrt_denom))
                .abs()
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        if wo.z == 0.0 {
            return Spectrum::zero();
        }
        let wh = self.distribution.sample_wh(wo, *sample);
        // Should be rare
        if dot3(wo, &wh) < 0.0 {
            return Spectrum::zero();
        }
        let eta = if cos_theta(wo) > 0.0 {
            self.eta_a / self.eta_b
        } else {
            self.eta_b / self.eta_a
        };
        if !refract(wo, &wh.into(), eta, wi) {
            return Spectrum::zero();
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if same_hemisphere(wo, wi) {
            return 0.0;
        }
        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        let eta = if cos_theta(wo) > 0.0 {
            self.eta_b / self.eta_a
        } else {
            self.eta_a / self.eta_b
        };

        let wh = (*wo + *wi * eta).normalize();

        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        let sqrt_denom = dot3(wo, &wh) + dot3(wi, &wh) * eta;
        let dwh_dwi = ((eta * eta * dot3(wi, &wh)) / (sqrt_denom * sqrt_denom)).abs();

        self.distribution.pdf(wo, &wh) * dwh_dwi
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_GLOSSY | BXDF_TRANSMISSION
    }
}

#[derive(Debug)]
pub struct FresnelBlend {
    rd: Spectrum<SPECTRUM_N>,
    rs: Spectrum<SPECTRUM_N>,
    distribution: Box<dyn MicrofacetDistribution>,
}

impl FresnelBlend {
    pub fn new(
        rd: Spectrum<SPECTRUM_N>,
        rs: Spectrum<SPECTRUM_N>,
        distribution: Box<dyn MicrofacetDistribution>,
    ) -> Self {
        Self {
            rd,
            rs,
            distribution,
        }
    }
    fn schlick_fresnel(&self, cos_theta: f64) -> Spectrum<SPECTRUM_N> {
        self.rs + (Spectrum::one() - self.rs) * pow5(1.0 - cos_theta)
    }
}

impl BxDF for FresnelBlend {
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let diffuse: Spectrum<SPECTRUM_N> = self.rd
            * (0.28 / (23.0 * PI))
            * (Spectrum::one() - self.rs)
            * (1.0 - pow5(1.0 - 0.5 * abs_cos_theta(wi)))
            * (1.0 - pow5(1.0 - 0.5 * abs_cos_theta(wo)));
        let wh: Vector3f = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Spectrum::one();
        }

        let wh = wh.normalize();
        let specular: Spectrum<SPECTRUM_N> = self.schlick_fresnel(dot3(wi, &wh))
            * self.distribution.d(&wh)
            / (4.0 * abs_dot3(wi, &wh) * abs_cos_theta(wi).max(abs_cos_theta(wo)));

        diffuse + specular
    }
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u_orig: &Point2f,
        pdf: &mut f64,
        _sampled_type: &mut BxDFType,
    ) -> Spectrum<SPECTRUM_N> {
        let mut u = *u_orig;

        if u[0] < 0.5 {
            u[0] = (ONE_MINUS_EPSILON).min(2.0 * u[0]);
            // Cosine-sample the hemisphere, flipping the direction if necessary
            *wi = cosine_sample_hemisphere(u);
            if wo.z < 0.0 {
                wi.z *= -1.0;
            }
        } else {
            u[0] = (2.0 * (u[0] - 0.5)).min(ONE_MINUS_EPSILON);
            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            let wh = self.distribution.sample_wh(wo, u);
            *wi = reflect(wo, &wh);
            if !same_hemisphere(wo, wi) {
                return Spectrum::one();
            }
        }

        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f64 {
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        let wh = (*wo + *wi).normalize();
        let pdf_wh = self.distribution.pdf(wo, &wh);
        0.5 * (abs_cos_theta(wi) / PI + pdf_wh / (4.0 * dot3(wo, &wh)))
    }
    fn bxdf_type(&self) -> BxDFType {
        BXDF_GLOSSY | BXDF_REFLECTION
    }
}
