use std::{
    f64::{consts::PI, INFINITY},
    fmt::Debug,
    rc::Rc,
    sync::Arc,
};

use crate::{
    geometry::{dot3, pnt3_distance, Normal3f, Point2f, Vector3f},
    interaction::{BaseInteraction, SurfaceInteraction},
    interpolation::{catmull_rom_weights, integrate_catmull_rom, sample_catmull_rom_2d},
    material::{Material, TransportMode},
    medium::phase_hg,
    misc::{clamp_t, INV_4_PI},
    primitives::Primitive,
    reflection::{cos_theta, fr_dielectric, Bsdf, BxDF, BXDF_DIFFUSE, BXDF_REFLECTION},
    rtoycore::SPECTRUM_N,
    scene::Scene,
    spectrum::Spectrum,
};

#[derive(Debug, Clone)]
pub struct BSSRDFData {
    po: SurfaceInteraction,
    eta: f64,
}

impl BSSRDFData {
    pub fn new(po: SurfaceInteraction, eta: f64) -> Self {
        Self { po, eta }
    }
}

pub trait BSSRDF: Debug {
    // BSSRDF Interface
    fn s(&self, pi: &SurfaceInteraction, wi: &Vector3f) -> Spectrum<SPECTRUM_N>;
    fn sample_s(
        &self,
        scene: &Scene,
        u1: f64,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut f64,
    ) -> Spectrum<SPECTRUM_N>;
    // fn bssrdfdata(&self) -> &BSSRDFData;
}

#[derive(Debug, Clone)]
pub struct SeparableBSSRDFData {
    // SeparableBSSRDF Private Data
    ns: Normal3f,
    ss: Vector3f,
    ts: Vector3f,
    material: Arc<dyn Material>,
    mode: TransportMode,
}

impl SeparableBSSRDFData {
    pub fn new(
        ns: Normal3f,
        ss: Vector3f,
        ts: Vector3f,
        material: Arc<dyn Material>,
        mode: TransportMode,
    ) -> Self {
        Self {
            ns,
            ss,
            ts,
            material,
            mode,
        }
    }
}

/// SeparableBSSRDF is a close enough approcimation to give a bssrdf, thus providing a good BSSRDF implementation
#[derive(Debug, Clone)]
pub struct SeparableBSSRDF {
    bd: BSSRDFData,
    sbd: SeparableBSSRDFData,
    // Generics part
    s: Rc<dyn ISeparableBSSRDF>,
}

impl SeparableBSSRDF {
    pub fn new(bd: BSSRDFData, sbd: SeparableBSSRDFData, s: Rc<dyn ISeparableBSSRDF>) -> Self {
        Self { bd, sbd, s }
    }
    fn sp(&self, pi: &SurfaceInteraction) -> Spectrum<SPECTRUM_N> {
        self.s.sr(
            pnt3_distance(&self.bd.po.ist.p, &pi.ist.p),
            &self.bd,
            &self.sbd,
        )
    }
    fn sw(&self, w: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let c = 1.0 - 2.0 * fresnel_moment1(1.0 / self.bd.eta);
        Spectrum::from((1.0 - fr_dielectric(cos_theta(w), 1.0, self.bd.eta)) / (c * PI))
    }
    fn sample_sp(
        &self,
        scene: &Scene,
        u1: f64,
        u2: &Point2f,
        pi: &mut SurfaceInteraction,
        pdf: &mut f64,
    ) -> Spectrum<SPECTRUM_N> {
        // ProfilePhase pp(Prof::BSSRDFEvaluation);
        // Choose projection axis for BSSRDF sampling
        let vx;
        let vy;
        let vz;
        let mut u1 = u1;
        if u1 < 0.5 {
            vx = self.sbd.ss;
            vy = self.sbd.ts;
            vz = Vector3f::from(self.sbd.ns);
            u1 *= 2.0;
        } else if u1 < 0.75 {
            // Prepare for sampling rays with respect to _ss_
            vx = self.sbd.ts;
            vy = Vector3f::from(self.sbd.ns);
            vz = self.sbd.ss;
            u1 = (u1 - 0.5) * 4.0;
        } else {
            // Prepare for sampling rays with respect to _ts_
            vx = Vector3f::from(self.sbd.ns);
            vy = self.sbd.ss;
            vz = self.sbd.ts;
            u1 = (u1 - 0.75) * 4.0;
        }

        // Choose spectral channel for BSSRDF sampling
        let ch = clamp_t(u1 * SPECTRUM_N as f64, 0.0, (SPECTRUM_N - 1) as f64) as usize;
        u1 = u1 * u1 * SPECTRUM_N as f64 - ch as f64;
        // Sample BSSRDF profile in polar coordinates
        let r = self.s.sample_sr(ch, u2[0], &self.bd, &self.sbd);
        if r < 0.0 {
            return Spectrum::zero();
        }

        let phi = 2.0 * PI * u2[1];
        // Compute BSSRDF profile bounds and intersection height
        let r_max = self.s.sample_sr(ch, 0.999, &self.bd, &self.sbd);
        if r >= r_max {
            return Spectrum::zero();
        }
        let l = 2.0 * (r_max * r_max - r * r).sqrt();
        // Compute BSSRDF sampling ray segment
        let mut base = BaseInteraction::default();
        base.p = self.bd.po.ist.p + (vx * phi.cos() + vy * phi.sin()) * r - vz * l * 0.5;
        base.time = self.bd.po.ist.time;
        let p_target = base.p + vz * l;

        // Intersect BSSRDF sampling ray against the scene geometry

        let mut chain = vec![];
        let mut n_found = 0_usize;
        // Accumulate chain of intersections along ray
        loop {
            let mut r = base.spawn_ray_to(p_target);
            let mut tmp_si = SurfaceInteraction::default();
            if r.d.length() == 0.0 || !scene.intersect(&mut r, &mut tmp_si) {
                break;
            }
            base = tmp_si.ist.clone();
            if let Some(tmp_pri) = &tmp_si.primitive {
                let tmp_pri = Rc::clone(tmp_pri);
                if Arc::ptr_eq(&tmp_pri.get_material(), &self.sbd.material) {
                    chain.push(tmp_si);
                    n_found += 1;
                }
            }
        }

        // Randomly choose one of several intersections during BSSRDF sampling
        if n_found == 0 {
            return Spectrum::zero();
        }
        let selected = clamp_t((u1 * n_found as f64) as usize, 0, n_found - 1);
        *pi = (&chain[selected]).clone();

        // Compute sample PDF and return the spatial BSSRDF term $\Sp$
        *pdf = self.pdf_sp(pi) / (n_found as f64);
        self.sp(pi)
    }

    fn pdf_sp(&self, pi: &mut SurfaceInteraction) -> f64 {
        // Express $\pti-\pto$ and $\bold{n}_i$ with respect to local coordinates at
        // $\pto$
        let d = self.bd.po.ist.p - pi.ist.p;
        let d_local = Vector3f::new(
            dot3(&self.sbd.ss, &d),
            dot3(&self.sbd.ts, &d),
            dot3(&self.sbd.ns, &d),
        );
        let n_local = Normal3f::new(
            dot3(&self.sbd.ss, &pi.ist.n),
            dot3(&self.sbd.ts, &pi.ist.n),
            dot3(&self.sbd.ns, &pi.ist.n),
        );
        // Compute BSSRDF profile radius under projection along each axis

        let r_proj: [f64; 3] = [
            (d_local.y * d_local.y + d_local.z * d_local.z).sqrt(),
            (d_local.z * d_local.z + d_local.x * d_local.x).sqrt(),
            (d_local.x * d_local.x + d_local.y * d_local.y).sqrt(),
        ];
        // Return combined probability from all BSSRDF sampling strategies
        let mut pdf = 0.0;
        let axis_prob: [f64; 3] = [0.25, 0.25, 0.5];
        let ch_prob = 1.0 / SPECTRUM_N as f64;
        for axis in 0..3 {
            for ch in 0..SPECTRUM_N {
                pdf += self.s.pdf_sr(ch, r_proj[axis], &self.bd, &self.sbd)
                    * n_local[axis as u8].abs()
                    * ch_prob
                    * axis_prob[axis];
            }
        }
        pdf
    }
}

pub trait ISeparableBSSRDF: Debug {
    fn sr(&self, d: f64, bd: &BSSRDFData, sbd: &SeparableBSSRDFData) -> Spectrum<SPECTRUM_N>;
    fn sample_sr(&self, ch: usize, u: f64, bd: &BSSRDFData, sbd: &SeparableBSSRDFData) -> f64;
    fn pdf_sr(&self, ch: usize, r: f64, bd: &BSSRDFData, sbd: &SeparableBSSRDFData) -> f64;
}

impl BSSRDF for SeparableBSSRDF {
    fn s(&self, pi: &SurfaceInteraction, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let ft = fr_dielectric(cos_theta(&self.bd.po.ist.wo), 1.0, self.bd.eta);
        self.sp(pi) * self.sw(wi) * (1.0 - ft)
    }

    fn sample_s(
        &self,
        scene: &Scene,
        u1: f64,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut f64,
    ) -> Spectrum<SPECTRUM_N> {
        let sp = self.sample_sp(scene, u1, u2, si, pdf);
        if !sp.is_black() {
            // Initialize material model at sampled surface interaction
            let mut bsdf = Bsdf::new(si, 1.0);
            bsdf.add(Rc::new(self.clone()));
            si.bsdf = Some(bsdf);
            si.ist.wo = Vector3f::from(si.shading.n);
        }
        sp
    }
}

impl BxDF for SeparableBSSRDF {
    fn f(&self, _wo: &Vector3f, wi: &Vector3f) -> Spectrum<SPECTRUM_N> {
        let mut f = self.sw(wi);
        // Update BSSRDF transmission term to account for adjoint light
        // transport
        if self.sbd.mode == TransportMode::Radiance {
            f *= self.bd.eta * self.bd.eta;
        }
        f
    }

    fn bxdf_type(&self) -> crate::reflection::BxDFType {
        BXDF_REFLECTION | BXDF_DIFFUSE
    }
}

#[derive(Debug, Clone)]
pub struct BSSRDFTable {
    // BSSRDFTable Public Data
    pub n_rho_samples: usize,
    pub n_radius_samples: usize,
    pub rho_samples: Vec<f64>,
    pub radius_samples: Vec<f64>,
    pub profile: Vec<f64>,
    pub rho_eff: Vec<f64>,
    pub profile_cdf: Vec<f64>,
}

impl BSSRDFTable {
    pub fn new(
        n_rho_samples: usize,
        n_radius_samples: usize,
        rho_samples: Vec<f64>,
        radius_samples: Vec<f64>,
        profile: Vec<f64>,
        rho_eff: Vec<f64>,
        profile_cdf: Vec<f64>,
    ) -> Self {
        Self {
            n_rho_samples,
            n_radius_samples,
            rho_samples,
            radius_samples,
            profile,
            rho_eff,
            profile_cdf,
        }
    }
    #[inline]
    pub fn eval_profile(&self, rho_index: usize, radius_index: usize) -> f64 {
        self.profile[rho_index * self.n_radius_samples + radius_index]
    }
}

#[derive(Debug)]
pub struct TabulatedBSSRDF {
    table: BSSRDFTable,
    sigma_t: Spectrum<SPECTRUM_N>,
    rho: Spectrum<SPECTRUM_N>,
}

impl ISeparableBSSRDF for TabulatedBSSRDF {
    fn sr(&self, r: f64, _bd: &BSSRDFData, _sbd: &SeparableBSSRDFData) -> Spectrum<SPECTRUM_N> {
        let mut sr = Spectrum::from(0.0);
        for ch in 0..SPECTRUM_N {
            // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
            let r_optical = r * self.sigma_t[ch];
            // Compute spline weights to interpolate BSSRDF on channel _ch_
            let mut rho_offset = 0_i32;
            let mut radius_offset = 0_i32;
            let mut rho_weights = [0_f64; 4];
            let mut raidus_weights = [0_f64; 4];
            if !catmull_rom_weights(
                &self.table.rho_samples,
                self.rho[ch],
                &mut rho_offset,
                &mut rho_weights,
            ) || !catmull_rom_weights(
                &self.table.radius_samples,
                r_optical,
                &mut radius_offset,
                &mut raidus_weights,
            ) {
                continue;
            }
            // Set BSSRDF value _Sr[ch]_ using tensor spline interpolation
            let mut sr_f = 0.0;

            for i in 0..4 {
                for j in 0..4 {
                    let weight = rho_weights[i] * raidus_weights[j];
                    if weight != 0.0 {
                        sr_f += weight
                            * self
                                .table
                                .eval_profile(rho_offset as usize + i, radius_offset as usize + j);
                    }
                }
            }

            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if r_optical != 0.0 {
                sr_f /= 2.0 * PI * r_optical;
            }
            sr[ch] = sr_f;
        }

        // Transform BSSRDF value into world space units
        sr *= self.sigma_t * self.sigma_t;
        sr.clamp(0.0, INFINITY)
    }

    fn sample_sr(&self, ch: usize, u: f64, _bd: &BSSRDFData, _sbd: &SeparableBSSRDFData) -> f64 {
        if self.sigma_t[ch] == 0.0 {
            return -1.0;
        }
        sample_catmull_rom_2d(
            &self.table.rho_samples,
            &self.table.radius_samples,
            &self.table.profile,
            &self.table.profile_cdf,
            self.rho[ch],
            u,
            None,
            None,
        ) / self.sigma_t[ch]
    }

    fn pdf_sr(&self, ch: usize, r: f64, _bd: &BSSRDFData, _sbd: &SeparableBSSRDFData) -> f64 {
        // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
        let r_optical = r * self.sigma_t[ch];
        // Compute spline weights to interpolate BSSRDF density on channel _ch_

        let mut rho_offset = 0_i32;
        let mut radius_offset = 0_i32;
        let mut rho_weights = [0_f64; 4];
        let mut raidus_weights = [0_f64; 4];
        if !catmull_rom_weights(
            &self.table.rho_samples,
            self.rho[ch],
            &mut rho_offset,
            &mut rho_weights,
        ) || !catmull_rom_weights(
            &self.table.radius_samples,
            r_optical,
            &mut radius_offset,
            &mut raidus_weights,
        ) {
            return 0.0;
        }
        // Return BSSRDF profile density for channel _ch_
        let mut sr = 0_f64;
        let mut rho_eff = 0_f64;
        for i in 0..4 {
            if rho_weights[i] == 0.0 {
                continue;
            }
            rho_eff += self.table.rho_eff[rho_offset as usize + i] * rho_weights[i];
            for j in 0..4 {
                if raidus_weights[j] == 0.0 {
                    continue;
                }
                sr += self
                    .table
                    .eval_profile(rho_offset as usize + i, radius_offset as usize + j)
                    * rho_weights[i]
                    * raidus_weights[j];
            }
        }

        // Cancel marginal PDF factor from tabulated BSSRDF profile
        if r_optical != 0.0 {
            sr /= 2.0 * PI * r_optical;
        }

        (sr * self.sigma_t[ch] * self.sigma_t[ch] / rho_eff).max(0.)
    }
}

pub fn fresnel_moment1(eta: f64) -> f64 {
    let eta2: f64 = eta * eta;
    let eta3: f64 = eta2 * eta;
    let eta4: f64 = eta3 * eta;
    let eta5: f64 = eta4 * eta;
    if eta < 1.0 as f64 {
        0.45966 as f64 - 1.73965 as f64 * eta + 3.37668 as f64 * eta2 - 3.904_945 * eta3
            + 2.49277 as f64 * eta4
            - 0.68441 as f64 * eta5
    } else {
        -4.61686 as f64 + 11.1136 as f64 * eta - 10.4646 as f64 * eta2 + 5.11455 as f64 * eta3
            - 1.27198 as f64 * eta4
            + 0.12746 as f64 * eta5
    }
}

pub fn fresnel_moment2(eta: f64) -> f64 {
    let eta2: f64 = eta * eta;
    let eta3: f64 = eta2 * eta;
    let eta4: f64 = eta3 * eta;
    let eta5: f64 = eta4 * eta;
    if eta < 1.0 as f64 {
        0.27614 as f64 - 0.87350 as f64 * eta + 1.12077 as f64 * eta2 - 0.65095 as f64 * eta3
            + 0.07883 as f64 * eta4
            + 0.04860 as f64 * eta5
    } else {
        let r_eta = 1.0 as f64 / eta;
        let r_eta2 = r_eta * r_eta;
        let r_eta3 = r_eta2 * r_eta;
        -547.033 as f64 + 45.3087 as f64 * r_eta3 - 218.725 as f64 * r_eta2
            + 458.843 as f64 * r_eta
            + 404.557 as f64 * eta
            - 189.519 as f64 * eta2
            + 54.9327 as f64 * eta3
            - 9.00603 as f64 * eta4
            + 0.63942 as f64 * eta5
    }
}

pub fn beam_diffusion_ms(sigma_s: f64, sigma_a: f64, g: f64, eta: f64, r: f64) -> f64 {
    let n_samples: i32 = 100;
    let mut ed: f64 = 0.0;

    // precompute information for dipole integrand

    // compute reduced scattering coefficients $\sigmaps, \sigmapt$
    // and albedo $\rhop$
    let sigmap_s: f64 = sigma_s * (1.0 as f64 - g);
    let sigmap_t: f64 = sigma_a + sigmap_s;
    let rhop: f64 = sigmap_s / sigmap_t;
    // compute non-classical diffusion coefficient $D_\roman{G}$ using
    // Equation (15.24)
    let d_g: f64 = (2.0 as f64 * sigma_a + sigmap_s) / (3.0 as f64 * sigmap_t * sigmap_t);
    // compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    let sigma_tr: f64 = (sigma_a / d_g).sqrt();
    // determine linear extrapolation distance $\depthextrapolation$
    // using Equation (15.28)
    let fm1: f64 = fresnel_moment1(eta);
    let fm2: f64 = fresnel_moment2(eta);
    let ze: f64 =
        -2.0 as f64 * d_g * (1.0 as f64 + 3.0 as f64 * fm2) / (1.0 as f64 - 2.0 as f64 * fm1);
    // determine exitance scale factors using Equations (15.31) and (15.32)
    let c_phi: f64 = 0.25 as f64 * (1.0 as f64 - 2.0 as f64 * fm1);
    let c_e = 0.5 as f64 * (1.0 as f64 - 3.0 as f64 * fm2);
    // for (int i = 0; i < n_samples; ++i) {
    for i in 0..n_samples {
        // sample real point source depth $\depthreal$
        let zr: f64 = -(1.0 as f64 - (i as f64 + 0.5 as f64) / n_samples as f64).ln() / sigmap_t;
        // evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to _ed_
        let zv: f64 = -zr + 2.0 as f64 * ze;
        let dr: f64 = (r * r + zr * zr).sqrt();
        let dv: f64 = (r * r + zv * zv).sqrt();
        // compute dipole fluence rate $\dipole(r)$ using Equation (15.27)
        let phi_d: f64 =
            INV_4_PI / d_g * ((-sigma_tr * dr).exp() / dr - (-sigma_tr * dv).exp() / dv);
        // compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$
        // using Equation (15.27)
        let ed_n: f64 = INV_4_PI
            * (zr * (1.0 as f64 + sigma_tr * dr) * (-sigma_tr * dr).exp() / (dr * dr * dr)
                - zv * (1.0 as f64 + sigma_tr * dv) * (-sigma_tr * dv).exp() / (dv * dv * dv));
        // add contribution from dipole for depth $\depthreal$ to _ed_
        let e: f64 = phi_d * c_phi + ed_n * c_e;
        let kappa: f64 = 1.0 as f64 - (-2.0 as f64 * sigmap_t * (dr + zr)).exp();
        ed += kappa * rhop * rhop * e;
    }
    ed / n_samples as f64
}

pub fn beam_diffusion_ss(sigma_s: f64, sigma_a: f64, g: f64, eta: f64, r: f64) -> f64 {
    // compute material parameters and minimum $t$ below the critical angle
    let sigma_t: f64 = sigma_a + sigma_s;
    let rho: f64 = sigma_s / sigma_t;
    let t_crit: f64 = r * (eta * eta - 1.0 as f64).sqrt();
    let mut ess: f64 = 0.0 as f64;
    let n_samples: i32 = 100;
    for i in 0..n_samples {
        // evaluate single scattering integrand and add to _ess_
        let ti: f64 =
            t_crit - (1.0 as f64 - (i as f64 + 0.5 as f64) / n_samples as f64).ln() / sigma_t;
        // determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        let d: f64 = (r * r + ti * ti).sqrt();
        let cos_theta_o: f64 = ti / d;
        // add contribution of single scattering at depth $t$
        ess += rho * (-sigma_t * (d + t_crit)).exp() / (d * d)
            * phase_hg(cos_theta_o, g)
            * (1.0 as f64 - fr_dielectric(-cos_theta_o, 1.0 as f64, eta))
            * (cos_theta_o).abs();
    }
    ess / n_samples as f64
}

pub fn compute_beam_diffusion_bssrdf(g: f64, eta: f64, t: &mut BSSRDFTable) {
    // choose radius values of the diffusion profile discretization
    t.radius_samples[0] = 0.0 as f64;
    t.radius_samples[1] = 2.5e-3 as f64;
    for i in 2..t.n_radius_samples as usize {
        let prev_radius_sample: f64 = t.radius_samples[i - 1];
        t.radius_samples[i] = prev_radius_sample * 1.2 as f64;
    }
    // choose albedo values of the diffusion profile discretization
    for i in 0..t.n_rho_samples as usize {
        t.rho_samples[i] = (1.0 as f64
            - (-8.0 as f64 * i as f64 / (t.n_rho_samples as f64 - 1.0 as f64)).exp())
            / (1.0 as f64 - (-8.0 as f64).exp());
    }
    // ParallelFor([&](int i) {
    for i in 0..t.n_rho_samples as usize {
        // compute the diffusion profile for the _i_th albedo sample

        // compute scattering profile for chosen albedo $\rho$
        for j in 0..t.n_radius_samples as usize {
            //         f64 rho = t.rho_samples[i], r = t.radius_samples[j];
            let rho: f64 = t.rho_samples[i];
            let r: f64 = t.radius_samples[j];
            t.profile[i * t.n_radius_samples as usize + j] = 2.0 as f64
                * PI
                * r
                * (beam_diffusion_ss(rho, 1.0 as f64 - rho, g, eta, r)
                    + beam_diffusion_ms(rho, 1.0 as f64 - rho, g, eta, r));
        }
        // compute effective albedo $\rho_{\roman{eff}}$ and CDF for
        // importance sampling
        t.rho_eff[i] = integrate_catmull_rom(
            t.n_radius_samples as i32,
            &t.radius_samples,
            i * t.n_radius_samples as usize,
            &t.profile,
            &mut t.profile_cdf,
        );
    }
    // }, t.n_rho_samples);
}
