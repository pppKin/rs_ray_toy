use crate::{
    bssrdf::BSSRDF,
    geometry::{
        cross, dot3, faceforward, pnt3_offset_ray_origin, Normal3f, Point2f, Point3f, Ray,
        RayDifferential, Vector3f,
    },
    material::TransportMode,
    medium::{MediumInterface, MediumOpArc, PhaseFunction},
    misc::{copy_option_arc, SHADOW_EPSILON},
    primitives::GeometricPrimitive,
    reflection::Bsdf,
    shape::Shape,
    spectrum::Spectrum,
    transform::solve_linear_system_2x2,
    SPECTRUM_N,
};

use std::sync::Arc;

#[derive(Default, Debug, Clone)]
pub struct BaseInteraction {
    pub p: Point3f,
    pub time: f64,
    pub p_error: Vector3f,
    pub wo: Vector3f,
    pub n: Normal3f,
    pub mi: Option<MediumInterface>,
}

impl BaseInteraction {
    pub fn new(
        p: Point3f,
        time: f64,
        p_error: Vector3f,
        wo: Vector3f,
        n: Normal3f,
        mi: Option<MediumInterface>,
    ) -> Self {
        Self {
            p,
            time,
            p_error,
            wo,
            n,
            mi,
        }
    }
    pub fn get_medium(&self, w: &Vector3f) -> MediumOpArc {
        match &self.mi {
            Some(mif) => {
                if dot3(&self.n, w) > 0.0 {
                    mif.outside.clone()
                } else {
                    mif.inside.clone()
                }
            }
            None => None,
        }
    }
    pub fn spawn_ray(&self, d: Vector3f) -> Ray {
        Ray::new_od(self.p, d)
    }
    pub fn spawn_ray_to(&self, p2: Point3f) -> Ray {
        self.spawn_ray(p2 - self.p)
    }
    pub fn spawn_ray_to_si(&self, ist: &BaseInteraction) -> Ray {
        let origin = pnt3_offset_ray_origin(&self.p, &self.p_error, &self.n, &(ist.p - self.p));
        let target = pnt3_offset_ray_origin(&ist.p, &ist.p_error, &ist.n, &(origin - ist.p));
        let d = target - origin;
        Ray::new(
            origin,
            d,
            1.0 - SHADOW_EPSILON,
            self.time,
            self.get_medium(&d),
        )
    }
}

pub enum Interaction {
    Surface(SurfaceInteraction),
    Medium(MediumInteraction),
}

#[derive(Clone, Copy, Debug)]
pub struct Shading {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Debug, Clone)]
pub struct SurfaceInteraction {
    pub ist: BaseInteraction,
    pub uv: Point2f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shape: Option<Arc<dyn Shape>>,
    pub shading: Shading,
    pub primitive: Option<Arc<GeometricPrimitive>>,
    pub bsdf: Option<Bsdf>,
    pub bssrdf: Option<Arc<dyn BSSRDF>>,
    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
    pub dudx: f64,
    pub dvdx: f64,
    pub dudy: f64,
    pub dvdy: f64,
}

impl Default for SurfaceInteraction {
    fn default() -> Self {
        SurfaceInteraction::new(
            Point3f::default(),
            Point2f::default(),
            Vector3f::default(),
            Vector3f::default(),
            Vector3f::default(),
            Normal3f::default(),
            Normal3f::default(),
            0.0,
        )
    }
}

impl SurfaceInteraction {
    pub fn new(
        p: Point3f,
        uv: Point2f,
        wo: Vector3f,
        dpdu: Vector3f,
        dpdv: Vector3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: f64,
    ) -> Self {
        let nv: Vector3f = cross(&dpdu, &dpdv).normalize();
        let n: Normal3f = Normal3f {
            x: nv.x,
            y: nv.y,
            z: nv.z,
        };
        let shading = Shading {
            n,
            dpdu,
            dpdv,
            dndu,
            dndv,
        };
        let p_error = Vector3f::default();
        let ist = BaseInteraction {
            p,
            time,
            p_error,
            wo,
            n,
            mi: None,
        };
        SurfaceInteraction {
            ist,
            uv,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shape: None,
            shading,
            primitive: None,
            bsdf: None,
            bssrdf: None,
            dpdx: Vector3f::default(),
            dpdy: Vector3f::default(),
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }
    pub fn set_shading_geometry(
        &mut self,
        dpdus: &Vector3f,
        dpdvs: &Vector3f,
        dndus: &Normal3f,
        dndvs: &Normal3f,
        orientation_is_authoritative: bool,
    ) {
        let mut n = Normal3f::from(cross(dpdus, dpdvs).normalize());
        if orientation_is_authoritative {
            n = faceforward(&self.ist.n, &n);
        } else {
            n = faceforward(&n, &self.ist.n);
        }
        self.shading.n = n;
        self.shading.dpdu = *dpdus;
        self.shading.dpdv = *dpdvs;
        self.shading.dndu = *dndus;
        self.shading.dndv = *dndvs;
    }
    pub fn compute_scattering_functions(
        &mut self,
        ray: &RayDifferential,
        allow_multiple_lobes: bool,
        mode: TransportMode,
    ) {
        self.compute_differentials(ray);
        let opt_pri = copy_option_arc(&self.primitive);
        if let Some(pri) = opt_pri {
            pri.compute_scattering_functions(self, mode, allow_multiple_lobes);
        }
    }
    fn empty_differentials(&mut self) {
        self.dudx = 0.0;
        self.dvdx = 0.0;
        self.dudy = 0.0;
        self.dvdy = 0.0;
        self.dpdx = Vector3f::new(0.0, 0.0, 0.0);
        self.dpdy = Vector3f::new(0.0, 0.0, 0.0);
    }
    pub fn compute_differentials(&mut self, ray: &RayDifferential) {
        if ray.has_differentials {
            // Estimate screen space change in $\pt{}$ and $(u,v)$
            // Compute auxiliary intersection points with plane
            let d = dot3(&self.ist.n, &self.ist.p);
            let tx =
                -(dot3(&self.ist.n, &(ray.rx_origin)) - d) / dot3(&self.ist.n, &ray.rx_direction);
            if tx.is_infinite() || tx.is_nan() {
                return self.empty_differentials();
            }
            let px = ray.rx_origin + ray.rx_direction * tx;
            let ty =
                -(dot3(&self.ist.n, &ray.ry_direction) - d) / dot3(&self.ist.n, &ray.ry_direction);
            if ty.is_infinite() || ty.is_nan() {
                return self.empty_differentials();
            }
            let py = ray.ry_origin + ray.ry_direction * ty;
            self.dpdx = px - self.ist.p;

            self.dpdy = py - self.ist.p;

            // Compute $(u,v)$ offsets at auxiliary points

            // Choose two dimensions to use for ray offset computation
            let mut dim: [u8; 2] = [0, 0];
            if self.ist.n.x.abs() > self.ist.n.y.abs() && self.ist.n.x.abs() > self.ist.n.z.abs() {
                dim[0] = 1;
                dim[1] = 2;
            } else if self.ist.n.y.abs() > self.ist.n.z.abs() {
                dim[0] = 0;
                dim[1] = 2;
            } else {
                dim[0] = 0;
                dim[1] = 1;
            }

            // Initialize _A_, _Bx_, and _By_ matrices for offset computation
            let a = [
                [self.dpdu[dim[0]], self.dpdv[dim[0]]],
                [self.dpdu[dim[1]], self.dpdv[dim[1]]],
            ];
            let bx = [
                px[dim[0]] - self.ist.p[dim[0]],
                px[dim[1]] - self.ist.p[dim[1]],
            ];
            let by = [
                py[dim[0]] - self.ist.p[dim[0]],
                py[dim[1]] - self.ist.p[dim[1]],
            ];

            if !solve_linear_system_2x2(a, bx, &mut self.dudx, &mut self.dvdx) {
                self.dudx = 0.0;
                self.dvdx = 0.0;
            }
            if !solve_linear_system_2x2(a, by, &mut self.dudy, &mut self.dvdy) {
                self.dudy = 0.0;
                self.dvdy = 0.0;
            }
        } else {
            self.empty_differentials()
        }
    }
    pub fn le(&self, w: &Vector3f) -> Spectrum<SPECTRUM_N> {
        if let Some(pri) = &self.primitive {
            if let Some(area) = pri.get_arealight() {
                return area.l(&self.ist, w);
            }
        }
        Spectrum::zero()
    }
}

#[derive(Debug, Default, Clone)]
pub struct MediumInteraction {
    pub ist: BaseInteraction,
    pub phase: Option<Arc<dyn PhaseFunction>>,
}

impl MediumInteraction {
    pub fn is_valid(&self) -> bool {
        self.phase.is_some()
    }
}

impl MediumInteraction {
    pub fn new(ist: BaseInteraction, phase: Option<Arc<dyn PhaseFunction>>) -> Self {
        Self { ist, phase }
    }
}
