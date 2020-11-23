use std::f64::consts::PI;

use crate::{
    geometry::{dot3, spherical_phi, spherical_theta, Point2f, Point3f, Vector2f, Vector3f},
    interaction::SurfaceInteraction,
    misc::{clamp_t, lerp},
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
    transform::Transform,
};

// Perlin Noise Data
pub const NOISE_PERM_SIZE: usize = 256;
pub const NOISE_PERM: [u8; 2 * NOISE_PERM_SIZE] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, // remainder of the noise permutation table
    8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203,
    117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74,
    165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
    105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132,
    187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3,
    64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59,
    227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
    221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194,
    233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
    159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253,
    19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];

pub trait Texture<T>: std::fmt::Debug {
    fn evaluate(&self, si: &SurfaceInteraction) -> T;
}

#[derive(Debug)]
pub struct ConstantTexture<T: Copy> {
    v: T,
}

impl<T: Copy> ConstantTexture<T> {
    pub fn new(v: T) -> Self {
        Self { v }
    }
}

impl Texture<f64> for ConstantTexture<f64> {
    fn evaluate(&self, _si: &SurfaceInteraction) -> f64 {
        self.v
    }
}

impl Texture<Spectrum<SPECTRUM_N>> for ConstantTexture<Spectrum<SPECTRUM_N>> {
    fn evaluate(&self, _si: &SurfaceInteraction) -> Spectrum<SPECTRUM_N> {
        self.v
    }
}

pub fn smooth_step(min: f64, max: f64, value: f64) -> f64 {
    let v: f64 = clamp_t((value - min) / (max - min), 0.0, 1.0);
    v * v * (-2.0 * v + 3.0)
}

pub fn noise_flt(x: f64, y: f64, z: f64) -> f64 {
    // compute noise cell coordinates and offsets
    let mut ix: i32 = x.floor() as i32;
    let mut iy: i32 = y.floor() as i32;
    let mut iz: i32 = z.floor() as i32;
    let dx: f64 = x - ix as f64;
    let dy: f64 = y - iy as f64;
    let dz: f64 = z - iz as f64;
    // compute gradient weights
    ix &= NOISE_PERM_SIZE as i32 - 1;
    iy &= NOISE_PERM_SIZE as i32 - 1;
    iz &= NOISE_PERM_SIZE as i32 - 1;
    let w000: f64 = grad(ix, iy, iz, dx, dy, dz);
    let w100: f64 = grad(ix + 1, iy, iz, dx - 1.0, dy, dz);
    let w010: f64 = grad(ix, iy + 1, iz, dx, dy - 1.0, dz);
    let w110: f64 = grad(ix + 1, iy + 1, iz, dx - 1.0, dy - 1.0, dz);
    let w001: f64 = grad(ix, iy, iz + 1, dx, dy, dz - 1.0);
    let w101: f64 = grad(ix + 1, iy, iz + 1, dx - 1.0, dy, dz - 1.0);
    let w011: f64 = grad(ix, iy + 1, iz + 1, dx, dy - 1.0, dz - 1.0);
    let w111: f64 = grad(ix + 1, iy + 1, iz + 1, dx - 1.0, dy - 1.0, dz - 1.0);
    // compute trilinear interpolation of weights
    let wx: f64 = noise_weight(dx);
    let wy: f64 = noise_weight(dy);
    let wz: f64 = noise_weight(dz);
    let x00: f64 = lerp(wx, w000, w100);
    let x10: f64 = lerp(wx, w010, w110);
    let x01: f64 = lerp(wx, w001, w101);
    let x11: f64 = lerp(wx, w011, w111);
    let y0: f64 = lerp(wy, x00, x10);
    let y1: f64 = lerp(wy, x01, x11);
    let ret: f64 = lerp(wz, y0, y1);
    ret
}

pub fn noise_pnt3(p: &Point3f) -> f64 {
    noise_flt(p.x, p.y, p.z)
}

pub fn grad(x: i32, y: i32, z: i32, dx: f64, dy: f64, dz: f64) -> f64 {
    let mut h: u8 =
        NOISE_PERM[NOISE_PERM[NOISE_PERM[x as usize] as usize + y as usize] as usize + z as usize];
    h &= 15_u8;
    let u = if h < 8_u8 || h == 12_u8 || h == 13_u8 {
        dx
    } else {
        dy
    };
    let v = if h < 4_u8 || h == 12_u8 || h == 13_u8 {
        dy
    } else {
        dz
    };
    let ret_u = if h & 1_u8 > 0_u8 { -u } else { u };
    let ret_v = if h & 2_u8 > 0_u8 { -v } else { v };
    ret_u + ret_v
}

pub fn noise_weight(t: f64) -> f64 {
    let t3: f64 = t * t * t;
    let t4: f64 = t3 * t;
    6.0 * t4 * t - 15.0 * t4 + 10.0 * t3
}

pub fn fbm(p: &Point3f, dpdx: &Vector3f, dpdy: &Vector3f, omega: f64, max_octaves: i32) -> f64 {
    // compute number of octaves for antialiased FBm
    let len2: f64 = dpdx.length_squared().max(dpdy.length_squared());
    let n: f64 = clamp_t(-1.0 - 0.5 * len2.log2(), 0.0, max_octaves as f64);
    let n_int: i32 = n.floor() as i32;
    // compute sum of octaves of noise for FBm
    let mut sum: f64 = 0.0;
    let mut lambda: f64 = 1.0;
    let mut o: f64 = 1.0;
    for _i in 0..n_int {
        sum += o * noise_pnt3(&(*p * lambda));
        lambda *= 1.99;
        o *= omega;
    }
    let n_partial: f64 = n - n_int as f64;
    sum += o * smooth_step(0.3, 0.7, n_partial) * noise_pnt3(&(*p * lambda));
    sum
}

pub fn turbulence(
    p: &Point3f,
    dpdx: &Vector3f,
    dpdy: &Vector3f,
    omega: f64,
    max_octaves: i32,
) -> f64 {
    // compute number of octaves for antialiased FBm
    let len2: f64 = dpdx.length_squared().max(dpdy.length_squared());
    let n: f64 = clamp_t(-1.0 - 0.5 * (len2.log2()), 0.0, max_octaves as f64);
    let n_int: usize = n.floor() as usize;
    // compute sum of octaves of noise for turbulence
    let mut sum: f64 = 0.0;
    let mut lambda: f64 = 1.0;
    let mut o: f64 = 1.0;
    for _i in 0..n_int {
        sum += o * noise_pnt3(&(*p * lambda)).abs();
        lambda *= 1.99;
        o *= omega;
    }
    // account for contributions of clamped octaves in turbulence
    let n_partial: f64 = n - n_int as f64;
    sum += o * lerp(
        smooth_step(0.3, 0.7, n_partial),
        0.2,
        noise_pnt3(&(*p * lambda)).abs(),
    );
    for _i in n_int..max_octaves as usize {
        sum += o * 0.2;
        o *= omega;
    }
    sum
}

pub fn lanczos(x: f64, tau: f64) -> f64 {
    let mut x: f64 = x;
    x = x.abs();
    if x < 1e-5 {
        return 1.0;
    }
    if x > 1.0 {
        return 0.0;
    }
    x *= PI;
    let s: f64 = (x * tau).sin() / (x * tau);
    let lanczos: f64 = x.sin() / x;
    s * lanczos
}

pub trait TextureMapping2D: std::fmt::Debug {
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f;
}

#[derive(Debug, Copy, Clone)]
pub struct UVMapping2D {
    su: f64,
    sv: f64,
    du: f64,
    dv: f64,
}

impl UVMapping2D {
    pub fn new(su: f64, sv: f64, du: f64, dv: f64) -> Self {
        Self { su, sv, du, dv }
    }
}

impl Default for UVMapping2D {
    fn default() -> Self {
        Self {
            su: 1.0,
            sv: 1.0,
            du: 0.0,
            dv: 0.0,
        }
    }
}

impl TextureMapping2D for UVMapping2D {
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        // Compute texture differentials for 2D identity mapping
        *dstdx = Vector2f::new(self.su * si.dudx, self.sv * si.dvdx);
        *dstdy = Vector2f::new(self.su * si.dudy, self.sv * si.dvdy);

        Point2f::new(self.su * si.uv[0] + self.du, self.sv * si.uv[1] + self.dv)
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct SphericalMapping2D {
    world_to_texture: Transform,
}

impl SphericalMapping2D {
    pub fn new(world_to_texture: Transform) -> Self {
        Self { world_to_texture }
    }
    fn sphere(&self, p: &Point3f) -> Point2f {
        let v = (self.world_to_texture.transform_point(p) - Point3f::zero()).normalize();
        let theta = spherical_theta(&v);
        let phi = spherical_phi(&v);
        Point2f::new(theta / PI, phi / (PI * 2.0))
    }
}

impl TextureMapping2D for SphericalMapping2D {
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        let st = self.sphere(&si.ist.p);
        // Compute texture coordinate differentials for sphere $(u,v)$ mapping
        let delta = 0.1;
        let st_delta_x = self.sphere(&(si.ist.p + si.dpdx * delta));
        *dstdx = (st_delta_x - st) / delta;
        let st_delta_y = self.sphere(&(si.ist.p + si.dpdy * delta));
        *dstdy = (st_delta_y - st) / delta;
        // Handle sphere mapping discontinuity for coordinate differentials
        if dstdx[1] > 0.5 {
            dstdx[1] = 1.0 - dstdx[1];
        } else if dstdx[1] < -0.5 {
            dstdx[1] = -(dstdx[1] + 1.0);
        }
        if dstdy[1] > 0.5 {
            dstdy[1] = 1.0 - dstdy[1];
        } else if dstdy[1] < -0.5 {
            dstdy[1] = -(dstdy[1] + 1.0);
        }
        st
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct CylindricalMapping2D {
    world_to_texture: Transform,
}

impl CylindricalMapping2D {
    pub fn new(world_to_texture: Transform) -> Self {
        Self { world_to_texture }
    }
    fn cylinder(&self, p: &Point3f) -> Point2f {
        let v = (self.world_to_texture.transform_point(p) - Point3f::zero()).normalize();
        Point2f::new((PI + f64::atan2(v.y, v.x)) / (2.0 * PI), v.z)
    }
}

impl TextureMapping2D for CylindricalMapping2D {
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        let st = self.cylinder(&si.ist.p);
        // Compute texture coordinate differentials for cylinder $(u,v)$ mapping
        let delta = 0.1;
        let st_delta_x = self.cylinder(&(si.ist.p + si.dpdx * delta));
        *dstdx = (st_delta_x - st) / delta;
        let st_delta_y = self.cylinder(&(si.ist.p + si.dpdy * delta));
        *dstdy = (st_delta_y - st) / delta;
        if dstdx[1] > 0.5 {
            dstdx[1] = 1.0 - dstdx[1];
        } else if dstdx[1] < -0.5 {
            dstdx[1] = -(dstdx[1] + 1.0);
        }
        if dstdy[1] > 0.5 {
            dstdy[1] = 1.0 - dstdy[1];
        } else if dstdy[1] < -0.5 {
            dstdy[1] = -(dstdy[1] + 1.0);
        }
        st
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct PlanarMapping2D {
    vs: Vector3f,
    vt: Vector3f,
    ds: f64,
    dt: f64,
}

impl PlanarMapping2D {
    pub fn new(vs: Vector3f, vt: Vector3f, ds: f64, dt: f64) -> Self {
        Self { vs, vt, ds, dt }
    }
}

impl TextureMapping2D for PlanarMapping2D {
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        // Vector3f vec(si.p);
        let v = Vector3f::from(si.ist.p);
        *dstdx = Vector2f::new(dot3(&si.dpdx, &self.vs), dot3(&si.dpdx, &self.vt));
        *dstdy = Vector2f::new(dot3(&si.dpdy, &self.vs), dot3(&si.dpdy, &self.vt));
        Point2f::new(self.ds + dot3(&v, &self.vs), self.dt + dot3(&v, &self.vt))
    }
}

pub trait TextureMapping3D {
    fn map(&self, si: &SurfaceInteraction, dpdx: &mut Vector3f, dpdy: &mut Vector3f) -> Point3f;
}

#[derive(Default, Debug, Copy, Clone)]
pub struct IdentityMapping3D {
    world_to_texture: Transform,
}

impl IdentityMapping3D {
    pub fn new(world_to_texture: Transform) -> Self {
        Self { world_to_texture }
    }
}

impl TextureMapping3D for IdentityMapping3D {
    fn map(&self, si: &SurfaceInteraction, dpdx: &mut Vector3f, dpdy: &mut Vector3f) -> Point3f {
        *dpdx = self.world_to_texture.transform_vector(&si.dpdx);
        *dpdy = self.world_to_texture.transform_vector(&si.dpdy);
        self.world_to_texture.transform_point(&si.ist.p)
    }
}

pub mod mix;
pub mod scale;
pub mod uv;
