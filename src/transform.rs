use crate::{
    geometry::{cross, faceforward, Bounds3f, Normal3f, Point3f, Ray, Vector3f},
    interaction::{BaseInteraction, SurfaceInteraction},
    misc::{copy_option_arc, radians},
};
use std::ops::Mul;

#[derive(Debug, Copy, Clone)]
pub struct Matrix4x4 {
    pub m: [[f64; 4]; 4],
}

impl Default for Matrix4x4 {
    fn default() -> Self {
        Matrix4x4 {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl Matrix4x4 {
    pub fn new(
        t00: f64,
        t01: f64,
        t02: f64,
        t03: f64,
        t10: f64,
        t11: f64,
        t12: f64,
        t13: f64,
        t20: f64,
        t21: f64,
        t22: f64,
        t23: f64,
        t30: f64,
        t31: f64,
        t32: f64,
        t33: f64,
    ) -> Self {
        Matrix4x4 {
            m: [
                [t00, t01, t02, t03],
                [t10, t11, t12, t13],
                [t20, t21, t22, t23],
                [t30, t31, t32, t33],
            ],
        }
    }
    pub fn transpose(m: &Matrix4x4) -> Matrix4x4 {
        Matrix4x4 {
            m: [
                [m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]],
                [m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]],
                [m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]],
                [m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]],
            ],
        }
    }
    pub fn inverse(m: &Matrix4x4) -> Matrix4x4 {
        let mut indxc = vec![0; 4];
        let mut indxr = vec![0; 4];
        let mut ipiv = vec![0; 4];
        let mut minv: Matrix4x4 = Matrix4x4::new(
            m.m[0][0], m.m[0][1], m.m[0][2], m.m[0][3], m.m[1][0], m.m[1][1], m.m[1][2], m.m[1][3],
            m.m[2][0], m.m[2][1], m.m[2][2], m.m[2][3], m.m[3][0], m.m[3][1], m.m[3][2], m.m[3][3],
        );
        for i in 0..4 {
            let mut irow = 0;
            let mut icol = 0;
            let mut big: f64 = 0.0;
            // choose pivot
            for j in 0..4 {
                if ipiv[j] != 1 {
                    for (k, item) in ipiv.iter().enumerate().take(4) {
                        if *item == 0 {
                            let abs: f64 = (minv.m[j][k]).abs();
                            if abs >= big {
                                big = abs;
                                irow = j;
                                icol = k;
                            }
                        } else if *item > 1 {
                            println!("Singular matrix in MatrixInvert");
                        }
                    }
                }
            }
            ipiv[icol] += 1;
            // swap rows _irow_ and _icol_ for pivot
            if irow != icol {
                for k in 0..4 {
                    // C++: std::swap(minv[irow][k], minv[icol][k]);
                    let swap = minv.m[irow][k];
                    minv.m[irow][k] = minv.m[icol][k];
                    minv.m[icol][k] = swap;
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if minv.m[icol][icol] == 0.0 {
                println!("Singular matrix in MatrixInvert");
            }
            // set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            let pivinv: f64 = 1.0 / minv.m[icol][icol];
            minv.m[icol][icol] = 1.0;
            for j in 0..4 {
                minv.m[icol][j] *= pivinv;
            }
            // subtract this row from others to zero out their columns
            for j in 0..4 {
                if j != icol {
                    let save: f64 = minv.m[j][icol];
                    minv.m[j][icol] = 0.0;
                    for k in 0..4 {
                        minv.m[j][k] -= minv.m[icol][k] * save;
                    }
                }
            }
        }
        // swap columns to reflect permutation
        for i in 0..4 {
            let j = 3 - i;
            if indxr[j] != indxc[j] {
                for k in 0..4 {
                    // C++: std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
                    minv.m[k].swap(indxr[j], indxc[j])
                }
            }
        }
        minv
    }
}

impl PartialEq for Matrix4x4 {
    fn eq(&self, rhs: &Matrix4x4) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if self.m[i][j] != rhs.m[i][j] {
                    return false;
                }
            }
        }
        true
    }
}

/// Finds the closed-form solution of a 2x2 linear system.
pub fn solve_linear_system_2x2(a: [[f64; 2]; 2], b: [f64; 2], x0: &mut f64, x1: &mut f64) -> bool {
    let det: f64 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-10 as f64 {
        return false;
    }
    *x0 = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
    *x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
    if (*x0).is_nan() || (*x1).is_nan() {
        return false;
    }
    true
}

/// The product of two matrices.
pub fn mtx_mul(m1: &Matrix4x4, m2: &Matrix4x4) -> Matrix4x4 {
    let mut r: Matrix4x4 = Matrix4x4::default();
    for i in 0..4 {
        for j in 0..4 {
            r.m[i][j] = m1.m[i][0] * m2.m[0][j]
                + m1.m[i][1] * m2.m[1][j]
                + m1.m[i][2] * m2.m[2][j]
                + m1.m[i][3] * m2.m[3][j];
        }
    }
    r
}

#[derive(Debug, Copy, Clone)]
pub struct Transform {
    pub m: Matrix4x4,
    pub m_inv: Matrix4x4,
}

impl Default for Transform {
    fn default() -> Self {
        Transform {
            m: Matrix4x4::default(),
            m_inv: Matrix4x4::default(),
        }
    }
}

impl Transform {
    pub fn new(
        t00: f64,
        t01: f64,
        t02: f64,
        t03: f64,
        t10: f64,
        t11: f64,
        t12: f64,
        t13: f64,
        t20: f64,
        t21: f64,
        t22: f64,
        t23: f64,
        t30: f64,
        t31: f64,
        t32: f64,
        t33: f64,
    ) -> Self {
        Transform {
            m: Matrix4x4::new(
                t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33,
            ),
            m_inv: Matrix4x4::inverse(&Matrix4x4::new(
                t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33,
            )),
        }
    }
    pub fn inverse(t: &Transform) -> Transform {
        Transform {
            m: t.m_inv,
            m_inv: t.m,
        }
    }
    pub fn is_identity(&self) -> bool {
        self.m.m[0][0] == 1.0 as f64
            && self.m.m[0][1] == 0.0 as f64
            && self.m.m[0][2] == 0.0 as f64
            && self.m.m[0][3] == 0.0 as f64
            && self.m.m[1][0] == 0.0 as f64
            && self.m.m[1][1] == 1.0 as f64
            && self.m.m[1][2] == 0.0 as f64
            && self.m.m[1][3] == 0.0 as f64
            && self.m.m[2][0] == 0.0 as f64
            && self.m.m[2][1] == 0.0 as f64
            && self.m.m[2][2] == 1.0 as f64
            && self.m.m[2][3] == 0.0 as f64
            && self.m.m[3][0] == 0.0 as f64
            && self.m.m[3][1] == 0.0 as f64
            && self.m.m[3][2] == 0.0 as f64
            && self.m.m[3][3] == 1.0 as f64
    }
    pub fn swaps_handedness(&self) -> bool {
        let det: f64 = self.m.m[0][0]
            * (self.m.m[1][1] * self.m.m[2][2] - self.m.m[1][2] * self.m.m[2][1])
            - self.m.m[0][1] * (self.m.m[1][0] * self.m.m[2][2] - self.m.m[1][2] * self.m.m[2][0])
            + self.m.m[0][2] * (self.m.m[1][0] * self.m.m[2][1] - self.m.m[1][1] * self.m.m[2][0]);
        det < 0.0 as f64
    }
    pub fn translate(delta: &Vector3f) -> Transform {
        Transform {
            m: Matrix4x4::new(
                1.0, 0.0, 0.0, delta.x, 0.0, 1.0, 0.0, delta.y, 0.0, 0.0, 1.0, delta.z, 0.0, 0.0,
                0.0, 1.0,
            ),
            m_inv: Matrix4x4::new(
                1.0, 0.0, 0.0, -delta.x, 0.0, 1.0, 0.0, -delta.y, 0.0, 0.0, 1.0, -delta.z, 0.0,
                0.0, 0.0, 1.0,
            ),
        }
    }
    pub fn scale(x: f64, y: f64, z: f64) -> Transform {
        Transform {
            m: Matrix4x4::new(
                x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
            m_inv: Matrix4x4::new(
                1.0 / x,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / y,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / z,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        }
    }
    pub fn rotate_x(theta: f64) -> Transform {
        let sin_theta: f64 = radians(theta).sin();
        let cos_theta: f64 = radians(theta).cos();
        let m = Matrix4x4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, cos_theta, -sin_theta, 0.0, 0.0, sin_theta, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        Transform {
            m,
            m_inv: Matrix4x4::transpose(&m),
        }
    }
    pub fn rotate_y(theta: f64) -> Transform {
        let sin_theta: f64 = radians(theta).sin();
        let cos_theta: f64 = radians(theta).cos();
        let m = Matrix4x4::new(
            cos_theta, 0.0, sin_theta, 0.0, 0.0, 1.0, 0.0, 0.0, -sin_theta, 0.0, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        Transform {
            m,
            m_inv: Matrix4x4::transpose(&m),
        }
    }
    pub fn rotate_z(theta: f64) -> Transform {
        let sin_theta: f64 = radians(theta).sin();
        let cos_theta: f64 = radians(theta).cos();
        let m = Matrix4x4::new(
            cos_theta, -sin_theta, 0.0, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        Transform {
            m,
            m_inv: Matrix4x4::transpose(&m),
        }
    }
    pub fn rotate(theta: f64, axis: &Vector3f) -> Transform {
        let a: Vector3f = axis.normalize();
        let sin_theta: f64 = radians(theta).sin();
        let cos_theta: f64 = radians(theta).cos();
        let mut m = Matrix4x4::default();
        // compute rotation of first basis vector
        m.m[0][0] = a.x * a.x + (1.0 - a.x * a.x) * cos_theta;
        m.m[0][1] = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
        m.m[0][2] = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;
        m.m[0][3] = 0.0;
        // compute rotations of second basis vectors
        m.m[1][0] = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
        m.m[1][1] = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
        m.m[1][2] = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;
        m.m[1][3] = 0.0;
        // compute rotations of third basis vectors
        m.m[2][0] = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
        m.m[2][1] = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
        m.m[2][2] = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;
        m.m[2][3] = 0.0;
        Transform {
            m,
            m_inv: Matrix4x4::transpose(&m),
        }
    }
    pub fn look_at(pos: &Point3f, look: &Point3f, up: &Vector3f) -> Transform {
        let mut camera_to_world = Matrix4x4::default();
        // initialize fourth column of viewing matrix
        camera_to_world.m[0][3] = pos.x;
        camera_to_world.m[1][3] = pos.y;
        camera_to_world.m[2][3] = pos.z;
        camera_to_world.m[3][3] = 1.0;
        // initialize first three columns of viewing matrix
        let dir: Vector3f = (*look - *pos).normalize();
        if cross(&up.normalize(), &dir).length() == 0.0 {
            println!(
                "\"up\" vector ({}, {}, {}) and viewing direction ({}, {}, {}) passed to \
                 LookAt are pointing in the same direction.  Using the identity \
                 transformation.",
                up.x, up.y, up.z, dir.x, dir.y, dir.z
            );
            Transform::default()
        } else {
            let left: Vector3f = cross(&up.normalize(), &dir).normalize();
            let new_up: Vector3f = cross(&dir, &left);
            camera_to_world.m[0][0] = left.x;
            camera_to_world.m[1][0] = left.y;
            camera_to_world.m[2][0] = left.z;
            camera_to_world.m[3][0] = 0.0;
            camera_to_world.m[0][1] = new_up.x;
            camera_to_world.m[1][1] = new_up.y;
            camera_to_world.m[2][1] = new_up.z;
            camera_to_world.m[3][1] = 0.0;
            camera_to_world.m[0][2] = dir.x;
            camera_to_world.m[1][2] = dir.y;
            camera_to_world.m[2][2] = dir.z;
            camera_to_world.m[3][2] = 0.0;
            Transform {
                m: Matrix4x4::inverse(&camera_to_world),
                m_inv: camera_to_world,
            }
        }
    }
    pub fn orthographic(z_near: f64, z_far: f64) -> Transform {
        let translate: Transform = Transform::translate(&Vector3f {
            x: 0.0,
            y: 0.0,
            z: -z_near,
        });
        let scale: Transform = Transform::scale(1.0, 1.0, 1.0 / (z_far - z_near));
        scale * translate
    }
    pub fn perspective(fov: f64, n: f64, f: f64) -> Transform {
        // perform projective divide for perspective projection
        let persp = Matrix4x4::new(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f / (f - n),
            -f * n / (f - n),
            0.0,
            0.0,
            1.0,
            0.0,
        );
        // scale canonical perspective view to specified field of view
        // Here the adjacent side has length 1, so the opposite side has the length tan(fov/2).
        // Scaling by the reciprocal of this length maps the field of view to range from [−1, 1].
        let inv_tan_ang: f64 = 1.0 / (radians(fov) / 2.0).tan();
        let scale: Transform = Transform::scale(inv_tan_ang, inv_tan_ang, 1.0);
        let persp_trans: Transform = Transform {
            m: persp,
            m_inv: Matrix4x4::inverse(&persp),
        };
        scale * persp_trans
    }
    pub fn t<T: Transformable>(&self, obj: &T) -> T {
        obj.t_by(self)
    }
}

impl PartialEq for Transform {
    fn eq(&self, rhs: &Transform) -> bool {
        rhs.m == self.m && rhs.m_inv == self.m_inv
    }
}

impl Mul for Transform {
    type Output = Transform;
    fn mul(self, rhs: Transform) -> Transform {
        Transform {
            m: mtx_mul(&self.m, &rhs.m),
            m_inv: mtx_mul(&rhs.m_inv, &self.m_inv),
        }
    }
}

pub trait Transformable {
    fn t_by(&self, transform: &Transform) -> Self;
}

impl Transformable for Point3f {
    fn t_by(&self, transform: &Transform) -> Self {
        let x: f64 = self.x;
        let y: f64 = self.y;
        let z: f64 = self.z;
        let xp: f64 = transform.m.m[0][0] * x
            + transform.m.m[0][1] * y
            + transform.m.m[0][2] * z
            + transform.m.m[0][3];
        let yp: f64 = transform.m.m[1][0] * x
            + transform.m.m[1][1] * y
            + transform.m.m[1][2] * z
            + transform.m.m[1][3];
        let zp: f64 = transform.m.m[2][0] * x
            + transform.m.m[2][1] * y
            + transform.m.m[2][2] * z
            + transform.m.m[2][3];
        let wp: f64 = transform.m.m[3][0] * x
            + transform.m.m[3][1] * y
            + transform.m.m[3][2] * z
            + transform.m.m[3][3];
        assert!(wp != 0.0, "wp = {:?} != 0.0", wp);
        if wp == 1.0 as f64 {
            Point3f {
                x: xp,
                y: yp,
                z: zp,
            }
        } else {
            let inv: f64 = 1.0 as f64 / wp;
            Point3f {
                x: inv * xp,
                y: inv * yp,
                z: inv * zp,
            }
        }
    }
}

impl Transformable for Vector3f {
    fn t_by(&self, transform: &Transform) -> Self {
        let x: f64 = self.x;
        let y: f64 = self.y;
        let z: f64 = self.z;
        Vector3f {
            x: transform.m.m[0][0] * x + transform.m.m[0][1] * y + transform.m.m[0][2] * z,
            y: transform.m.m[1][0] * x + transform.m.m[1][1] * y + transform.m.m[1][2] * z,
            z: transform.m.m[2][0] * x + transform.m.m[2][1] * y + transform.m.m[2][2] * z,
        }
    }
}

impl Transformable for Normal3f {
    fn t_by(&self, transform: &Transform) -> Self {
        let x: f64 = self.x;
        let y: f64 = self.y;
        let z: f64 = self.z;
        Normal3f {
            x: transform.m_inv.m[0][0] * x
                + transform.m_inv.m[1][0] * y
                + transform.m_inv.m[2][0] * z,
            y: transform.m_inv.m[0][1] * x
                + transform.m_inv.m[1][1] * y
                + transform.m_inv.m[2][1] * z,
            z: transform.m_inv.m[0][2] * x
                + transform.m_inv.m[1][2] * y
                + transform.m_inv.m[2][2] * z,
        }
    }
}
impl Transformable for Ray {
    fn t_by(&self, transform: &Transform) -> Self {
        let o = self.o.t_by(transform);
        let d = self.d.t_by(transform);

        Ray::new(
            o,
            d.normalize(),
            self.t_max,
            self.time,
            copy_option_arc(&self.medium),
        )
    }
}
impl Transformable for Bounds3f {
    fn t_by(&self, transform: &Transform) -> Self {
        let p: Point3f = (Point3f {
            x: self.p_min.x,
            y: self.p_min.y,
            z: self.p_min.z,
        })
        .t_by(transform);
        let mut ret: Bounds3f = Bounds3f { p_min: p, p_max: p };
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_max.x,
                y: self.p_min.y,
                z: self.p_min.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_min.x,
                y: self.p_max.y,
                z: self.p_min.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_min.x,
                y: self.p_min.y,
                z: self.p_max.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_min.x,
                y: self.p_max.y,
                z: self.p_max.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_max.x,
                y: self.p_max.y,
                z: self.p_min.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_max.x,
                y: self.p_min.y,
                z: self.p_max.z,
            })
            .t_by(transform),
        );
        ret = Bounds3f::union(
            &ret,
            &(Point3f {
                x: self.p_max.x,
                y: self.p_max.y,
                z: self.p_max.z,
            })
            .t_by(transform),
        );
        ret
    }
}

impl Transformable for BaseInteraction {
    fn t_by(&self, transform: &Transform) -> Self {
        BaseInteraction::new(
            self.p.t_by(transform),
            self.time,
            Vector3f::default(),
            self.wo.t_by(transform),
            self.n.t_by(transform),
            self.mi.clone(),
        )
    }
}

impl Transformable for SurfaceInteraction {
    fn t_by(&self, transform: &Transform) -> Self {
        let ist = self.ist.t_by(transform);
        let mut r_si = SurfaceInteraction::default();
        r_si.ist = ist;
        r_si.uv = self.uv;
        r_si.shape = copy_option_arc(&self.shape);
        r_si.dpdu = self.dpdu.t_by(transform);
        r_si.dpdv = self.dpdv.t_by(transform);
        r_si.dndu = self.dndu.t_by(transform);
        r_si.dndv = self.dndv.t_by(transform);
        r_si.shading.n = self.shading.n.t_by(transform).normalize();
        r_si.shading.dpdu = self.shading.dpdu.t_by(transform);
        r_si.shading.dpdv = self.shading.dpdv.t_by(transform);
        r_si.shading.dndu = self.shading.dndu.t_by(transform);
        r_si.shading.dndv = self.shading.dndv.t_by(transform);
        r_si.dudx = self.dudx;
        r_si.dvdx = self.dvdx;
        r_si.dudy = self.dudy;
        r_si.dvdy = self.dvdy;
        r_si.dpdx = self.dpdx;
        r_si.dpdy = self.dpdy;
        r_si.bsdf = self.bsdf.clone();
        r_si.bssrdf = self.bssrdf.clone();
        r_si.primitive = copy_option_arc(&self.primitive);
        r_si.shading.n = faceforward(&r_si.shading.n, &r_si.ist.n);
        r_si
    }
}

pub trait ToWorld {
    fn to_world(&self) -> &Transform;
}

pub trait ToLocal {
    fn to_local(&self) -> &Transform;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generic_transform() {
        let test_tr = Transform::default();
        let p1 = Point3f::default();
        let _tp = p1.t_by(&test_tr);
        let _tp1 = test_tr.t(&p1);
    }
}
