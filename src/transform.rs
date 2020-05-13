use crate::geometry::{
    bnd3_union_pnt3, vec3_cross_vec3, vec3_dot_vec3, Bounds3f, Normal3f, Point3f, Ray, Vector3f,
};
use crate::misc::{gamma, radians};
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
        if vec3_cross_vec3(&up.normalize(), &dir).length() == 0.0 {
            println!(
                "\"up\" vector ({}, {}, {}) and viewing direction ({}, {}, {}) passed to \
                 LookAt are pointing in the same direction.  Using the identity \
                 transformation.",
                up.x, up.y, up.z, dir.x, dir.y, dir.z
            );
            Transform::default()
        } else {
            let left: Vector3f = vec3_cross_vec3(&up.normalize(), &dir).normalize();
            let new_up: Vector3f = vec3_cross_vec3(&dir, &left);
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
    pub fn transform_point(&self, p: &Point3f) -> Point3f {
        let x: f64 = p.x;
        let y: f64 = p.y;
        let z: f64 = p.z;
        let xp: f64 = self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z + self.m.m[0][3];
        let yp: f64 = self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z + self.m.m[1][3];
        let zp: f64 = self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z + self.m.m[2][3];
        let wp: f64 = self.m.m[3][0] * x + self.m.m[3][1] * y + self.m.m[3][2] * z + self.m.m[3][3];
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
    pub fn transform_vector(&self, v: &Vector3f) -> Vector3f {
        let x: f64 = v.x;
        let y: f64 = v.y;
        let z: f64 = v.z;
        Vector3f {
            x: self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z,
            y: self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z,
            z: self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z,
        }
    }
    pub fn transform_normal(&self, n: &Normal3f) -> Normal3f {
        let x: f64 = n.x;
        let y: f64 = n.y;
        let z: f64 = n.z;
        Normal3f {
            x: self.m_inv.m[0][0] * x + self.m_inv.m[1][0] * y + self.m_inv.m[2][0] * z,
            y: self.m_inv.m[0][1] * x + self.m_inv.m[1][1] * y + self.m_inv.m[2][1] * z,
            z: self.m_inv.m[0][2] * x + self.m_inv.m[1][2] * y + self.m_inv.m[2][2] * z,
        }
    }
    pub fn transform_ray(&self, r: &Ray) -> Ray {
        let mut o_error: Vector3f = Vector3f::default();
        let mut o: Point3f = self.transform_point_with_error(&r.origin, &mut o_error);
        let d: Vector3f = self.transform_vector(&r.direction);
        // println!("transform ray :: {:?}", d);
        // offset ray origin to edge of error bounds and compute _tMax_
        let length_squared: f64 = d.length_squared();
        let mut t_max: f64 = r.inter_dist;
        if length_squared > 0.0 as f64 {
            let dt: f64 = vec3_dot_vec3(&d.abs(), &o_error) / length_squared;
            o += d * dt;
            t_max -= dt;
        }
        Ray {
            origin: o,
            direction: d.normalize(),
            inter_dist: t_max,
            inter_obj: -1,
        }
    }
    pub fn transform_bounds(&self, b: &Bounds3f) -> Bounds3f {
        let m: Transform = *self;
        let p: Point3f = self.transform_point(&Point3f {
            x: b.p_min.x,
            y: b.p_min.y,
            z: b.p_min.z,
        });
        let mut ret: Bounds3f = Bounds3f { p_min: p, p_max: p };
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_max.x,
                y: b.p_min.y,
                z: b.p_min.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_min.x,
                y: b.p_max.y,
                z: b.p_min.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_min.x,
                y: b.p_min.y,
                z: b.p_max.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_min.x,
                y: b.p_max.y,
                z: b.p_max.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_max.x,
                y: b.p_max.y,
                z: b.p_min.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_max.x,
                y: b.p_min.y,
                z: b.p_max.z,
            }),
        );
        ret = bnd3_union_pnt3(
            &ret,
            &m.transform_point(&Point3f {
                x: b.p_max.x,
                y: b.p_max.y,
                z: b.p_max.z,
            }),
        );
        ret
    }
    pub fn transform_point_with_error(&self, p: &Point3f, p_error: &mut Vector3f) -> Point3f {
        let x: f64 = p.x;
        let y: f64 = p.y;
        let z: f64 = p.z;
        // compute transformed coordinates from point _pt_
        let xp: f64 = self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z + self.m.m[0][3];
        let yp: f64 = self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z + self.m.m[1][3];
        let zp: f64 = self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z + self.m.m[2][3];
        let wp: f64 = self.m.m[3][0] * x + self.m.m[3][1] * y + self.m.m[3][2] * z + self.m.m[3][3];
        // compute absolute error for transformed point
        let x_abs_sum: f64 = (self.m.m[0][0] * x).abs()
            + (self.m.m[0][1] * y).abs()
            + (self.m.m[0][2] * z).abs()
            + self.m.m[0][3].abs();
        let y_abs_sum: f64 = (self.m.m[1][0] * x).abs()
            + (self.m.m[1][1] * y).abs()
            + (self.m.m[1][2] * z).abs()
            + self.m.m[1][3].abs();
        let z_abs_sum: f64 = (self.m.m[2][0] * x).abs()
            + (self.m.m[2][1] * y).abs()
            + (self.m.m[2][2] * z).abs()
            + self.m.m[2][3].abs();
        *p_error = Vector3f {
            x: x_abs_sum,
            y: y_abs_sum,
            z: z_abs_sum,
        } * gamma(3i32);
        assert!(wp != 0.0, "wp = {:?} != 0.0", wp);
        if wp == 1. {
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
    pub fn transform_point_with_abs_error(
        &self,
        pt: &Point3f,
        pt_error: &Vector3f,
        abs_error: &mut Vector3f,
    ) -> Point3f {
        let x: f64 = pt.x;
        let y: f64 = pt.y;
        let z: f64 = pt.z;
        // compute transformed coordinates from point _pt_
        let xp: f64 = self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z + self.m.m[0][3];
        let yp: f64 = self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z + self.m.m[1][3];
        let zp: f64 = self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z + self.m.m[2][3];
        let wp: f64 = self.m.m[3][0] * x + self.m.m[3][1] * y + self.m.m[3][2] * z + self.m.m[3][3];
        abs_error.x = (gamma(3i32) + 1.0 as f64)
            * (self.m.m[0][0].abs() * pt_error.x
                + self.m.m[0][1].abs() * pt_error.y
                + self.m.m[0][2].abs() * pt_error.z)
            + gamma(3i32)
                * ((self.m.m[0][0] * x).abs()
                    + (self.m.m[0][1] * y).abs()
                    + (self.m.m[0][2] * z).abs()
                    + self.m.m[0][3].abs());
        abs_error.y = (gamma(3i32) + 1.0 as f64)
            * (self.m.m[1][0].abs() * pt_error.x
                + self.m.m[1][1].abs() * pt_error.y
                + self.m.m[1][2].abs() * pt_error.z)
            + gamma(3i32)
                * ((self.m.m[1][0] * x).abs()
                    + (self.m.m[1][1] * y).abs()
                    + (self.m.m[1][2] * z).abs()
                    + self.m.m[1][3].abs());
        abs_error.z = (gamma(3i32) + 1.0 as f64)
            * (self.m.m[2][0].abs() * pt_error.x
                + self.m.m[2][1].abs() * pt_error.y
                + self.m.m[2][2].abs() * pt_error.z)
            + gamma(3i32)
                * ((self.m.m[2][0] * x).abs()
                    + (self.m.m[2][1] * y).abs()
                    + (self.m.m[2][2] * z).abs()
                    + self.m.m[2][3].abs());
        assert!(wp != 0.0, "wp = {:?} != 0.0", wp);
        if wp == 1. {
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
    pub fn transform_vector_with_error(&self, v: &Vector3f, abs_error: &mut Vector3f) -> Vector3f {
        let x: f64 = v.x;
        let y: f64 = v.y;
        let z: f64 = v.z;
        let gamma: f64 = gamma(3i32);
        abs_error.x = gamma
            * ((self.m.m[0][0] * v.x).abs()
                + (self.m.m[0][1] * v.y).abs()
                + (self.m.m[0][2] * v.z).abs());
        abs_error.y = gamma
            * ((self.m.m[1][0] * v.x).abs()
                + (self.m.m[1][1] * v.y).abs()
                + (self.m.m[1][2] * v.z).abs());
        abs_error.z = gamma
            * ((self.m.m[2][0] * v.x).abs()
                + (self.m.m[2][1] * v.y).abs()
                + (self.m.m[2][2] * v.z).abs());
        Vector3f {
            x: self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z,
            y: self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z,
            z: self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z,
        }
    }
    // pub fn transform_ray_with_error(
    //     &self,
    //     r: &Ray,
    //     o_error: &mut Vector3f,
    //     d_error: &mut Vector3f,
    // ) -> Ray {
    //     let mut o: Point3f = self.transform_point_with_error(&r.o, o_error);
    //     let d: Vector3f = self.transform_vector_with_error(&r.d, d_error);
    //     let length_squared: f64 = d.length_squared();
    //     if length_squared > 0.0 {
    //         let dt: f64 = vec3_dot_vec3(&d.abs(), &*o_error) / length_squared;
    //         o += d * dt;
    //     }
    //     Ray {
    //         o,
    //         d,
    //         t_max: r.t_max,
    //         time: r.time,
    //         differential: None,
    //         medium: r.medium.clone(),
    //     }
    // }
    // pub fn transform_surface_interaction(&self, si: &mut Rc<SurfaceInteraction>) {
    //     let mut ret: SurfaceInteraction = SurfaceInteraction::default();
    //     // transform _p_ and _pError_ in _SurfaceInteraction_
    //     ret.p = self.transform_point_with_abs_error(&si.p, &si.p_error, &mut ret.p_error);
    //     // transform remaining members of _SurfaceInteraction_
    //     ret.n = self.transform_normal(&si.n).normalize();
    //     ret.wo = self.transform_vector(&si.wo).normalize();
    //     ret.time = si.time;
    //     ret.uv = si.uv;
    //     ret.shape = None; // TODO? si.shape;
    //     ret.dpdu = self.transform_vector(&si.dpdu);
    //     ret.dpdv = self.transform_vector(&si.dpdv);
    //     ret.dndu = self.transform_normal(&si.dndu);
    //     ret.dndv = self.transform_normal(&si.dndv);
    //     ret.shading.n = self.transform_normal(&si.shading.n).normalize();
    //     ret.shading.dpdu = self.transform_vector(&si.shading.dpdu);
    //     ret.shading.dpdv = self.transform_vector(&si.shading.dpdv);
    //     ret.shading.dndu = self.transform_normal(&si.shading.dndu);
    //     ret.shading.dndv = self.transform_normal(&si.shading.dndv);
    //     ret.dudx = Cell::new(si.dudx.get());
    //     ret.dvdx = Cell::new(si.dvdx.get());
    //     ret.dudy = Cell::new(si.dudy.get());
    //     ret.dvdy = Cell::new(si.dvdy.get());
    //     ret.dpdx = Cell::new(si.dpdx.get());
    //     ret.dpdy = Cell::new(si.dpdy.get());
    //     // if let Some(bsdf) = &si.bsdf {
    //     //     if let Some(mut bsdf2) = ret.bsdf {
    //     //         for bxdf_idx in 0..8 {
    //     //             bsdf2.bxdfs[bxdf_idx] = match bsdf.bxdfs[bxdf_idx] {
    //     //                 _ => bsdf.bxdfs[bxdf_idx],
    //     //             };
    //     //         }
    //     //     }
    //     // }
    //     // ret.bssrdf = si.bssrdf.clone();
    //     ret.primitive = None; // TODO? si.primitive;
    //     ret.shading.n = nrm_faceforward_nrm(&ret.shading.n, &ret.n);
    //     // TODO: ret.faceIndex = si.faceIndex;
    //     *si = Rc::new(ret);
    // }
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
