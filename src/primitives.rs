use crate::core;
use crate::geometry;
use crate::geometry::{Cxyz, Normal3f, Point3f, Ray, Vector3f};
use crate::misc;

use std::sync::Arc;

pub trait Object {
    fn mat(&self) -> u32;
    fn intersect(&self, r: &mut Ray, i: i32) -> bool;
    fn get_normal(&self, point: Point3f) -> Normal3f;
}

pub struct Sphere {
    pub mat: u32,
    pub position: Point3f,
    pub radius: f64,
}

pub struct Plane {
    // material  int
    pub mat: u32,
    pub normal: Normal3f,
    pub distancia: f64,
}

pub enum Primitive {
    Sphere(Arc<Sphere>),
    Plane(Arc<Plane>),
}

impl Object for Primitive {
    fn mat(&self) -> u32 {
        match self {
            Primitive::Sphere(sph) => (*sph).mat,
            Primitive::Plane(pln) => (*pln).mat,
        }
    }
    fn intersect(&self, r: &mut Ray, i: i32) -> bool {
        match self {
            Primitive::Sphere(sph) => {
                let obj_space_ray_o = r.origin - (*sph).position;
                let tmp_a: f64 = r.direction.length_squared();
                let tmp_b: f64 = 2.0
                    * (r.direction.x * obj_space_ray_o.x
                        + r.direction.y * obj_space_ray_o.y
                        + r.direction.z * obj_space_ray_o.z);
                let tmp_c: f64 = obj_space_ray_o.x.powi(2)
                    + obj_space_ray_o.y.powi(2)
                    + obj_space_ray_o.z.powi(2)
                    - (*sph).radius.powi(2);

                let mut t0: f64 = 0.0;
                let mut t1: f64 = 0.0;
                // println!("intersection quadratic: {}, {}, {}", tmp_a, tmp_b, tmp_c);

                if misc::quadratic(tmp_a, tmp_b, tmp_c, &mut t0, &mut t1) {
                    // println!(
                    //     "intersection quadratic: {}, {}",
                    //     &t0, &t1
                    // );
                } else {
                    return false;
                }

                if t1 < 0.0 || t0 > core::MAX_DIST {
                    return false;
                }
                let mut tmp_t = t0;
                if t0 < 0.0 {
                    tmp_t = t1;
                    if tmp_t > core::MAX_DIST {
                        return false;
                    }
                }
                if tmp_t > r.inter_dist {
                    return false;
                }
                r.inter_dist = tmp_t;
                r.inter_obj = i;

                return true;
            }
            Primitive::Plane(pln) => {
                let v = geometry::vec3_dot_nrm(&r.direction, &((*pln).normal));
                if v == 0.0 {
                    return false;
                }
                let collision_dist = ((*pln).distancia
                    - geometry::vec3_dot_nrm(
                        &(r.origin
                            - Point3f {
                                x: 0.0,
                                y: 0.0,
                                z: 0.0,
                            }),
                        &((*pln).normal),
                    ))
                    / v;

                if collision_dist < 0.0 {
                    return false;
                }
                if collision_dist > r.inter_dist {
                    return false;
                }

                r.inter_dist = collision_dist;
                r.inter_obj = i;
                return true;
            }
            _ => false,
        }
    }
    fn get_normal(&self, point: Point3f) -> Normal3f {
        match self {
            Primitive::Sphere(sph) => Normal3f::from((point - (*sph).position).normalize()),
            Primitive::Plane(pln) => (*pln).normal,
            _ => Normal3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::Cxyz;
    #[test]
    fn test_plane() {
        let mat: u32 = 1;
        let dis: f64 = 6.0;
        let nrm = Normal3f::from_xyz(0.0, 1.0, 0.0);
        let p = Plane {
            mat: mat,
            distancia: dis,
            normal: nrm,
        };
        let pln = Primitive::Plane(Arc::new(p));

        let mut r1 = Ray {
            direction: Vector3f::from_xyz(0.0, 1.0, 0.0),
            inter_dist: core::MAX_DIST,
            inter_obj: -1,
            origin: Point3f::default(),
        };
        assert!(pln.intersect(&mut r1, 4));
        assert!(r1.inter_dist == 6.0);

        let p1 = Plane {
            mat: mat,
            distancia: 14.0,
            normal: Normal3f::from_xyz(1.0, 1.0, 0.0).normalize(),
        };
        let pln1 = Primitive::Plane(Arc::new(p1));

        let mut r2 = Ray {
            direction: Vector3f::from_xyz(0.0, 1.0, 0.0).normalize(),
            inter_dist: core::MAX_DIST,
            inter_obj: -1,
            origin: Point3f::default(),
        };
        assert!(pln1.intersect(&mut r2, 8));
        assert!((r2.inter_dist - 19.79898987322333).abs() <= core::SMALL);
    }

    #[test]
    fn test_sph() {
        let s1 = Primitive::Sphere(Arc::new(Sphere {
            mat: 1,
            position: Point3f {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            radius: 1.0,
        }));
        let mut r1 = Ray {
            direction: Vector3f::from_xyz(0.0, 1.0, 0.0).normalize(),
            inter_dist: core::MAX_DIST,
            inter_obj: -1,
            origin: Point3f::from_xyz(0.0, -2.0, 0.0),
        };
        s1.intersect(&mut r1, 8);
        println!("intersecting r1 sphere {}", r1.inter_dist);
        assert!((r1.inter_dist - 2.0).abs() <= core::SMALL);

        let s2 = Primitive::Sphere(Arc::new(Sphere {
            mat: 1,
            position: Point3f {
                x: 2.0,
                y: 2.0,
                z: 2.0,
            },
            radius: 1.0,
        }));
        let s2p = Point3f {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let mut r2 = Ray {
            direction: Vector3f::from_xyz(1.0, 1.0, 1.0).normalize(),
            inter_dist: core::MAX_DIST,
            inter_obj: -1,
            origin: Point3f::default(),
        };
        s2.intersect(&mut r2, 8);
        let s2rad: f64 = 1.0;
        println!(
            "intersecting r2 sphere {} :: {} ",
            r2.inter_dist,
            ((s2p - r2.origin).length() - s2rad)
        );
        assert!((r2.inter_dist - ((s2p - r2.origin).length() - s2rad)).abs() <= core::SMALL);
    }
}
