use super::*;
use crate::{
    geometry::{
        cross, dot3, pnt3_distance, pnt3_distance_squared, Bounds3f, Cxyz, IntersectP, Normal3f,
        Point2f, Point3f, Ray, Vector3f,
    },
    interaction::{BaseInteraction, SurfaceInteraction},
    misc::{clamp_t, quadratic},
    sampling::uniform_sample_sphere,
    transform::Transform,
    MAX_DIST,
};

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Sphere {
    obj_to_world: Transform,
    world_to_obj: Transform,
    radius: f64,
    z_min: f64,
    z_max: f64,
    theta_min: f64,
    theta_max: f64,
    phi_max: f64,
}

impl Sphere {
    pub fn new(
        obj_to_world: Transform,
        world_to_obj: Transform,
        radius: f64,
        z_min: f64,
        z_max: f64,
        phi_max: f64,
    ) -> Self {
        Self {
            obj_to_world,
            world_to_obj,
            radius,
            z_min,
            z_max,
            theta_min: clamp_t(z_min.min(z_max) / radius, -1.0, 1.0).acos(),
            theta_max: clamp_t(z_min.max(z_max) / radius, -1.0, 1.0).acos(),
            phi_max: clamp_t(phi_max, 0.0, 360.0).to_radians(),
        }
    }
}

impl IntersectP for Sphere {
    fn intersect_p(&self, r: &Ray) -> bool {
        let mut phi = 0.0;
        let mut p_hit = Point3f::default();
        let ray = self.world2obj().t(r);
        let (ox, oy, oz) = ray.o.to_xyz();
        let (dx, dy, dz) = ray.d.to_xyz();
        let a = dx * dx + dy * dy + dz * dz;
        let b = 2.0 * (dx * ox + dy * oy + dz * oz);
        let c = ox * ox + oy * oy + oz * oz - self.radius * self.radius;

        let mut t0 = 0.0;
        let mut t1 = 0.0;

        if !quadratic(a, b, c, &mut t0, &mut t1) {
            return false;
        }
        if t0 > MAX_DIST || t1 <= 0.0 {
            return false;
        }
        let mut t_s_hit = t0;
        if t0 <= 0.0 {
            t_s_hit = t1;
            if t_s_hit > MAX_DIST {
                return false;
            }
        }
        // Test sphere intersection against clipping parameters
        if (self.z_min > -self.radius && p_hit.z < self.z_min)
            || (self.z_max < self.radius && p_hit.z > self.z_max)
            || (phi > self.phi_max)
        {
            if t_s_hit == t1 {
                return false;
            }
            if t1 > MAX_DIST {
                return false;
            }
            t_s_hit = t1;
            p_hit = ray.position(t_s_hit);

            // Refine sphere intersection point
            p_hit *= self.radius / (pnt3_distance(&p_hit, &Point3f::default()));
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = f64::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if (self.z_min > -self.radius && p_hit.z < self.z_min)
                || (self.z_max < self.radius && p_hit.z > self.z_max)
                || (phi > self.phi_max)
            {
                return false;
            }
        }
        true
    }
}

impl Shape for Sphere {
    fn obj2world(&self) -> &Transform {
        &self.obj_to_world
    }
    fn world2obj(&self) -> &Transform {
        &self.world_to_obj
    }
    fn object_bound(&self) -> Bounds3f {
        Bounds3f::new(
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        )
    }
    fn intersect(
        &self,
        r: &Ray,
        thit: &mut f64,
        ist: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let mut phi;
        let mut p_hit;
        let ray = self.world2obj().t(r);
        let (ox, oy, oz) = ray.o.to_xyz();
        let (dx, dy, dz) = ray.d.to_xyz();
        let a = dx * dx + dy * dy + dz * dz;
        let b = 2.0 * (dx * ox + dy * oy + dz * oz);
        let c = ox * ox + oy * oy + oz * oz - self.radius * self.radius;

        let mut t0 = 0.0;
        let mut t1 = 0.0;

        if !quadratic(a, b, c, &mut t0, &mut t1) {
            return false;
        }
        if t0 > MAX_DIST || t1 <= 0.0 {
            return false;
        }
        let mut t_s_hit = t0;
        if t0 <= 0.0 {
            t_s_hit = t1;
            if t_s_hit > MAX_DIST {
                return false;
            }
        }

        p_hit = r.position(t_s_hit);
        // Refine sphere intersection point
        if p_hit.x == 0.0 && p_hit.y == 0.0 {
            p_hit.x = 1e-5 * self.radius;
        }
        phi = f64::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }

        // Test sphere intersection against clipping parameters
        if (self.z_min > -self.radius && p_hit.z < self.z_min)
            || (self.z_max < self.radius && p_hit.z > self.z_max)
            || (phi > self.phi_max)
        {
            if t_s_hit == t1 {
                return false;
            }
            if t1 > MAX_DIST {
                return false;
            }
            t_s_hit = t1;
            p_hit = ray.position(t_s_hit);

            // Refine sphere intersection point
            p_hit *= self.radius / (pnt3_distance(&p_hit, &Point3f::default()));
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = f64::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if (self.z_min > -self.radius && p_hit.z < self.z_min)
                || (self.z_max < self.radius && p_hit.z > self.z_max)
                || (phi > self.phi_max)
            {
                return false;
            }
        }

        // Find parametric representation of sphere hit
        let u = phi / self.phi_max;
        let theta = f64::acos(clamp_t(p_hit.z / self.radius, -1.0, 1.0));
        let v = (theta - self.theta_min) / (self.theta_max - self.theta_min);
        // Compute sphere $\dpdu$ and $\dpdv$
        let z_radius = f64::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
        let inv_z_radius = 1.0 / z_radius;
        let cos_phi = p_hit.x * inv_z_radius;
        let sin_phi = p_hit.y * inv_z_radius;
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(
            p_hit.z * cos_phi,
            p_hit.z * sin_phi,
            -self.radius * f64::sin(theta),
        ) * (self.theta_max - self.theta_min);

        // DIFFERENTIAL GEOMETRY
        // Compute sphere $\dndu$ and $\dndv$
        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * -self.phi_max * self.phi_max;
        let d2pduv = Vector3f::new(-sin_phi, cos_phi, 0.0)
            * (self.theta_max - self.theta_min)
            * p_hit.z
            * self.phi_max;
        let d2pdvv = Vector3f::from(
            p_hit * -(self.theta_max - self.theta_min) * (self.theta_max - self.theta_min),
        );

        // // Compute coefficients for fundamental forms
        let E = dot3(&dpdu, &dpdu);
        let F = dot3(&dpdu, &dpdv);
        let G = dot3(&dpdv, &dpdv);
        let N = cross(&dpdu, &dpdv).normalize();
        let e = dot3(&N, &d2pduu);
        let f = dot3(&N, &d2pduv);
        let g = dot3(&N, &d2pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        let inv_EFG2 = 1.0 / (E * G - F * F);
        let dndu = Normal3f::from(
            dpdu * ((f * F - e * G) * inv_EFG2) + dpdv * ((e * F - f * E) * inv_EFG2),
        );

        let dndv = Normal3f::from(
            dpdu * ((g * F - f * G) * inv_EFG2) + dpdv * ((f * F - g * E) * inv_EFG2),
        );

        // Initialize _SurfaceInteraction_ from parametric information
        *ist = SurfaceInteraction::new(
            p_hit,
            Point2f::new(u, v),
            -ray.d,
            dpdu,
            dpdv,
            dndu,
            dndv,
            0.0,
        );
        *ist = self.obj2world().t(ist);
        // Update _tHit_ for quadric intersection
        *thit = t_s_hit;
        return true;
    }

    fn area(&self) -> f64 {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: &Point2f, pdf: &mut f64) -> BaseInteraction {
        let mut p_obj: Point3f = Point3f::default() + uniform_sample_sphere(*u) * self.radius;
        let mut it: BaseInteraction = BaseInteraction::default();
        it.n = self
            .obj2world()
            .t(&Normal3f {
                x: p_obj.x,
                y: p_obj.y,
                z: p_obj.z,
            })
            .normalize();
        // if self.reverse_orientation {
        //     it.n *= -1.0 as Float;
        // }
        // reproject _p_obj_ to sphere surface and compute _p_obj_error_
        p_obj *= self.radius / pnt3_distance(&p_obj, &Point3f::default());

        it.p = self.obj2world().t(&p_obj);
        *pdf = 1.0 / self.area();
        it
    }

    fn solid_angle(&self, p: &Point3f, _n_samples: u32) -> f64 {
        let p_center = self.obj2world().t(&Point3f::default());
        // Point3f pCenter = (*ObjectToWorld)(Point3f(0, 0, 0));
        if pnt3_distance_squared(&p, &p_center) <= self.radius * self.radius {
            return 4.0 * PI;
        }
        // if (DistanceSquared(p, pCenter) <= radius * radius)
        //     return 4 * Pi;
        // Float sinTheta2 = radius * radius / DistanceSquared(p, pCenter);
        let sin_theta2 = self.radius * self.radius / pnt3_distance_squared(&p, &p_center);
        // Float cosTheta = std::sqrt(std::max((Float)0, 1 - sinTheta2));
        let cos_theta = f64::sqrt(f64::max(0.0, 1.0 - sin_theta2));

        return 2.0 * PI * (1.0 - cos_theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere() {
        let to_world = Transform::translate(&Vector3f::new(1.0, 0.0, 0.0));
        let to_obj = Transform::inverse(&to_world);
        let sp = Sphere::new(to_world, to_obj, 1.0, -1.0, 1.0, 360.0);
        eprintln!("sp world_bound {:?}", sp.world_bound());
        let r = Ray::new_od(Point3f::default(), Vector3f::new(1.0, 0.0, 0.0));
        assert!(sp.intersect_p(&r));
    }
}
