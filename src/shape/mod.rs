use crate::{
    geometry::{abs_dot3, Bounds3f, IntersectP, Point2f, Point3f, Ray, Vector3f},
    interaction::{BaseInteraction, SurfaceInteraction},
    transform::Transform,
};

pub trait Shape: IntersectP {
    fn obj2world(&self) -> &Transform;
    fn world2obj(&self) -> &Transform;
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f {
        self.obj2world().transform_bounds(&self.object_bound())
    }
    fn intersect(
        &self,
        r: &Ray,
        thit: &mut f64,
        ist: &mut SurfaceInteraction,
        test_alpha_texture: bool,
    ) -> bool;
    fn area(&self) -> f64;
    // Sample a point on the surface of the shape and return the PDF with
    // respect to area on the surface.
    fn sample(&self, u: &Point2f, pdf: &mut f64) -> BaseInteraction;
    fn pdf(&self, ist: &BaseInteraction) -> f64 {
        1.0 / self.area()
    }

    // Sample a point on the shape given a reference point |ref| and
    // return the PDF with respect to solid angle from |ref|.
    fn sample_ref(&self, ref_ist: &BaseInteraction, u: &Point2f, pdf: &mut f64) -> BaseInteraction {
        let intr = self.sample(u, pdf);

        let mut wi = intr.p - ref_ist.p;
        let wi_len_sq = wi.length_squared();
        if wi_len_sq == 0.0 {
            *pdf = 0.0;
        } else {
            wi = wi.normalize();
            *pdf = wi_len_sq / abs_dot3(&(-wi), &(intr.n));
            if (*pdf).is_infinite() {
                *pdf = 0.0;
            }
        }
        intr
    }
    fn pdf_ref(&self, ref_ist: &BaseInteraction, wi: &Vector3f) -> f64 {
        // Intersect sample ray with area light geometry
        let r = ref_ist.spawn_ray(*wi);
        let mut thit = 0.0;
        let mut ist_light = SurfaceInteraction::default();

        if !self.intersect(&r, &mut thit, &mut ist_light, false) {
            return 0.0;
        }

        let mut pdf = (ref_ist.p - ist_light.ist.p).length_squared()
            / (abs_dot3(&(-*wi), &(ist_light.ist.n)) * self.area());
        // // Convert light sample weight to solid angle measure
        if pdf.is_infinite() {
            pdf = 0.0;
        }
        pdf
    }

    // Returns the solid angle subtended by the shape w.r.t. the reference
    // point p, given in world space. Some shapes compute this value in
    // closed-form, while the default implementation uses Monte Carlo
    // integration; the nSamples parameter determines how many samples are
    // used in this case.
    fn solid_angle(&self, p: &Point3f, n_samples: u32) -> f64;
}

pub mod sphere;
pub mod triangle;
