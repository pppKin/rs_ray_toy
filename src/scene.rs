use std::sync::Arc;

use crate::{
    geometry::{Bounds3f, IntersectP, Ray},
    interaction::SurfaceInteraction,
    lights::Light,
    primitives::Primitive,
    samplers::Sampler,
    spectrum::Spectrum,
    SPECTRUM_N,
};

pub struct Scene {
    pub lights: Vec<Arc<dyn Light>>,
    pub infinite_lights: Vec<Arc<dyn Light>>,

    aggregate: Arc<dyn Primitive>,
    w_bound: Bounds3f,
}

impl Scene {
    pub fn new(
        lights: Vec<Arc<dyn Light>>,
        infinite_lights: Vec<Arc<dyn Light>>,
        aggregate: Arc<dyn Primitive>,
    ) -> Self {
        let w_bound = aggregate.world_bound();
        Self {
            lights,
            infinite_lights,
            aggregate,
            w_bound,
        }
    }

    pub fn intersect_tr(
        &self,
        ray: &mut Ray,
        sampler: &mut Sampler,
        isect: &mut SurfaceInteraction,
        transmittance: &mut Spectrum<SPECTRUM_N>,
    ) -> bool {
        *transmittance = Spectrum::one();
        loop {
            let hit_surface = self.intersect(ray, isect);
            if let Some(a_mi) = &ray.medium {
                let m = Arc::clone(a_mi);
                // Accumulate beam transmittance for ray segment
                *transmittance *= m.tr(ray, sampler);

                // Initialize next ray segment or terminate transmittance computation
                if !hit_surface {
                    return false;
                }
                if isect.primitive.is_some() {
                    return true;
                }
                *ray = isect.ist.spawn_ray(ray.d);
            }
        }
    }
}

impl Primitive for Scene {
    fn world_bound(&self) -> Bounds3f {
        self.w_bound
    }

    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        assert_ne!(r.d.length(), 0.0);
        self.aggregate.intersect(r, si)
    }
}

impl IntersectP for Scene {
    fn intersect_p(&self, r: &Ray) -> bool {
        assert_ne!(r.d.length(), 0.0);
        self.aggregate.intersect_p(r)
    }
}
