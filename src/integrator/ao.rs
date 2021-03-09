use std::sync::Arc;

use crate::{
    geometry::{cross, dot3, faceforward, IntersectP, RayDifferential, Vector3f},
    interaction::SurfaceInteraction,
    primitives::Primitive,
    samplers::Sampler,
    sampling::{
        cosine_hemisphere_pdf, cosine_sample_hemisphere, uniform_hemisphere_pdf,
        uniform_sample_hemisphere,
    },
    scene::Scene,
    spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{Integrator, SamplerIntegrator, SamplerIntegratorData};

#[derive(Debug)]
pub struct AOIntegrator<T>
where
    T: Sampler + Clone,
{
    cos_sample: bool,
    n_samples: u32,

    i: Arc<SamplerIntegratorData<T>>,
}

impl<T> AOIntegrator<T>
where
    T: Sampler + Clone,
{
    pub fn new(cos_sample: bool, ns: u32, i: Arc<SamplerIntegratorData<T>>) -> Self {
        let mut s = (*i.sampler).clone();
        let n_samples = s.round_count(ns);
        if n_samples != ns {
            // TODO: warn n_samples != ns
        }
        s.request_2d_array(n_samples);
        Self {
            cos_sample,
            n_samples,
            i,
        }
    }
}

impl<T> Integrator for AOIntegrator<T>
where
    T: Sampler + Clone,
{
    fn render(&mut self, scene: &Scene) {
        self.si_render(scene)
    }
}

impl<T> SamplerIntegrator<T> for AOIntegrator<T>
where
    T: Sampler + Clone,
{
    fn itgt(&self) -> Arc<SamplerIntegratorData<T>> {
        Arc::clone(&self.i)
    }

    fn li(
        &self,
        r: &mut RayDifferential,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        _depth: usize,
    ) -> Spectrum<SPECTRUM_N> {
        let mut l = Spectrum::zero();
        let mut ray = r.clone();

        // Intersect _ray_ with scene and store intersection in _isect_
        let mut isect = SurfaceInteraction::default();
        if scene.intersect(&mut ray.ray, &mut isect) {
            if isect.bsdf.is_none() {
                return Spectrum::zero();
            }
            // Compute coordinate frame based on true geometry, not shading
            // geometry.
            let n = faceforward(&isect.ist.n, &-ray.ray.d);
            let s = isect.dpdu.normalize();
            let t = cross(&isect.ist.n, &s);
            let u = sampler.get_2d_array(self.n_samples);
            for i in 0..self.n_samples {
                let mut wi;
                let pdf;
                if self.cos_sample {
                    wi = cosine_sample_hemisphere(u[i as usize]);
                    pdf = cosine_hemisphere_pdf(wi.z.abs());
                } else {
                    wi = uniform_sample_hemisphere(u[i as usize]);
                    pdf = uniform_hemisphere_pdf();
                }
                // Transform wi from local frame to world space.
                wi = Vector3f::new(
                    s.x * wi.x + t.x * wi.y + n.x * wi.z,
                    s.y * wi.x + t.y * wi.y + n.y * wi.z,
                    s.z * wi.x + t.z * wi.y + n.z * wi.z,
                );

                if !scene.intersect_p(&isect.ist.spawn_ray(wi)) {
                    l += Spectrum::from(dot3(&wi, &n) / (pdf * self.n_samples as f64));
                }
            }
        }

        l
    }
}
