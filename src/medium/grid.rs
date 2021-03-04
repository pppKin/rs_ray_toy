use std::sync::Arc;

use crate::{
    geometry::{
        pnt3_floor, Bounds3f, Bounds3i, Normal3f, Point3f, Point3i, Ray, Vector3f, Vector3i,
    },
    interaction::{BaseInteraction, MediumInteraction},
    medium::{HenyeyGreenstein, Medium},
    misc::lerp,
    samplers::Sampler,
    spectrum::Spectrum,
    transform::Transform,
    SPECTRUM_N,
};

#[derive(Debug)]
pub struct GridDensityMedium {
    sigma_a: Spectrum<SPECTRUM_N>,
    sigma_s: Spectrum<SPECTRUM_N>,

    g: f64,

    nx: i32,
    ny: i32,
    nz: i32,

    world_to_medium: Transform,
    density: Vec<f64>,
    sigma_t: f64,
    inv_max_density: f64,
}

impl GridDensityMedium {
    pub fn new(
        sigma_a: Spectrum<SPECTRUM_N>,
        sigma_s: Spectrum<SPECTRUM_N>,
        g: f64,
        nx: i32,
        ny: i32,
        nz: i32,
        world_to_medium: Transform,
        d: &[f64],
    ) -> Self {
        let mut density = vec![];
        for i in 0..((nx * ny * nz) as usize).min(d.len()) {
            density.push(d[i]);
        }
        // Precompute values for Monte Carlo sampling of _GridDensityMedium_
        let sigma_t = (sigma_a + sigma_s)[0];

        if Spectrum::from(sigma_t) != (sigma_s + sigma_a) {
            eprintln!("GridDensityMedium requires a spectrally uniform attenuation coefficient!");
        }
        let mut max_density: f64 = 0.0;
        for i in 0..nx * ny * nz {
            max_density = max_density.max(density[i as usize]);
        }
        let inv_max_density = 1.0 / max_density;
        Self {
            sigma_a,
            sigma_s,
            g,
            nx,
            ny,
            nz,
            world_to_medium,
            density,
            sigma_t,
            inv_max_density,
        }
    }
    fn d(&self, p: &Point3i) -> f64 {
        let sample_bounds = Bounds3i::new(
            Point3i::new(0, 0, 0),
            Point3i::new(self.nx as i64, self.ny as i64, self.nz as i64),
        );
        if !Bounds3i::inside_exclusive(p, &sample_bounds) {
            return 0.0;
        }
        self.density[((p.z * self.ny as i64 + p.y) * self.nx as i64 + p.x) as usize]
    }
    fn density(&self, p: &Point3i) -> f64 {
        // Compute voxel coordinates and offsets for _p_
        let p_samples = Point3f::new(
            (p.x as f64 * self.nx as f64) - 0.5,
            p.y as f64 * self.ny as f64 - 0.5,
            p.z as f64 * self.nz as f64 - 0.5,
        );

        let p_f = pnt3_floor(&p_samples);
        let pi = Point3i::from_pnt3f(&p_f);
        let d = p_samples - p_f;

        // Trilinearly interpolate density values to compute local density
        let d00 = lerp(d.x, self.d(&pi), self.d(&(pi + Vector3i::new(1, 0, 0))));
        let d10 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 1, 0))),
            self.d(&(pi + Vector3i::new(1, 1, 0))),
        );
        let d01 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 0, 1))),
            self.d(&(pi + Vector3i::new(1, 0, 1))),
        );
        let d11 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 1, 1))),
            self.d(&(pi + Vector3i::new(1, 1, 1))),
        );

        let d0 = lerp(d.y, d00, d10);
        let d1 = lerp(d.y, d01, d11);
        lerp(d.z, d0, d1)
    }
}

impl Medium for GridDensityMedium {
    fn tr(&self, r_world: &Ray, sampler: &mut dyn Sampler) -> Spectrum<SPECTRUM_N> {
        let ray = self.world_to_medium.t(&Ray::new(
            r_world.o,
            r_world.d.normalize(),
            r_world.t_max * r_world.d.length(),
            0.0,
            None,
        ));
        // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
        let b = Bounds3f::new(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let mut t_min = 1.0;
        let mut t_max = 1.0;
        if !b.intersect_b(&ray, &mut t_min, &mut t_max) {
            return Spectrum::from(1.0);
        }
        // Perform ratio tracking to estimate the transmittance value
        let mut tr = 1.0;
        let mut t = t_min;
        loop {
            t -= (1.0 - sampler.get_1d()).ln() * self.inv_max_density / self.sigma_t as f64;
            if t >= t_max {
                break;
            }
            let density = self.density(&Point3i::from_pnt3f(&ray.position(t)));
            tr *= 1.0 - (density * self.inv_max_density).max(0.0);
            // Added after book publication: when transmittance gets low,
            // start applying Russian roulette to terminate sampling.
            let rr_threshold = 0.1;
            if tr < rr_threshold {
                let q = (1.0 - tr).max(0.05);
                if sampler.get_1d() < q {
                    return Spectrum::zero();
                }
                tr /= 1.0 - q;
            }
        }
        Spectrum::from(tr)
    }

    fn sample(
        &self,
        r_world: &crate::geometry::Ray,
        sampler: &mut dyn Sampler,
        mi: &mut MediumInteraction,
    ) -> Spectrum<SPECTRUM_N> {
        let ray = self.world_to_medium.t(&Ray::new(
            r_world.o,
            r_world.d.normalize(),
            r_world.t_max * r_world.d.length(),
            0.0,
            None,
        ));
        // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
        let b = Bounds3f::new(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let mut t_min = 1.0;
        let mut t_max = 1.0;
        if !b.intersect_b(&ray, &mut t_min, &mut t_max) {
            return Spectrum::from(1.0);
        }

        // Run delta-tracking iterations to sample a medium interaction
        let mut t = t_min;
        loop {
            t -= (1.0 - sampler.get_1d()).ln() * self.inv_max_density / self.sigma_t as f64;
            if t >= t_max {
                break;
            }
            if self.density(&Point3i::from_pnt3f(&ray.position(t))) * self.inv_max_density
                > sampler.get_1d()
            {
                // Populate _mi_ with medium interaction information and return
                let phase = HenyeyGreenstein::new(self.g);

                *mi = MediumInteraction::new(
                    BaseInteraction::new(
                        r_world.position(t),
                        r_world.time,
                        Vector3f::default(),
                        -r_world.d,
                        Normal3f::default(),
                        None,
                    ),
                    Some(Arc::new(phase)),
                );
                return self.sigma_s / self.sigma_t;
            }
        }
        Spectrum::from(1.0)
    }
}
