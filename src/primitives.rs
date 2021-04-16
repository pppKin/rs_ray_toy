// The abstract Primitive class is the bridge between geometry processing and shaing subsystems of pbrt
use crate::{
    geometry::{dot3, Bounds3f, IntersectP, Ray},
    interaction::SurfaceInteraction,
    lights::AreaLight,
    material::{Material, TransportMode},
    medium::MediumInterface,
    misc::copy_option_arc,
    shape::Shape,
    transform::Transform,
};
use std::sync::Arc;

pub trait Primitive: IntersectP + Send + Sync {
    fn world_bound(&self) -> Bounds3f;
    fn intersect(self: Arc<Self>, r: &mut Ray, si: &mut SurfaceInteraction) -> bool;
}

#[derive(Debug)]
pub struct GeometricPrimitive {
    shape: Arc<dyn Shape>,
    material: Arc<dyn Material>,
    area_light: Option<Arc<dyn AreaLight>>,
    medium_interface: MediumInterface,
}

pub struct TransformedPrimitive {
    primitive: Arc<GeometricPrimitive>,
    primitive_to_world: Transform,
}

impl TransformedPrimitive {
    pub fn new(primitive: Arc<GeometricPrimitive>, primitive_to_world: Transform) -> Self {
        Self {
            primitive,
            primitive_to_world,
        }
    }
}

impl IntersectP for GeometricPrimitive {
    fn intersect_p(&self, r: &Ray) -> bool {
        self.shape.intersect_p(r)
    }
}

impl Primitive for GeometricPrimitive {
    fn world_bound(&self) -> Bounds3f {
        self.shape.world_bound()
    }
    fn intersect(self: Arc<Self>, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let mut t_hit: f64 = 0.0;
        if !self.shape.intersect(r, &mut t_hit, si, false) {
            return false;
        }
        si.primitive = Some(self.clone());
        r.t_max = t_hit;
        if self.clone().is_medium_transition() {
            si.ist.mi = Some(self.medium_interface.clone());
        } else {
            si.ist.mi = Some(MediumInterface {
                inside: r.medium.clone(),
                outside: r.medium.clone(),
            });
        }
        assert!(dot3(&si.ist.n, &si.shading.n) >= 0.0);
        true
    }
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<dyn Shape>,
        material: Arc<dyn Material>,
        area_light: Option<Arc<dyn AreaLight>>,
        medium_interface: MediumInterface,
    ) -> Self {
        Self {
            shape,
            material,
            area_light,
            medium_interface,
        }
    }

    pub fn get_arealight(&self) -> Option<Arc<dyn AreaLight>> {
        copy_option_arc(&self.area_light)
    }
    pub fn get_material(&self) -> Arc<dyn Material> {
        Arc::clone(&self.material)
    }
    // initializes representations of the light-scattering properties of the material at the intersection point on the surface.
    pub fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        assert!(dot3(&si.ist.n, &si.shading.n) >= 0.0);
        self.material
            .compute_scattering_functions(si, mode, allow_multiple_lobes)
    }

    pub fn is_medium_transition(&self) -> bool {
        if let (Some(inside), Some(outside)) = (
            &self.medium_interface.inside,
            &self.medium_interface.outside,
        ) {
            return Arc::ptr_eq(&inside.clone(), &outside.clone());
        }
        false
    }
}

impl IntersectP for TransformedPrimitive {
    fn intersect_p(&self, r: &Ray) -> bool {
        let world_to_prim = Transform::inverse(&self.primitive_to_world);
        return self.primitive.intersect_p(&world_to_prim.t(r));
    }
}

impl Primitive for TransformedPrimitive {
    fn world_bound(&self) -> Bounds3f {
        self.primitive_to_world.t(&self.primitive.world_bound())
    }
    fn intersect(self: Arc<Self>, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let world_to_prim = Transform::inverse(&self.primitive_to_world);
        let mut ray = world_to_prim.t(r);
        if !self.primitive.clone().intersect(&mut ray, si) {
            return false;
        }
        r.t_max = ray.t_max;
        assert!(dot3(&si.ist.n, &si.shading.n) >= 0.0);
        // Transform instance's intersection data to world space
        if !self.primitive_to_world.is_identity() {
            *si = (&self.primitive_to_world).t(si);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Point3f, Vector3f};
    use crate::shape::triangle::*;
    use crate::{material::plastic::PlasticMaterial, spectrum::Spectrum, texture::ConstantTexture};

    use super::*;

    #[test]
    fn test_primitive() {
        let obj_to_world = Transform::default();
        let world_to_obj = Transform::default();
        let mesh = create_triangle_mesh(
            obj_to_world,
            world_to_obj,
            12,
            8,
            vec![
                0, 4, 6, 4, 6, 2, 3, 2, 6, 2, 6, 7, 7, 6, 4, 6, 4, 5, 5, 1, 3, 1, 3, 7, 1, 0, 2, 0,
                2, 3, 5, 4, 0, 4, 0, 1,
            ],
            vec![],
            vec![],
            vec![
                Point3f::new(1.0, 1.0, -1.0),
                Point3f::new(1.0, -1.0, -1.0),
                Point3f::new(1.0, 1.0, 1.0),
                Point3f::new(1.0, -1.0, 1.0),
                Point3f::new(-1.0, 1.0, -1.0),
                Point3f::new(-1.0, -1.0, -1.0),
                Point3f::new(-1.0, 1.0, 1.0),
                Point3f::new(-1.0, -1.0, 1.0),
            ],
            vec![],
            vec![],
            vec![],
        );

        let kd = ConstantTexture::new(Spectrum::from(0.25));
        let ks = ConstantTexture::new(Spectrum::from(0.25));
        let roughness = ConstantTexture::new(0.1);
        let bump_map = None;
        let remap_roughness = false;
        let m = Arc::new(PlasticMaterial::new(
            Arc::new(kd),
            Arc::new(ks),
            Arc::new(roughness),
            bump_map,
            remap_roughness,
        ));

        let mut gs = Vec::with_capacity(mesh.len());
        for tri in mesh {
            gs.push(Arc::new(GeometricPrimitive::new(
                tri.clone(),
                m.clone(),
                None,
                MediumInterface {
                    inside: None,
                    outside: None,
                },
            )))
        }
        let world_pos_instances = vec![
            Transform::translate(&Vector3f::new(10.0, 10.0, 15.0))
                * Transform::scale(1.0, 1.0, 1.0),
            Transform::translate(&Vector3f::new(15.0, 10.0, 15.0))
                * Transform::scale(1.0, 1.0, 1.0),
            Transform::translate(&Vector3f::new(3.0, 3.0, 3.0)) * Transform::scale(1.0, 1.0, 1.0),
        ];

        let test_rays = vec![
            Ray::new_od(Point3f::default(), Vector3f::new(0.8, 1.0, 0.8)),
            Ray::new_od(Point3f::default(), Vector3f::new(1.0, 0.9, 0.9)),
            Ray::new_od(Point3f::default(), Vector3f::new(1.0, 1.0, 1.0)),
        ];

        let mut hit_count = 0_usize;
        let mut intersect_tested = 0_usize;
        for i in 0..3 {
            let r = &test_rays[i];
            let mut ts_tri = vec![];
            for g in &gs {
                ts_tri.push(TransformedPrimitive::new(g.clone(), world_pos_instances[i]));
            }
            for tri in ts_tri {
                intersect_tested += 1;
                if tri.intersect_p(r) {
                    hit_count += 1;
                } else {
                    // eprintln!("world bound {:?}\n ==> {:?}", tri.world_bound(), r);
                }
            }
        }
        eprintln!("Hit Count {}/{}", hit_count, intersect_tested);
        assert!(hit_count > 0);
    }
}
