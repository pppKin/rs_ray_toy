use super::*;
use crate::{
    geometry::{
        cross, dot3, faceforward, vec3_coordinate_system, Bounds3f, IntersectP, Normal3f, Point2f,
        Point3f, Ray, Vector3f,
    },
    interaction::{BaseInteraction, SurfaceInteraction},
    misc::{clamp_t, uniform_sample_sphere},
    transform::Transform,
};

use std::{f64::consts::PI, sync::Arc};

#[derive(Debug)]
pub struct TriangleMesh {
    pub n_triangles: u32, // The total number of triangles in the mesh.
    pub n_vertices: u32,  // The total number of vertices in the mesh.
    // A pointer to an array of vertex indices. For the ith triangle, its three vertex positions are
    pub vertex_indices: Vec<u32>,
    pub p: Vec<Point3f>,  // An array of n_vertices vertex positions.
    pub n: Vec<Normal3f>, // An optional array of normals
    pub s: Vec<Vector3f>, // An optional array of tangent vectors, one per vertex, used to compute shading tangents.
    pub uv: Vec<Point2f>, // An optional array of (u, v)
                          // std::shared_ptr<Texture<Float>> alphaMask, shadowAlphaMask;
}

impl TriangleMesh {
    pub fn new(
        n_triangles: u32,
        n_vertices: u32,
        vertex_indices: Vec<u32>,
        p: Vec<Point3f>,
        n: Vec<Normal3f>,
        s: Vec<Vector3f>,
        uv: Vec<Point2f>,
    ) -> Self {
        TriangleMesh {
            n_triangles,
            n_vertices,
            vertex_indices,
            p,
            n,
            s,
            uv,
        }
    }
}

#[derive(Debug)]
pub struct Triangle {
    obj_to_world: Transform,
    world_to_obj: Transform,
    v: [usize; 3],
    mesh: Arc<TriangleMesh>,
}

impl Triangle {
    pub fn new(
        obj_to_world: Transform,
        world_to_obj: Transform,
        tri_num: usize,
        mesh: Arc<TriangleMesh>,
    ) -> Self {
        Triangle {
            obj_to_world,
            world_to_obj,
            v: [
                mesh.vertex_indices[3 * tri_num] as usize,
                mesh.vertex_indices[3 * tri_num + 1] as usize,
                mesh.vertex_indices[3 * tri_num + 2] as usize,
            ], //mesh.vertex_indices[3 * tri_num .. 3 * tri_num + 3],
            mesh,
        }
    }

    pub fn get_uvs(&self) -> [Point2f; 3] {
        if self.mesh.uv.is_empty() {
            [
                Point2f { x: 0.0, y: 0.0 },
                Point2f { x: 1.0, y: 0.0 },
                Point2f { x: 1.0, y: 1.0 },
            ]
        } else {
            [
                self.mesh.uv[self.v[0]],
                self.mesh.uv[self.v[1]],
                self.mesh.uv[self.v[2]],
            ]
        }
    }
}

pub fn create_triangle_mesh(
    obj_to_world: Transform,
    world_to_obj: Transform,
    n_triangles: u32,
    n_vertices: u32,
    vertex_indices: Vec<u32>,
    p: Vec<Point3f>,
    n: Vec<Normal3f>,
    s: Vec<Vector3f>,
    uv: Vec<Point2f>,
) -> Vec<Arc<Triangle>> {
    let mesh = Arc::new(TriangleMesh::new(
        n_triangles,
        n_vertices,
        vertex_indices,
        p,
        n,
        s,
        uv,
    ));
    let mut tri = vec![];
    for i in 0..n_triangles {
        tri.push(Arc::new(Triangle::new(
            obj_to_world,
            world_to_obj,
            i as usize,
            Arc::clone(&mesh),
        )));
    }
    tri
}

// pub fn create_triangle_mesh_shape() -> Vec<Triangle>{

// }
impl IntersectP for Triangle {
    fn intersect_p(&self, r: &Ray) -> bool {
        // Möller–Trumbore intersection algorithm
        // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let E1 = p1 - p0;
        let E2 = p2 - p1;
        let D = r.d;
        // [E1, E2, -D] Transpose([t,u,v]) = T
        // use Cramer's rule

        let P = cross(&D, &E2);
        let a = dot3(&E1, &P);
        if a > -0.0000001 && a < 0.0000001 {
            return false;
        }
        let f = 1.0 / a;

        let T = r.o - p0;
        let u = f * dot3(&T, &P);
        if u < 0.0 || u > 1.0 {
            return false;
        }

        let Q = cross(&T, &E1);
        let v = f * dot3(&D, &Q);
        if v < 0.0 || (u + v) > 1.0 {
            return false;
        }
        // // At this stage we can compute t to find out where the intersection point is on the line.
        // float t = f * edge2.dotProduct(q);
        let t = f * dot3(&E2, &Q);
        if t < 0.0000001 {
            return false;
        }
        true
    }
}

impl Shape for Triangle {
    fn obj2world(&self) -> &Transform {
        &self.obj_to_world
    }
    fn world2obj(&self) -> &Transform {
        &self.world_to_obj
    }
    fn object_bound(&self) -> Bounds3f {
        let p0 = self.world2obj().t(&self.mesh.p[self.v[0]]);
        let p1 = self.world2obj().t(&self.mesh.p[self.v[1]]);
        let p2 = self.world2obj().t(&self.mesh.p[self.v[2]]);
        Bounds3f::union(&Bounds3f::new(p0, p1), &p2)
    }
    fn world_bound(&self) -> Bounds3f {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        Bounds3f::union(&Bounds3f::new(p0, p1), &p2)
    }
    fn intersect(
        &self,
        r: &Ray,
        thit: &mut f64,
        ist: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        // Möller–Trumbore intersection algorithm
        // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let E1 = p1 - p0;
        let E2 = p2 - p1;
        let D = r.d;
        // [E1, E2, -D] Transpose([t,u,v]) = T
        // use Cramer's rule

        let P = cross(&D, &E2);
        let a = dot3(&E1, &P);
        if a > -0.0000001 && a < 0.0000001 {
            return false;
        }
        let f = 1.0 / a;

        let T = r.o - p0;
        let u = f * dot3(&T, &P);
        if u < 0.0 || u > 1.0 {
            return false;
        }

        let Q = cross(&T, &E1);
        let v = f * dot3(&D, &Q);
        if v < 0.0 || (u + v) > 1.0 {
            return false;
        }
        // // At this stage we can compute t to find out where the intersection point is on the line.
        // float t = f * edge2.dotProduct(q);
        let t = f * dot3(&E2, &Q);
        if t < 0.0000001 {
            return false;
        }
        *thit = t;
        // Compute triangle partial derivatives
        let uv = self.get_uvs();
        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
        let degenerate_uv = determinant.abs() < 1e-8;

        let mut dpdu = Vector3f::default();
        let mut dpdv = Vector3f::default();
        if !degenerate_uv {
            let i_det = 1.0 / determinant;
            dpdu = (dp02 * duv12[1] - dp12 * duv02[1]) * i_det;
            dpdv = (dp02 * -duv12[0] + dp12 * duv02[0]) * i_det;
        }
        if degenerate_uv || cross(&dpdu, &dpdv).length_squared() == 0.0 {
            // Handle zero determinant for triangle partial derivative matrix
            let ng = cross(&(p2 - p0), &(p1 - p0));
            if ng.length_squared() == 0.0 {
                // The triangle is actually degenerate; the intersection is
                // bogus.
                return false;
            }

            vec3_coordinate_system(&ng.normalize(), &mut dpdu, &mut dpdv);
        }
        let p_hit = r.position(t);
        let uv_hit = uv[0] * (1.0 - u - v) + uv[1] * u + uv[2] * v;
        *ist = SurfaceInteraction::new(
            p_hit,
            uv_hit,
            -r.d,
            dpdu,
            dpdv,
            Normal3f::default(),
            Normal3f::default(),
            0.0,
        );
        let ist_n = Normal3f::from(cross(&dp02, &dp12).normalize());
        ist.ist.n = ist_n;
        ist.shading.n = ist_n;
        if !self.mesh.n.is_empty() || !self.mesh.s.is_empty() {
            // Initialize _Triangle_ shading geometry
            // Compute shading normal _ns_ for triangle
            let mut ns;
            if !self.mesh.n.is_empty() {
                ns = self.mesh.n[self.v[0]] * (1.0 - u - v)
                    + self.mesh.n[self.v[1]] * u
                    + self.mesh.n[self.v[2]] * v;

                if ns.length_squared() > 0.0 {
                    ns = ns.normalize();
                } else {
                    ns = ist_n;
                }
            } else {
                ns = ist_n;
            }

            // Compute shading tangent _ss_ for triangle
            let mut ss;
            if !self.mesh.s.is_empty() {
                ss = self.mesh.s[self.v[0]] * (1.0 - u - v)
                    + self.mesh.s[self.v[1]] * u
                    + self.mesh.s[self.v[2]] * v;
                if ss.length_squared() > 0.0 {
                    ss = ss.normalize();
                } else {
                    ss = ist.dpdu.normalize();
                }
            } else {
                ss = ist.dpdu.normalize();
            }

            let mut ts = cross(&ss, &ns);
            if ts.length_squared() > 0.0 {
                ts = ts.normalize();
                ss = cross(&ts, &ns);
            } else {
                vec3_coordinate_system(&Vector3f::from(ns), &mut ss, &mut ts);
            }
            // Compute $\dndu$ and $\dndv$ for triangle shading geometry
            let mut dndu = Normal3f::default();
            let mut dndv = Normal3f::default();
            if !self.mesh.n.is_empty() {
                // Compute deltas for triangle partial derivatives of normal
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];
                let dn1 = self.mesh.n[self.v[0]] - self.mesh.n[self.v[2]];
                let dn2 = self.mesh.n[self.v[1]] - self.mesh.n[self.v[2]];
                // Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
                let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
                let degenerate_uv = determinant.abs() < 1e-8;
                if degenerate_uv {
                    // We can still compute dndu and dndv, with respect to the
                    // same arbitrary coordinate system we use to compute dpdu
                    // and dpdv when this happens. It's important to do this
                    // (rather than giving up) so that ray differentials for
                    // rays reflected from triangles with degenerate
                    // parameterizations are still reasonable.
                    let dn = cross(
                        &Vector3f::from(self.mesh.n[self.v[2]] - self.mesh.n[self.v[0]]),
                        &Vector3f::from(self.mesh.n[self.v[1]] - self.mesh.n[self.v[0]]),
                    );
                    if dn.length_squared() != 0.0 {
                        let mut dnu = Vector3f::default();
                        let mut dnv = Vector3f::default();
                        vec3_coordinate_system(&Vector3f::from(dn), &mut dnu, &mut dnv);
                        dndu = Normal3f::from(dnu);
                        dndv = Normal3f::from(dnv);
                    }
                } else {
                    let i_det = 1.0 / determinant;
                    dndu = (dn1 * duv12[1] - dn2 * duv02[1]) * i_det;
                    dndv = (dn1 * -duv12[0] + dn2 * duv02[0]) * i_det;
                }
            }
            ist.set_shading_geometry(&ss, &ts, &dndu, &dndv, true);
        }

        true
    }

    fn sample(&self, u: &Point2f, pdf: &mut f64) -> BaseInteraction {
        let mut it: BaseInteraction = BaseInteraction::default();

        // Point2f b = UniformSampleTriangle(u);
        let b = uniform_sample_sphere(*u);
        // Get triangle vertices in _p0_, _p1_, and _p2_
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];

        it.p = p0 * b[0] + p1 * b[1] + p2 * b[2];
        // Compute surface normal for sampled point on triangle
        it.n = Normal3f::from(cross(&(p1 - p0), &(p2 - p0)).normalize());
        // Ensure correct orientation of the geometric normal; follow the same
        // approach as was used in Triangle::Intersect().
        if !self.mesh.n.is_empty() {
            let ns = self.mesh.n[self.v[0]] * b[0]
                + self.mesh.n[self.v[1]] * b[1]
                + self.mesh.n[self.v[2]] * b[2];

            it.n = faceforward(&it.n, &ns);
        }

        *pdf = 1.0 / self.area();

        it
    }

    fn area(&self) -> f64 {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        0.5 * cross(&(p1 - p0), &(p2 - p0)).length()
    }

    fn solid_angle(&self, p: &Point3f, _n_samples: u32) -> f64 {
        // Project the vertices into the unit sphere around p.
        // std::array<Vector3f, 3> pSphere = {
        //     Normalize(mesh->p[v[0]] - p), Normalize(mesh->p[v[1]] - p),
        //     Normalize(mesh->p[v[2]] - p)
        // };
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let p_sphere = vec![
            (p0 - *p).normalize(),
            (p1 - *p).normalize(),
            (p2 - *p).normalize(),
        ];

        // http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
        // Girard's theorem: surface area of a spherical triangle on a unit
        // sphere is the 'excess angle' alpha+beta+gamma-pi, where
        // alpha/beta/gamma are the interior angles at the vertices.
        //
        // Given three vertices on the sphere, a, b, c, then we can compute,
        // for example, the angle c->a->b by
        //
        // cos theta =  Dot(Cross(c, a), Cross(b, a)) /
        //              (Length(Cross(c, a)) * Length(Cross(b, a))).
        //
        let mut cross01 = cross(&p_sphere[0], &p_sphere[1]);
        let mut cross12 = cross(&p_sphere[1], &p_sphere[2]);
        let mut cross20 = cross(&p_sphere[2], &p_sphere[0]);

        // Some of these vectors may be degenerate. In this case, we don't want
        // to normalize them so that we don't hit an assert. This is fine,
        // since the corresponding dot products below will be zero.
        if cross01.length_squared() > 0.0 {
            cross01 = cross01.normalize();
        }
        if cross12.length_squared() > 0.0 {
            cross12 = cross12.normalize();
        }
        if cross20.length_squared() > 0.0 {
            cross20 = cross20.normalize();
        }

        // We only need to do three cross products to evaluate the angles at
        // all three vertices, though, since we can take advantage of the fact
        // that Cross(a, b) = -Cross(b, a).
        (clamp_t(dot3(&cross01, &cross12), -1.0, 1.0).acos()
            + clamp_t(dot3(&cross12, &cross20), -1.0, 1.0).acos()
            + clamp_t(dot3(&cross20, &cross01), -1.0, 1.0).acos()
            - PI)
            .abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::geometry::*;
    // use rand::prelude::*;
    #[test]
    fn test_tri() {
        let obj_to_world = Transform::default();
        let world_to_obj = Transform::default();
        let _mesh = create_triangle_mesh(
            obj_to_world,
            world_to_obj,
            2,
            6,
            vec![0, 1, 2, 3, 4, 5],
            vec![
                Point3f::new(-1.0, -1.0, -1.0),
                Point3f::new(1.0, 1.0, 1.0),
                Point3f::new(-1.0, 1.0, -1.0),
                Point3f::new(-1.0, -1.0, -1.0) / 2.0,
                Point3f::new(1.0, 1.0, 1.0) / 2.0,
                Point3f::new(-1.0, 1.0, -1.0) / 2.0,
                Point3f::new(-1.0, -1.0, -1.0) / 4.0,
                Point3f::new(1.0, 1.0, 1.0) / 4.0,
                Point3f::new(-1.0, 1.0, -1.0) / 4.0,
            ],
            vec![],
            vec![],
            vec![],
        );
    }
}
