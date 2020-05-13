use crate::core::MAX_DIST;
use crate::geometry::{Bounds2f, Cxyz, Point2f, Point2i, Point3f, Ray, Vector3f};
use crate::misc::concentric_sample_disk;
use crate::transform::Transform;

pub struct Camera {
    pub camera_to_world: Transform,
    pub resolution: Point2i,
    pub shutter_open: f64,
    pub shutter_close: f64,
    pub raster_to_camera: Transform,
    pub lens_radius: f64,
    pub focal_distance: f64,

    pub dx_camera: Vector3f,
    pub dy_camera: Vector3f,

    pub a: f64,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct CameraSample {
    pub p_film: Point2f, // the point on the film
    pub p_lens: Point2f, // the point on the lens that the ray pass through
    pub time: f64,
}

impl Camera {
    pub fn new(
        camera_to_world: Transform,
        screen_window: Bounds2f,
        resolution: Point2i,
        fov: f64,
        shutter_open: f64,
        shutter_close: f64,
        lens_radius: f64,
        focal_distance: f64,
    ) -> Self {
        let camera_to_screen: Transform = Transform::perspective(fov, 1e-2, 1000.0);
        // compute projective camera screen transformations
        let scale1 = Transform::scale(resolution.x as f64, resolution.y as f64, 1.0);
        let scale2 = Transform::scale(
            1.0 / (screen_window.p_max.x - screen_window.p_min.x),
            1.0 / (screen_window.p_min.y - screen_window.p_max.y),
            1.0,
        );
        let translate = Transform::translate(&Vector3f {
            x: -screen_window.p_min.x,
            y: -screen_window.p_max.y,
            z: 0.0,
        });
        let screen_to_raster = scale1 * scale2 * translate;
        let raster_to_screen = Transform::inverse(&screen_to_raster);
        let raster_to_camera = Transform::inverse(&camera_to_screen) * raster_to_screen;
        // see perspective.cpp
        // compute differential changes in origin for perspective camera rays
        let dx_camera: Vector3f = raster_to_camera.transform_point(&Point3f {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }) - raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        let dy_camera: Vector3f = raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }) - raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        // compute image plane bounds at $z=1$ for _PerspectiveCamera_
        let mut p_min: Point3f = raster_to_camera.transform_point(&Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        // Point3f p_max = RasterToCamera(Point3f(res.x, res.y, 0));
        let mut p_max: Point3f = raster_to_camera.transform_point(&Point3f {
            x: resolution.x as f64,
            y: resolution.y as f64,
            z: 0.0,
        });
        p_min /= p_min.z;
        p_max /= p_max.z;
        let a: f64 = ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs();

        Camera {
            camera_to_world,
            resolution,
            shutter_open,
            shutter_close,
            raster_to_camera,
            lens_radius,
            focal_distance,
            dx_camera,
            dy_camera,
            a,
        }
    }
    pub fn create(
        cam2world: Transform,
        resolution: Point2i,
        fov: f64,
        shutteropen: f64,
        shutterclose: f64,
        lensradius: f64,
        focaldistance: f64,
    ) -> Camera {
        assert!(shutterclose >= shutteropen);
        let frame = resolution.x as f64 / resolution.y as f64;

        let mut screen: Bounds2f = Bounds2f::default();
        if frame > 1.0 {
            screen.p_min.x = -frame;
            screen.p_max.x = frame;
            screen.p_min.y = -1.0;
            screen.p_max.y = 1.0;
        } else {
            screen.p_min.x = -1.0;
            screen.p_max.x = 1.0;
            screen.p_min.y = -1.0 / frame;
            screen.p_max.y = 1.0 / frame;
        }

        Camera::new(
            cam2world,
            screen,
            resolution,
            fov,
            shutteropen,
            shutterclose,
            lensradius,
            focaldistance,
        )
    }

    // Camera
    pub fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f64 {
        // We will use a simplified version here
        let p_film: Point3f = Point3f {
            x: sample.p_film.x,
            y: sample.p_film.y,
            z: 0.0,
        };
        let p_camera = self.raster_to_camera.transform_point(&p_film);
        // every ray passed in here has default origin 0.0
        // ray.origin = Point3f::default();
        ray.direction = Vector3f::from(p_camera);
        // TODO: modified according to depth of field here
        *ray = self.camera_to_world.transform_ray(&ray);
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use crate::primitives::Object;
    use std::sync::Arc;
    #[test]
    fn test_camera() {
        let cam_pos = Point3f::from_xyz(0.0, 0.0, 0.0);
        let cam_lookat = Point3f::from_xyz(0.0, 0.0, 1.0);
        let cam_up = Vector3f::from_xyz(0.0, 1.0, 0.0);

        let t: Transform = Transform::look_at(&cam_pos, &cam_lookat, &cam_up);
        let resolution = Point2i { x: 32, y: 32 };
        let it: Transform = Transform {
            m: t.m_inv.clone(),
            m_inv: t.m.clone(),
        };
        // let cam = Camera::create(it, resolution, 0.0, 0.0, 14.5, 26.0);

        let mut r1 = Ray::new();
        let mut r2 = Ray::new();
        let mut r3 = Ray::new();
        let mut r4 = Ray::new();
        let mut r5 = Ray::new();

        // cam.generate_ray(&mut r1, 8.0, 8.0);
        // cam.generate_ray(&mut r2, 0.0, 0.0);
        // cam.generate_ray(&mut r3, 32.0, 32.0);
        // cam.generate_ray(&mut r4, 16.0, 16.0);
        // cam.generate_ray(&mut r5, 17.0, 17.0);

        println!("Generated ray direction 1:: {:?}", r1.direction);
        println!("Generated ray direction 2:: {:?}", r2.direction);
        println!("Generated ray direction 3:: {:?}", r3.direction);
        println!("Generated ray direction 4:: {:?}", r4.direction);
        println!("Generated ray direction 5:: {:?}", r5.direction);

        let s2 = primitives::Primitive::Sphere(Arc::new(primitives::Sphere {
            mat: 1,
            position: Point3f {
                x: 0.0,
                y: 0.0,
                z: 5.0,
            },
            radius: 2.0,
        }));

        assert!(!s2.intersect(&mut r1, 8));
        assert!(s2.intersect(&mut r4, 8));
        assert!(s2.intersect(&mut r5, 8));
    }
}
