use crate::geometry;
use crate::geometry::{Point2i, Point3f, Vector3f};
use crate::lights::DeprecatedLight;
use crate::material::Material;
use crate::misc::read_lines;
use crate::primitives;
use crate::primitives::Primitive;
use crate::rtoycore;
use crate::transform::Transform;
use crate::{camera::PerspectiveCamera, geometry::Ray, interaction::SurfaceInteraction};
use image::{ImageBuffer, RgbaImage};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;
pub struct Scene {
    pub img_width: u32,
    pub img_height: u32,
    pub trace_depth: u32,
    pub oversampling: u8,
    pub startline: u32,
    pub endline: u32,
    pub cam: Arc<PerspectiveCamera>,
    pub img: Arc<Mutex<RgbaImage>>,
    pub object_list: Vec<Arc<dyn Primitive>>,
    pub light_list: Vec<Arc<DeprecatedLight>>,
    pub material_list: Vec<Arc<dyn Material>>,
}

impl Scene {
    pub fn intersect(&self, ray: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        todo!();
    }
}

fn parse_err(kw: &str, ln: u32, filename: &str) -> String {
    format!("Error reading {} at Line {}, {}", kw, ln, filename)
}

fn parse_xyz<T>(data: Vec<&str>) -> Result<T, &str>
where
    T: geometry::Cxyz<f64>,
{
    if data.len() < 3 {
        return Err("Not enough components for a vector");
    }
    let x = f64::from_str(data[0]);
    if x.is_err() {
        return Err(data[0]);
    };
    let y = f64::from_str(data[1]);
    if y.is_err() {
        return Err(data[1]);
    };
    let z = f64::from_str(data[2]);
    if z.is_err() {
        return Err(data[2]);
    };
    return Ok(T::from_xyz(x.unwrap(), y.unwrap(), z.unwrap()));
}

pub fn make_scene(scene_filename: &str) -> Scene {
    // TODO: figure out this part
    let zero_vec3 = Vector3f::default();

    // defaults
    let mut img_w: u32 = 320;
    let mut img_h: u32 = 240;
    let mut oversampling: u8 = 3;
    let mut trace_depth = 3; // bounces
    let mut lens_radius = 14.5;
    let mut focal_distance = 26.0;
    let mut startline = 0; // Start rendering line
    let mut endline = 1;

    let mut camera_pos: Point3f = Point3f::default();
    let mut camera_look: Point3f = Point3f::default();
    let mut camera_up: Vector3f = zero_vec3;
    let mut fov = 90.0;

    let mut obj_list: Vec<Arc<dyn Primitive>> = vec![];
    let mut light_list: Vec<Arc<DeprecatedLight>> = vec![];
    let mut mat_list: Vec<Arc<Material>> = vec![];

    if let Ok(lines) = read_lines(scene_filename) {
        // Consumes the iterator, returns an (Optional) String
        let mut cur_line_num: u32 = 0;
        for line in lines {
            if let Ok(l) = line {
                cur_line_num = cur_line_num + 1;
                // println!("{}", l);
                if l.is_empty() {
                    continue;
                }
                if Some("#") == l.get(..1) {
                    continue;
                }
                let mut sline = l.split_whitespace();
                let keyword = sline.next().unwrap();
                let mut data: Vec<&str> = vec![];
                for tmp in sline {
                    // println!("{}", tmp);
                    data.push(tmp);
                }
                match keyword {
                    "image_size" => {
                        img_w = u32::from_str(data[0])
                            .expect(&(parse_err("image width", cur_line_num, scene_filename)));
                        img_h = u32::from_str(data[1])
                            .expect(&(parse_err("image height", cur_line_num, scene_filename)));
                        endline = img_h - 1;
                    }
                    "depth" => {
                        trace_depth = u32::from_str(data[0])
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "oversampling" => {
                        oversampling = u8::from_str(data[0])
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "renderslice" => {
                        startline = u32::from_str(data[0])
                            .expect(&(parse_err("start line", cur_line_num, scene_filename)));
                        endline = u32::from_str(data[0])
                            .expect(&(parse_err("end line", cur_line_num, scene_filename)));
                    }
                    "camera_position" => {
                        camera_pos = parse_xyz(data)
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "camera_look" => {
                        camera_look = parse_xyz(data)
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "camera_up" => {
                        camera_up = parse_xyz(data)
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "lens_radius" => {
                        lens_radius = f64::from_str(data[0])
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "focal_distance" => {
                        focal_distance = f64::from_str(data[0])
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    "fov" => {
                        fov = f64::from_str(data[0])
                            .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    }
                    // "sphere" => {
                    //     let mat: u32 = u32::from_str(data[0])
                    //         .expect(&(parse_err("sphere material", cur_line_num, scene_filename)));
                    //     let rad: f64 = f64::from_str(data[4])
                    //         .expect(&(parse_err("sphere radius", cur_line_num, scene_filename)));
                    //     let pos = parse_xyz(data[1..4].to_vec())
                    //         .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    //     let sph = primitives::Sphere {
                    //         mat: mat,
                    //         radius: rad,
                    //         position: pos,
                    //     };
                    //     obj_list.push(Arc::new(Primitive::Sphere(Arc::new(sph))));
                    // }
                    // "plane" => {
                    //     let mat: u32 = u32::from_str(data[0])
                    //         .expect(&(parse_err("plane material", cur_line_num, scene_filename)));
                    //     let dis: f64 = f64::from_str(data[4])
                    //         .expect(&(parse_err("plane distancia", cur_line_num, scene_filename)));
                    //     let nrm = parse_xyz(data[1..4].to_vec())
                    //         .expect(&(parse_err(keyword, cur_line_num, scene_filename)));
                    //     let pln = primitives::Plane {
                    //         mat: mat,
                    //         distancia: dis,
                    //         normal: nrm,
                    //     };
                    //     obj_list.push(Arc::new(Primitive::Plane(Arc::new(pln))));
                    // }
                    "light" => {
                        let l = DeprecatedLight {
                            position: parse_xyz(data[1..4].to_vec()).expect(
                                &(parse_err("light position", cur_line_num, scene_filename)),
                            ),
                            color: parse_xyz(data[4..7].to_vec()).expect(
                                &(parse_err("light position", cur_line_num, scene_filename)),
                            ),
                            kind: String::from(data[0]),
                        };
                        light_list.push(Arc::new(l));
                    }
                    // "material" => mat_list.push(Arc::new(Material {
                    //     color: parse_xyz(data[0..3].to_vec())
                    //         .expect(&(parse_err("material color", cur_line_num, scene_filename))),
                    //     diffuse_col: f64::from_str(data[3]).expect(
                    //         &(parse_err("material disfuse color", cur_line_num, scene_filename)),
                    //     ),
                    //     specular_col: f64::from_str(data[4]).expect(
                    //         &(parse_err("material specular_col", cur_line_num, scene_filename)),
                    //     ),
                    //     specular_d: f64::from_str(data[5]).expect(
                    //         &(parse_err("material specular_d", cur_line_num, scene_filename)),
                    //     ),
                    //     reflection_col: f64::from_str(data[6]).expect(
                    //         &(parse_err("material reflection_col", cur_line_num, scene_filename)),
                    //     ),
                    //     transmit_col: f64::from_str(data[3]).expect(
                    //         &(parse_err("material transmit_col", cur_line_num, scene_filename)),
                    //     ),
                    //     ior: f64::from_str(data[3])
                    //         .expect(&(parse_err("material ior", cur_line_num, scene_filename))),
                    // })),
                    _ => println!(
                        "Unrecognized keyword {} at line {} {}",
                        keyword, cur_line_num, scene_filename
                    ),
                }
            } else {
                break;
            }
        }
    } else {
        panic!(format!("failed to open scene file: {}", scene_filename))
    }

    let tmp_cam_look = camera_look - Point3f::default();
    if camera_up.length() <= rtoycore::SMALL {
        let tmp = geometry::cross(
            &tmp_cam_look,
            &Vector3f {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        );
        camera_up = geometry::cross(&tmp, &tmp_cam_look);
    }
    let img_buf = ImageBuffer::new(img_w as u32, img_h as u32);
    let img = Arc::new(Mutex::new(img_buf));

    let t: Transform = Transform::look_at(&camera_pos, &camera_look, &camera_up);

    let it: Transform = Transform {
        m: t.m_inv.clone(),
        m_inv: t.m.clone(),
    };

    let resolution = Point2i {
        x: img_w as i64,
        y: img_h as i64,
    };

    let cam = PerspectiveCamera::create(it, resolution, fov, 0.0, 0.0, lens_radius, focal_distance);

    return Scene {
        img_width: img_w,
        img_height: img_h,
        oversampling: oversampling,
        trace_depth: trace_depth,
        startline: startline,
        endline: endline,
        cam: Arc::new(cam),
        img: img,
        object_list: obj_list,
        light_list: light_list,
        material_list: mat_list,
    };
}
