use image::{ImageBuffer, ImageError, Rgba};

use crate::{
    geometry::{Bounds2i, Point2i},
    misc::{clamp_t, gamma_correct},
};
pub const N_SPECTRAL_SAMPLES: usize = 60;
pub const SPECTRUM_SAMPLED_N: usize = N_SPECTRAL_SAMPLES;
pub const SPECTRUM_RGB_N: usize = 3;
// Change this to use different Spectrum Representation
pub const SPECTRUM_N: usize = SPECTRUM_RGB_N;

// use crate::camera::CameraSample;
// use crate::color::Color;
// use crate::geometry::{dot3, dot3, Point2f, Ray, Vector3f};
// use crate::material;
// // use crate::primitives::Object;
// use crate::reflection;
// use crate::scene;
// use rand::prelude::*;
// use scene::Scene;
// use std::sync::mpsc;
// use std::sync::mpsc::{Receiver, Sender};
// use std::sync::Arc;
// use std::sync::Mutex;
// use std::thread;

pub const MAX_DIST: f64 = 1999999999.0;
pub const SMALL: f64 = 0.000000001;
pub const MACHINE_EPSILON: f64 = std::f64::EPSILON * 0.5;

// pub fn calc_shadow(scn: &Scene, r: &mut Ray, collision_obj: i64) -> f64 {
//     let mut shadow: f64 = 1.0;

//     for i in 0..(scn.object_list.len()) {
//         if (i as i64) == collision_obj {
//             continue;
//         }
//         (&scn.object_list[i]).intersect(r, i as i64);

//         if r.inter_obj >= 0 {
//             let i_obj = &scn.object_list[r.inter_obj as usize];
//             shadow *= scn.material_list[i_obj.mat() as usize].transmit_col;
//             break;
//         }
//     }

//     return shadow;
// }

// pub fn trace(scn: &Scene, r: &mut Ray, depth: u64) -> Color {
//     let mut c = Color::new();

//     for i in 0..scn.object_list.len() {
//         let obj = &scn.object_list[i];
//         obj.intersect(r, i as i64);
//     }

//     if r.inter_obj >= 0 {
//         let its_obj = &scn.object_list[r.inter_obj as usize];
//         let mat_idx = its_obj.mat();
//         let inter_point = r.origin + (r.d * r.t_max);
//         let mut incident_v = inter_point - r.origin;
//         let origin_back_v = -r.d.normalize();
//         let mut v_normal = its_obj.get_normal(inter_point);

//         let t_mat = &scn.material_list[mat_idx as usize];

//         for light in &scn.light_list {
//             match light.kind.as_ref() {
//                 "ambient" => c += light.color,
//                 "point" => {
//                     let light_dir = (light.position - inter_point).normalize();
//                     let mut light_ray = Ray::create(inter_point, light_dir, MAX_DIST, -1);
//                     let shadow = calc_shadow(scn, &mut light_ray, r.inter_obj);

//                     let (ss, ts) = t_mat.bump(&v_normal);
//                     let wo_w = (-r.d).normalize();
//                     let ns = Vector3f::from(v_normal);
//                     let wo = Vector3f::new(
//                         dot3(&wo_w, &ss),
//                         dot3(&wo_w, &ts),
//                         dot3(&wo_w, &ns),
//                     );
//                     let mut wi = Vector3f::new(
//                         dot3(&wo_w, &ss),
//                         dot3(&wo_w, &ts),
//                         dot3(&wo_w, &ns),
//                     );
//                     let df =
//                         reflection::DisneyFresnel::new(light.color, t_mat.specular_d, t_mat.ior);

//                     if t_mat.diffuse_col > SMALL {
//                         let spec_trans = reflection::SpecularTransmission::new(
//                             t_mat.color,
//                             1.0,
//                             t_mat.ior,
//                             material::TransportMode::Radiance,
//                             None,
//                         );
//                         let mut st = 0;
//                         let mut pdf = 1.0;
//                         c += spec_trans.sample_f(
//                             &wo,
//                             &mut wi,
//                             Point2f::default(),
//                             &mut pdf,
//                             &mut st,
//                         );
//                     }
//                     if t_mat.specular_col > SMALL {
//                         let spec_ref = reflection::SpecularReflection::new(
//                             t_mat.color,
//                             reflection::Fresnel::Disney(df),
//                             None,
//                         );
//                         let mut st = 0;
//                         let mut pdf = 1.0;
//                         c += spec_ref.sample_f(&wo, &mut wi, Point2f::default(), &mut pdf, &mut st)
//                             * shadow;
//                     }
//                 }
//                 _ => {}
//             }
//         }

//         if depth < scn.trace_depth {
//             if t_mat.reflection_col > SMALL {
//                 let t = dot3(&origin_back_v, &v_normal);
//                 if t > SMALL {
//                     let v_dir_ref = (v_normal * 2.0).to_vec3() - origin_back_v;
//                     let v_offset_inter = inter_point + (v_dir_ref * SMALL);
//                     let mut ray_o_ref = Ray::create(v_offset_inter, v_dir_ref, MAX_DIST, -1);
//                     c += trace(&scn, &mut ray_o_ref, depth + 1) * t_mat.reflection_col;
//                 }
//             }
//             if t_mat.transmit_col > SMALL {
//                 let mut rn = dot3(&(-incident_v), &v_normal);
//                 incident_v = incident_v.normalize();
//                 let n1;
//                 let n2;
//                 if dot3(&incident_v, &v_normal) > 0.0 {
//                     v_normal = -v_normal;
//                     rn = -rn;
//                     n1 = t_mat.ior;
//                     n2 = 1.0;
//                 } else {
//                     n2 = t_mat.ior;
//                     n1 = 1.0;
//                 }

//                 if n1 != 0.0 && n2 != 0.0 {
//                     let par_sqrt = (1.0 - (n1 * n1 / n2 * n2) * (1.0 - rn * rn)).sqrt();
//                     let refract_dir = incident_v + Vector3f::from(v_normal * rn * (n1 / n2))
//                         - Vector3f::from(v_normal * par_sqrt);
//                     // let v_offset_inter = inter_point + refract_dir * SMALL;
//                     let mut refract_ray = Ray::create(inter_point, refract_dir, MAX_DIST, -1);
//                     c += trace(scn, &mut refract_ray, depth + 1) * t_mat.transmit_col;
//                 }
//             }
//         }
//     }

//     return c;
// }

// pub fn render_pixel(line_rx: Arc<Mutex<Receiver<i64>>>, done_tx: Sender<bool>, scn: &Scene) {
//     loop {
//         let rx = line_rx.lock().unwrap();
//         match rx.recv() {
//             Ok(y) => {
//                 let mut rendered_pixel = vec![];
//                 for x in 0..scn.img_width {
//                     let mut c: Color = Color::new();
//                     for _elem in 0..scn.oversampling {
//                         let mut r = Ray::new();
//                         let mut sample = CameraSample::default();
//                         sample.p_film = Point2f {
//                             x: x as f64 + random::<f64>(),
//                             y: y as f64 + random::<f64>(),
//                         };
//                         sample.p_lens = Point2f {
//                             x: random::<f64>(), //x as f64,
//                             y: random::<f64>(), //y as f64,
//                         };
//                         let weight = scn.cam.generate_ray(&sample, &mut r);
//                         c += trace(scn, &mut r, 1) * weight;
//                     }
//                     c /= scn.oversampling as f64;
//                     rendered_pixel.push(c);
//                 }
//                 let mut img = scn.img.lock().unwrap();
//                 for x in 0..scn.img_width {
//                     let c = rendered_pixel[x as usize];
//                     img.put_pixel(x, y as u64, c.to_pixel());
//                 }
//                 done_tx.send(true).unwrap();
//             }
//             Err(_e) => {
//                 break;
//             }
//         }
//     }
// }

// pub fn deploy_renderer(scene_filename: &str, workers_num: u8, save_path: &str) {
//     let scn = Arc::new(scene::make_scene(scene_filename));

//     println!("Rendering {}", scene_filename);
//     let (done_tx, done_rx): (Sender<bool>, Receiver<bool>) = mpsc::channel();
//     let (line_tx, line_rx): (Sender<i64>, Receiver<i64>) = mpsc::channel();
//     let l_rx = Arc::new(Mutex::new(line_rx));
//     let mut hns = vec![];

//     let mut done_result = 0;
//     let total_count = (scn.img_height - 1) as f64;
//     hns.push(thread::spawn(move || {
//         print!(
//             "Render Progress: {:.2}%",
//             (done_result as f64 / total_count) * 100.0
//         );
//         for d_b in done_rx {
//             if d_b {
//                 done_result += 1;
//                 print!(
//                     "\rRender Progress: {:.2}%",
//                     (done_result as f64 / total_count) * 100.0
//                 );
//             } else {
//                 // done_result[1] += 1;
//                 println!("Error Rendering Image!")
//             }
//         }
//         print!("\n");
//     }));

//     for _i in 0..workers_num {
//         let c_scn = Arc::clone(&scn);
//         let l_rx_mt = Arc::clone(&l_rx);
//         let d_tx = done_tx.clone();
//         hns.push(thread::spawn(move || {
//             render_pixel(l_rx_mt, d_tx, &c_scn);
//         }));
//     }

//     for y in scn.startline..scn.endline {
//         line_tx.send(y as i64).unwrap();
//     }

//     drop(done_tx);
//     drop(line_tx);

//     for hn in hns {
//         hn.join().unwrap();
//     }

//     scn.img.lock().unwrap().save(save_path).unwrap();
// }

pub fn write_image(
    filename: &str,
    rgb: &[f64],
    output_bounds: Bounds2i,
    total_resolution: Point2i,
) -> Result<(), ImageError> {
    let resolution = output_bounds.diagonal();

    let mut img_buf = ImageBuffer::new(resolution.x as u32, resolution.y as u32);
    for y in 0..resolution.y {
        for x in 0..resolution.x {
            let r = rgb[(3 * (y * resolution.x + x) + 0) as usize];
            let g = rgb[(3 * (y * resolution.x + x) + 1) as usize];
            let b = rgb[(3 * (y * resolution.x + x) + 2) as usize];
            img_buf.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    (clamp_t(255.0 * gamma_correct(r) + 0.5, 0.0, 255.0)) as u8,
                    (clamp_t(255.0 * gamma_correct(g) + 0.5, 0.0, 255.0)) as u8,
                    (clamp_t(255.0 * gamma_correct(b) + 0.5, 0.0, 255.0)) as u8,
                    255 as u8,
                ]),
            )
        }
    }
    img_buf.save(filename)
}
