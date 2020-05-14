use crate::camera::CameraSample;
use crate::color::Color;
use crate::geometry::{vec3_dot_nrm, Point2f, Ray, Vector3f};
use crate::primitives::Object;
use crate::scene;
use scene::Scene;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

pub const MAX_DIST: f64 = 1999999999.0;
pub const SMALL: f64 = 0.000000001;
pub const MACHINE_EPSILON: f64 = std::f64::EPSILON * 0.5;

pub fn calc_shadow(scn: &Scene, r: &mut Ray, collision_obj: i32) -> f64 {
    let mut shadow: f64 = 1.0;

    for i in 0..(scn.object_list.len() - 1) {
        let obj = &scn.object_list[i];
        r.inter_obj = -1;
        r.inter_dist = MAX_DIST;

        if obj.intersect(r, i as i32) && (i as i32) != collision_obj {
            shadow *= scn.material_list[obj.mat() as usize].transmit_col;
        }
    }

    return shadow;
}

pub fn trace(scn: &Scene, r: &mut Ray, depth: u32) -> Color {
    let mut c = Color::new();

    for i in 0..scn.object_list.len() {
        let obj = &scn.object_list[i];
        obj.intersect(r, i as i32);
    }

    if r.inter_obj >= 0 {
        let its_obj = &scn.object_list[r.inter_obj as usize];
        let mat_idx = its_obj.mat();
        let inter_point = r.origin + (r.direction * r.inter_dist);
        let mut incident_v = inter_point - r.origin;
        let origin_back_v = -r.direction.normalize();
        let mut v_normal = its_obj.get_normal(inter_point);

        let t_mat = &scn.material_list[mat_idx as usize];

        for light in &scn.light_list {
            match light.kind.as_ref() {
                "ambient" => c += light.color,
                "point" => {
                    let light_dir = (light.position - inter_point).normalize();
                    let mut light_ray = Ray {
                        origin: inter_point,
                        inter_dist: MAX_DIST,
                        direction: light_dir,
                        inter_obj: -1,
                    };

                    let shadow = calc_shadow(scn, &mut light_ray, r.inter_obj);
                    let nl = vec3_dot_nrm(&light_dir, &v_normal);

                    if nl > SMALL {
                        if t_mat.diffuse_col > SMALL {
                            let dif_col =
                                light.color * t_mat.diffuse_col * nl * t_mat.color * shadow;
                            c += dif_col;
                        }

                        if t_mat.specular_col > SMALL {
                            let tmp = (light_dir + origin_back_v) / 2.0;
                            let tmp_spec = vec3_dot_nrm(&tmp, &v_normal);

                            if tmp_spec > SMALL {
                                let spec = t_mat.specular_col * tmp_spec.powf(t_mat.specular_d);
                                let spec_col = light.color * spec * shadow;
                                c += spec_col;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if depth < scn.trace_depth {
            if t_mat.reflection_col > SMALL {
                let t = vec3_dot_nrm(&origin_back_v, &v_normal);
                if t > SMALL {
                    let v_dir_ref = (v_normal * 2.0 * t).to_vec3() - origin_back_v;
                    let v_offset_inter = inter_point + (v_dir_ref * SMALL);
                    let mut ray_o_ref = Ray {
                        origin: v_offset_inter,
                        direction: v_dir_ref,
                        inter_dist: MAX_DIST,
                        inter_obj: -1,
                    };
                    c += trace(&scn, &mut ray_o_ref, depth + 1) * t_mat.reflection_col;
                }
            }
            if t_mat.transmit_col > SMALL {
                let mut rn = vec3_dot_nrm(&(-incident_v), &v_normal);
                incident_v = incident_v.normalize();
                let n1;
                let n2;
                if vec3_dot_nrm(&incident_v, &v_normal) > 0.0 {
                    v_normal = -v_normal;
                    rn = -rn;
                    n1 = t_mat.ior;
                    n2 = 1.0;
                } else {
                    n2 = t_mat.ior;
                    n1 = 1.0;
                }

                if n1 != 0.0 && n2 != 0.0 {
                    let par_sqrt = (1.0 - (n1 * n1 / n2 * n2) * (1.0 - rn * rn)).sqrt();
                    let refract_dir = incident_v + Vector3f::from(v_normal * rn * (n1 / n2))
                        - Vector3f::from(v_normal * par_sqrt);
                    let v_offset_inter = inter_point + refract_dir * SMALL;
                    let mut refract_ray = Ray {
                        origin: v_offset_inter,
                        direction: refract_dir,
                        inter_dist: MAX_DIST,
                        inter_obj: -1,
                    };
                    c += trace(scn, &mut refract_ray, depth + 1) * t_mat.transmit_col;
                }
            }
        }
    }

    return c;
}

pub fn render_pixel(line_rx: Arc<Mutex<Receiver<i32>>>, done_tx: Sender<bool>, scn: &Scene) {
    loop {
        let rx = line_rx.lock().unwrap();
        match rx.recv() {
            Ok(y) => {
                for x in 0..scn.img_width {
                    let mut c: Color = Color::new();
                    let mut r = Ray::new();
                    let mut sample = CameraSample::default();
                    sample.p_film = Point2f {
                        x: x as f64,
                        y: y as f64,
                    };
                    let weight = scn.cam.generate_ray(&sample, &mut r);
                    c += trace(scn, &mut r, 1) * weight;
                    let mut img = scn.img.lock().unwrap();
                    img.put_pixel(x, y as u32, c.to_pixel());
                    done_tx.send(true).unwrap();
                }
            }
            Err(_e) => {
                break;
            }
        }
    }
}

pub fn deploy_renderer(scene_filename: &str, workers_num: u8, save_path: &str) {
    let scn = Arc::new(scene::make_scene(scene_filename));

    println!("Rendering {}", scene_filename);
    let (done_tx, done_rx): (Sender<bool>, Receiver<bool>) = mpsc::channel();
    let (line_tx, line_rx): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let l_rx = Arc::new(Mutex::new(line_rx));
    let mut hns = vec![];
    for _i in 0..workers_num {
        let c_scn = Arc::clone(&scn);
        let l_rx_mt = Arc::clone(&l_rx);
        let d_tx = done_tx.clone();
        hns.push(thread::spawn(move || {
            render_pixel(l_rx_mt, d_tx, &c_scn);
        }));
    }

    for y in scn.startline..scn.endline {
        line_tx.send(y as i32).unwrap();
    }

    drop(done_tx);
    drop(line_tx);

    for hn in hns {
        hn.join().unwrap();
    }
    let mut done_result = vec![0, 0];
    for d_b in done_rx {
        if d_b {
            done_result[0] += 1;
        } else {
            done_result[1] += 1;
        }
    }

    println!(
        "Renderers Result Success:{} Failure: {}",
        done_result[0], done_result[1]
    );

    scn.img.lock().unwrap().save(save_path).unwrap();
}
