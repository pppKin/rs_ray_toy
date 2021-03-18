use image::{ImageBuffer, ImageError, Rgba};
use serde_json::Value;

use std::{collections::HashMap, fs, sync::Arc};

use crate::{
    geometry::{Bounds2i, Cxyz, Point3f},
    integrator::Integrator,
    lights::{diffuse::DiffuseAreaLight, point::PointLight, Light},
    material::{
        disney::DisneyMaterial,
        glass::GlassMaterial,
        matte::MatteMaterial,
        metal::{MetalMaterial, COPPER_K, COPPER_N},
        mirror::MirrorMaterial,
        mixmat::MixMaterial,
        plastic::PlasticMaterial,
        translucent::TranslucentMaterial,
        Material,
    },
    medium::{grid::GridDensityMedium, homogeneous::HomogeneousMedium, Medium, MediumInterface},
    misc::{clamp_t, gamma_correct},
    primitives::Primitive,
    scene::Scene,
    shape::{sphere::Sphere, triangle::Triangle, Shape},
    spectrum::Spectrum,
    texture::{ConstantTexture, Texture},
    transform::Transform,
    SPECTRUM_N,
};

struct SceneGlobalData {
    float_texture: HashMap<String, Arc<dyn Texture<f64>>>,
    rgb_texture: HashMap<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>,

    materials: HashMap<String, Arc<dyn Material>>,
    triangle_mesh: HashMap<String, Vec<Arc<Triangle>>>,

    lights: Vec<Arc<dyn Light>>,
    infinite_lights: Vec<Arc<dyn Light>>,
}

/// read scene config file (json), create required resources, and start rendering
pub fn deploy_render(filepath: &str, save_to: &str) {
    let scene_config_str = fs::read_to_string(filepath).unwrap();
    let scene_config: Value = serde_json::from_str(&scene_config_str).unwrap();
    let (scene, scene_global) = make_scene(&scene_config);
    let mut inte = make_integrator(&scene_config, &scene_global, save_to);
    inte.render(&scene);
}

/// Search first level of value for a key
fn search_object<'a>(root: &'a Value, key: &str) -> Result<&'a Value, String> {
    if let Value::Object(obj) = root {
        if let Some(o) = obj.get(key) {
            return Ok(o);
        }
    }
    Err(format!(
        "Failed to Retrive {} from scene config {}",
        key, root
    ))
}

// all integer are parsed as i64, use as uszie/i32 to get desired type
fn read_i64(root: &Value, key: &str) -> i64 {
    root.get(key).unwrap().as_i64().unwrap()
}

fn read_f64(root: &Value, key: &str) -> f64 {
    root.get(key).unwrap().as_f64().unwrap()
}

fn read_bool(root: &Value, key: &str) -> bool {
    root.get(key).unwrap().as_bool().unwrap()
}

fn read_num_array(v: &Value, length: usize) -> Result<Vec<f64>, String> {
    if let Value::Array(ary) = v {
        if ary.len() != length {
            return Err(format!(
                "Failed to parse Transform from {:?}, expected and array of {} numbers",
                ary, length
            ));
        }
        let mut a = Vec::with_capacity(length);
        for num in ary {
            if num.is_f64() {
                a.push(num.as_f64().unwrap());
            } else {
                return Err(format!("Failed to parse into a f64: {:?}", num));
            }
        }
        return Ok(a);
    } else {
        return Err(format!("Not an array: {:?}", v));
    }
}

fn make_xyz<'a, T>(root: &'a Value) -> Result<T, String>
where
    T: Cxyz<f64>,
{
    match read_num_array(root, 3) {
        Ok(nums) => {
            return Ok(T::from_xyz(nums[0], nums[1], nums[2]));
        }
        Err(e) => Err(format!(
            "Failed to parse x, y, z from {:?}, with error: {}",
            root, e
        )),
    }
}

/// create a transform
fn make_transform<'a>(root: &'a Value) -> Result<Transform, String> {
    match read_num_array(root, 3) {
        Ok(m) => {
            return Ok(Transform::new(
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
                m[13], m[14], m[15],
            ));
        }
        Err(e) => Err(format!(
            "Failed to parse Transform from {:?}, with error: {}",
            root, e
        )),
    }
}

fn make_scene(scene_config: &Value) -> (Scene, SceneGlobalData) {
    // material and triangle mesh is used globally so we create them first and passed them around
    let (float_texture, rgb_texture) = make_textures(scene_config);
    let mut scene_global_data = SceneGlobalData {
        float_texture,
        rgb_texture,
        materials: HashMap::new(),
        triangle_mesh: HashMap::new(),
        lights: vec![],
        infinite_lights: vec![],
    };
    scene_global_data.materials = make_materials(&scene_config, &scene_global_data);
    scene_global_data.triangle_mesh = make_triangle_mesh(&scene_config, &scene_global_data);

    let (lights, infinite_lights) = make_all_lights(&scene_config, &scene_global_data);
    scene_global_data.lights = lights.clone();
    scene_global_data.infinite_lights = infinite_lights.clone();
    let aggregate = make_aggregate(&scene_config, &scene_global_data);
    (
        Scene::new(lights, infinite_lights, aggregate),
        scene_global_data,
    )
}

fn make_textures(
    scene_config: &Value,
) -> (
    HashMap<String, Arc<dyn Texture<f64>>>,
    HashMap<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>,
) {
    todo!();
}

fn fetch_mat_float_texture(
    mat_config: &Value,
    scene_global: &SceneGlobalData,
    tex_key: &str,
    default_value: Option<f64>,
) -> Arc<dyn Texture<f64>> {
    return if let Some(Value::String(tex_name)) = mat_config.get(tex_key) {
        scene_global.float_texture[tex_name].clone()
    } else if let Some(d) = default_value {
        Arc::new(ConstantTexture::new(d))
    } else {
        panic!("Failed to fetch Texture: {}", tex_key)
    };
}

fn fetch_mat_float_texture_opt(
    mat_config: &Value,
    scene_global: &SceneGlobalData,
    tex_key: &str,
    default_value: Option<f64>,
) -> Option<Arc<dyn Texture<f64>>> {
    return if let Some(Value::String(tex_name)) = mat_config.get(tex_key) {
        Some(scene_global.float_texture[tex_name].clone())
    } else if let Some(d) = default_value {
        Some(Arc::new(ConstantTexture::new(d)))
    } else {
        None
    };
}

fn fetch_mat_rgb_texture<T>(
    mat_config: &Value,
    scene_global: &SceneGlobalData,
    tex_key: &str,
    default_value: Option<T>,
) -> Arc<dyn Texture<Spectrum<SPECTRUM_N>>>
where
    T: Into<Spectrum<SPECTRUM_N>> + Copy + Clone,
{
    return if let Some(Value::String(tex_name)) = mat_config.get(tex_key) {
        scene_global.rgb_texture[tex_name].clone()
    } else if let Some(d) = default_value {
        Arc::new(ConstantTexture::new(d.into()))
    } else {
        panic!("Failed to fetch Texture: {}", tex_key)
    };
}

fn make_materials(
    scene_config: &Value,
    scene_global: &SceneGlobalData,
) -> HashMap<String, Arc<dyn Material>> {
    let mut materials_map = HashMap::<String, Arc<dyn Material>>::new();
    if let Some(Value::Array(mat_ary)) = scene_config.get("materials") {
        for mat_config in mat_ary {
            let material_type = mat_config.get("material_type").unwrap().as_str().unwrap();
            let material_name =
                String::from(mat_config.get("material_name").unwrap().as_str().unwrap());
            match material_type {
                "MixMaterial" => {
                    let mat1_name = mat_config.get("mat1").unwrap().as_str().unwrap();
                    let mat2_name = mat_config.get("mat2").unwrap().as_str().unwrap();
                    if materials_map.contains_key(mat1_name)
                        && materials_map.contains_key(mat2_name)
                    {
                        let scale =
                            fetch_mat_rgb_texture(mat_config, scene_global, "scale", Some(0.5));
                        materials_map.insert(
                            material_name,
                            Arc::new(MixMaterial::new(
                                scene_global.materials[mat1_name].clone(),
                                scene_global.materials[mat1_name].clone(),
                                scale,
                            )),
                        );
                    }
                }
                "TranslucentMaterial" => {
                    let kd = fetch_mat_rgb_texture(mat_config, scene_global, "kd", Some(0.25));
                    let ks = fetch_mat_rgb_texture(mat_config, scene_global, "ks", Some(0.25));
                    let roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "roughness", Some(0.1));
                    let reflect =
                        fetch_mat_rgb_texture(mat_config, scene_global, "reflect", Some(0.25));
                    let transmit =
                        fetch_mat_rgb_texture(mat_config, scene_global, "transmit", Some(0.25));
                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness");

                    materials_map.insert(
                        material_name,
                        Arc::new(TranslucentMaterial::new(
                            kd,
                            ks,
                            roughness,
                            reflect,
                            transmit,
                            bump_map,
                            remap_roughness,
                        )),
                    );
                }
                "MetalMaterial" => {
                    let eta =
                        fetch_mat_rgb_texture(mat_config, scene_global, "eta", Some(*COPPER_N));
                    let k = fetch_mat_rgb_texture(mat_config, scene_global, "k", Some(*COPPER_K));
                    let roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "roughness", Some(0.01));
                    let u_roughness =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "u_roughness", None);
                    let v_roughness =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "v_roughness", None);
                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness");
                    materials_map.insert(
                        material_name,
                        Arc::new(MetalMaterial::new(
                            eta,
                            k,
                            roughness,
                            u_roughness,
                            v_roughness,
                            bump_map,
                            remap_roughness,
                        )),
                    );
                }
                "PlasticMaterial" => {
                    let kd = fetch_mat_rgb_texture(mat_config, scene_global, "kd", Some(0.25));
                    let ks = fetch_mat_rgb_texture(mat_config, scene_global, "ks", Some(0.25));
                    let roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "roughness", Some(0.1));
                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness");
                    materials_map.insert(
                        material_name,
                        Arc::new(PlasticMaterial::new(
                            kd,
                            ks,
                            roughness,
                            bump_map,
                            remap_roughness,
                        )),
                    );
                }
                "MirrorMaterial" => {
                    let kr = fetch_mat_rgb_texture(mat_config, scene_global, "kr", Some(0.9));
                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    materials_map
                        .insert(material_name, Arc::new(MirrorMaterial::new(kr, bump_map)));
                }
                "GlassMaterial" => {
                    let kr = fetch_mat_rgb_texture(mat_config, scene_global, "kr", Some(1.0));
                    let kt = fetch_mat_rgb_texture(mat_config, scene_global, "kt", Some(1.0));
                    let eta = fetch_mat_float_texture(mat_config, scene_global, "eta", Some(1.5));

                    let u_roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "u_roughness", Some(0.0));
                    let v_roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "v_roughness", Some(0.0));

                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness");

                    materials_map.insert(
                        material_name,
                        Arc::new(GlassMaterial::new(
                            kr,
                            kt,
                            u_roughness,
                            v_roughness,
                            eta,
                            bump_map,
                            remap_roughness,
                        )),
                    );
                }
                "MatteMaterial" => {
                    let kd = fetch_mat_rgb_texture(mat_config, scene_global, "kd", Some(0.5));
                    let sigma =
                        fetch_mat_float_texture(mat_config, scene_global, "sigma", Some(0.0));

                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    materials_map.insert(
                        material_name,
                        Arc::new(MatteMaterial::new(kd, sigma, bump_map)),
                    );
                }
                "DisneyMaterial" => {
                    let color = fetch_mat_rgb_texture(mat_config, scene_global, "color", Some(0.5));
                    let metallic =
                        fetch_mat_float_texture(mat_config, scene_global, "metallic", Some(0.0));
                    let eta = fetch_mat_float_texture(mat_config, scene_global, "eta", Some(1.5));
                    let roughness =
                        fetch_mat_float_texture(mat_config, scene_global, "roughness", Some(0.5));
                    let specular_tint = fetch_mat_float_texture(
                        mat_config,
                        scene_global,
                        "specular_tint",
                        Some(0.0),
                    );
                    let anisotropic =
                        fetch_mat_float_texture(mat_config, scene_global, "anisotropic", Some(0.0));
                    let sheen =
                        fetch_mat_float_texture(mat_config, scene_global, "sheen", Some(0.0));
                    let sheen_tint =
                        fetch_mat_float_texture(mat_config, scene_global, "sheen_tint", Some(0.5));
                    let clearcoat =
                        fetch_mat_float_texture(mat_config, scene_global, "clearcoat", Some(0.0));
                    let clearcoat_gloss = fetch_mat_float_texture(
                        mat_config,
                        scene_global,
                        "clearcoat_gloss",
                        Some(1.0),
                    );
                    let spec_trans =
                        fetch_mat_float_texture(mat_config, scene_global, "spec_trans", Some(0.0));
                    let scatter_distance = fetch_mat_rgb_texture(
                        mat_config,
                        scene_global,
                        "scatter_distance",
                        Some(0.0),
                    );
                    let thin = read_bool(mat_config, "thin");
                    let flatness =
                        fetch_mat_float_texture(mat_config, scene_global, "flatness", Some(0.0));
                    let diff_trans =
                        fetch_mat_float_texture(mat_config, scene_global, "diff_trans", Some(1.0));
                    let bump_map =
                        fetch_mat_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    materials_map.insert(
                        material_name,
                        Arc::new(DisneyMaterial::new(
                            color,
                            metallic,
                            eta,
                            roughness,
                            specular_tint,
                            anisotropic,
                            sheen,
                            sheen_tint,
                            clearcoat,
                            clearcoat_gloss,
                            spec_trans,
                            scatter_distance,
                            thin,
                            flatness,
                            diff_trans,
                            bump_map,
                        )),
                    );
                }
                _ => {
                    eprintln!("Unsupported Material Type {}", material_type)
                }
            }
        }
    }
    materials_map
}

fn make_triangle_mesh(
    scene_config: &Value,
    scene_global: &SceneGlobalData,
) -> HashMap<String, Vec<Arc<Triangle>>> {
    todo!();
}

fn make_all_lights(
    scene_config: &Value,
    scene_global: &SceneGlobalData,
) -> (Vec<Arc<dyn Light>>, Vec<Arc<dyn Light>>) {
    let lights_config = search_object(scene_config, "lights");
    let mut lights_len = 0;
    let mut lights;
    let mut infinite_lights;
    if let Ok(Value::Array(a_l)) = lights_config {
        lights_len = a_l.len();
        lights = Vec::with_capacity(lights_len);
        for light_config in a_l {
            lights.push(make_light(light_config, scene_global));
        }
    } else {
        lights = vec![];
    }
    let infinite_lights_config = search_object(scene_config, "infinite_lights");
    let mut inf_lights_len = 0;
    if let Ok(Value::Array(a_l)) = infinite_lights_config {
        inf_lights_len = a_l.len();
        infinite_lights = Vec::with_capacity(inf_lights_len);
        for light_config in a_l {
            infinite_lights.push(make_light(light_config, scene_global));
        }
    } else {
        infinite_lights = vec![];
    }
    if lights_len + inf_lights_len == 0 {
        eprintln!("No lights found!")
    }

    return (lights, infinite_lights);
}

fn make_light(light_config: &Value, scene_global: &SceneGlobalData) -> Arc<dyn Light> {
    assert!(light_config.is_object());
    if let Some(Value::String(light_type)) = light_config.get("light_type") {
        let world_pos: Point3f = make_xyz(light_config.get("world_pos").unwrap()).unwrap();
        let light_to_world = Transform::translate(&(Point3f::zero() - world_pos));
        let mut inside_medium = None;
        let mut outside_medium = None;
        if let Some(all_medium_config) = light_config.get("medium") {
            if let Some(inside_medium_config) = all_medium_config.get("inside") {
                if let Ok(m) = make_medium(inside_medium_config) {
                    inside_medium = Some(m);
                }
            }
            if let Some(inside_medium_config) = all_medium_config.get("outside") {
                if let Ok(m) = make_medium(inside_medium_config) {
                    outside_medium = Some(m);
                }
            }
        }

        let mi = MediumInterface {
            inside: inside_medium,
            outside: outside_medium,
        };
        match light_type.as_str() {
            "point" => {
                let i = make_spectrum(light_config.get("spectrum").unwrap());
                let pl = PointLight::new(light_to_world, mi, Point3f::default(), i);
                return Arc::new(pl);
            }
            "diffuse" => {
                let lemit = make_spectrum(light_config.get("spectrum").unwrap());
                let n_samples = read_i64(light_config, "n_samples") as usize;
                let area = read_f64(light_config, "area");
                let s: Arc<dyn Shape>;
                if let Some(shape_config) = light_config.get("light_shape") {
                    s = make_light_shape(shape_config, scene_global);
                    return Arc::new(DiffuseAreaLight::new(
                        light_to_world,
                        mi,
                        n_samples,
                        lemit,
                        s,
                        area,
                    ));
                } else {
                    panic!("Shape Required for a DiffuseLight! {:?}", light_config)
                }
            }
            _ => {
                panic!("Failed to parse light {:?}", light_type);
            }
        }
    }
    panic!("Failed to parse light {:?}", light_config)
}

fn make_spectrum(spectrum_config: &Value) -> Spectrum<SPECTRUM_N> {
    if let Some(Value::String(spectrum_type)) = spectrum_config.get("spectrum_type") {
        match spectrum_type.as_str() {
            "RGB" => {
                if let Some(rgb) = spectrum_config.get("values") {
                    match read_num_array(rgb, 3) {
                        Ok(rgb_value) => {
                            return Spectrum::new([rgb_value[0], rgb_value[1], rgb_value[2]]);
                        }
                        Err(e) => {
                            panic!(
                                "Failed to parse Spectrum {:?} with error {}",
                                spectrum_config, e
                            )
                        }
                    }
                }
            }
            _ => {}
        }
    }
    panic!("Failed to parse Spectrum {:?}", spectrum_config)
}

fn make_light_shape(shape_config: &Value, scene_global: &SceneGlobalData) -> Arc<dyn Shape> {
    if let Some(Value::String(shape_type)) = shape_config.get("shape_type") {
        match shape_type.as_str() {
            "sphere" => {
                return Arc::new(make_sphere(shape_config));
            }
            "triangle" => {
                let obj_name = shape_config.get("obj_name").unwrap().as_str().unwrap();
                let mesh = scene_global.triangle_mesh[obj_name].clone();
                let tri_num = read_i64(shape_config, "tri_num") as usize;

                return mesh[tri_num].clone();
            }
            _ => {}
        }
    }
    panic!("Failed to parse a Shape from {:?}", shape_config)
}

fn make_sphere(sphere_config: &Value) -> Sphere {
    let world_pos: Point3f = make_xyz(sphere_config.get("world_pos").unwrap()).unwrap();
    let to_world = Transform::translate(&(Point3f::zero() - world_pos));
    let to_local = Transform::inverse(&to_world);

    let radius = read_f64(sphere_config, "radius");
    let z_min = read_f64(sphere_config, "z_min");
    let z_max = read_f64(sphere_config, "z_max");
    let phi_max = read_f64(sphere_config, "phi_max");
    Sphere::new(to_world, to_local, radius, z_min, z_max, phi_max)
}

fn make_medium(medium_config: &Value) -> Result<Arc<dyn Medium + Send + Sync>, String> {
    let parse_err = "Failed to parse medium";
    if let Some(Value::String(medium_type)) = medium_config.get("medium_type") {
        let world_pos: Point3f = make_xyz(medium_config.get("world_pos").ok_or(parse_err)?)?;
        let world_to_medium =
            Transform::inverse(&Transform::translate(&(Point3f::zero() - world_pos)));
        match medium_type.as_str() {
            "GridDensity" => {
                let g = read_f64(medium_config, "g");
                let nx = read_i64(medium_config, "nx") as i32;
                let ny = read_i64(medium_config, "nx") as i32;
                let nz = read_i64(medium_config, "nx") as i32;
                let den_len = (nx * ny * nz) as usize;
                let d = read_num_array(medium_config.get("d").ok_or(parse_err)?, den_len)?;
                let sigma_a = make_spectrum(medium_config.get("sigma_a").ok_or(parse_err)?);
                let sigma_s = make_spectrum(medium_config.get("sigma_s").ok_or(parse_err)?);
                return Ok(Arc::new(GridDensityMedium::new(
                    sigma_a,
                    sigma_s,
                    g,
                    nx,
                    ny,
                    nz,
                    world_to_medium,
                    &d,
                )));
            }
            "Homogeneous" => {
                let g = read_f64(medium_config, "g");
                let sigma_a = make_spectrum(medium_config.get("sigma_a").ok_or(parse_err)?);
                let sigma_s = make_spectrum(medium_config.get("sigma_s").ok_or(parse_err)?);
                let sigma_t = make_spectrum(medium_config.get("sigma_t").ok_or(parse_err)?);
                return Ok(Arc::new(HomogeneousMedium::new(
                    sigma_a, sigma_s, sigma_t, g,
                )));
            }
            _ => return Err(format!("Unsupported medium type: {}", medium_type)),
        }
    } else {
        return Err(format!("Unsupported medium type: {:?}", medium_config));
    }
}

fn make_aggregate(scene_config: &Value, scene_global: &SceneGlobalData) -> Arc<dyn Primitive> {
    todo!();
}

fn make_integrator(
    scene_config: &Value,
    scene_global: &SceneGlobalData,
    save_to: &str,
) -> Box<dyn Integrator> {
    todo!();
}

pub fn write_image(filename: &str, rgb: &[f64], output_bounds: Bounds2i) -> Result<(), ImageError> {
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::geometry::Normal3f;
    #[test]
    fn test_make_xyz() {
        let v = json!([2, 4, 6]);
        let n: Normal3f = make_xyz(&v).unwrap();
        println!("{:?}", n);
    }
}
