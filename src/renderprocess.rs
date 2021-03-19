use image::{io::Reader as ImageReader, DynamicImage, ImageBuffer, ImageError, Rgba};
use serde_json::Value;

use std::{collections::HashMap, fs, sync::Arc};

use crate::{
    geometry::{Bounds2i, Cxyz, Point2, Point3f, Vector3f},
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
    medium::{
        grid::GridDensityMedium, homogeneous::HomogeneousMedium, Medium, MediumInterface,
        SUBSURFACE_PARAMETER_TABLE,
    },
    mipmap::{ImageWrap, MIPMap},
    misc::{clamp_t, gamma_correct},
    primitives::Primitive,
    scene::Scene,
    shape::{sphere::Sphere, triangle::Triangle, Shape},
    spectrum::{ISpectrum, Spectrum, SpectrumType},
    texture::{
        imagemap::{ImageTexture, TexInfo},
        uv::UVTexture,
        ConstantTexture, CylindricalMapping2D, PlanarMapping2D, SphericalMapping2D, Texture,
        TextureMapping2D, UVMapping2D,
    },
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
fn read_i64(root: &Value, key: &str, default_value: i64) -> i64 {
    if let Some(Value::Number(value)) = root.get(key) {
        if let Some(i) = value.as_i64() {
            return i;
        }
    }
    default_value
}

fn read_f64(root: &Value, key: &str, default_value: f64) -> f64 {
    if let Some(Value::Number(value)) = root.get(key) {
        if let Some(i) = value.as_f64() {
            return i;
        }
    }
    default_value
}

fn read_bool(root: &Value, key: &str, default_value: bool) -> bool {
    if let Some(Value::Bool(value)) = root.get(key) {
        return *value;
    }
    default_value
}

fn read_string<T>(root: &Value, key: &str, default_value: T) -> String
where
    T: Into<String>,
{
    if let Some(Value::String(value)) = root.get(key) {
        return String::from(value);
    }
    default_value.into()
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

fn fetch_point3f(config: &Value, key: &str, default_value: Point3f) -> Point3f {
    if let Some(values) = config.get(key) {
        match make_xyz(values) {
            Ok(p) => return p,
            Err(_) => {}
        }
    }
    default_value
}

fn fetch_vector3f(config: &Value, key: &str, default_value: Vector3f) -> Vector3f {
    if let Some(values) = config.get(key) {
        match make_xyz(values) {
            Ok(p) => return p,
            Err(_) => {}
        }
    }
    default_value
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
    let mut images = HashMap::<TexInfo, Arc<MIPMap>>::new();

    let mut float_texture = HashMap::<String, Arc<dyn Texture<f64>>>::new();
    let mut rgb_texture = HashMap::<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>::new();
    if let Some(Value::Array(float_texture_configs)) = scene_config.get("float_texture") {
        for texture_config in float_texture_configs {
            let world_pos = fetch_point3f(texture_config, "world_pos", Point3f::zero());
            let to_world = Transform::translate(&(Point3f::zero() - world_pos));
            let texture_type = read_string(texture_config, "texture_type", "");
            let texture_name = read_string(texture_config, "texture_name", "DefaultTextureName");
            match texture_type.as_str() {
                _ => {
                    eprintln!("Unsupported Texture Type {}", texture_type)
                }
            }
        }
    }
    if let Some(Value::Array(rgb_texture_configs)) = scene_config.get("rgb_texture") {
        for texture_config in rgb_texture_configs {
            let world_pos = fetch_point3f(texture_config, "world_pos", Point3f::zero());
            let to_world = Transform::translate(&(Point3f::zero() - world_pos));
            let texture_type = read_string(texture_config, "texture_type", "");
            let texture_name = read_string(texture_config, "texture_name", "DefaultTextureName");
            match texture_type.as_str() {
                "UVTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    rgb_texture.insert(texture_name, Arc::new(UVTexture::new(mapping)));
                }
                "ImageTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    let tex_info = make_tex_info(texture_config);
                    if images.contains_key(&tex_info) {
                        let m = images[&tex_info].clone();
                        rgb_texture.insert(texture_name, Arc::new(ImageTexture::new(mapping, m)));
                    } else {
                        if let Ok(loaded_m) = load_image(&tex_info) {
                            rgb_texture.insert(
                                texture_name,
                                Arc::new(ImageTexture::new(mapping, loaded_m)),
                            );
                        }
                    }
                }
                _ => {
                    eprintln!("Unsupported Texture Type {}", texture_type)
                }
            }
        }
    }
    (float_texture, rgb_texture)
}

fn make_tex_info(config: &Value) -> TexInfo {
    let filename = read_string(config, "filename", "DefaultTexture");
    let do_trilinear = read_bool(config, "do_trilinear", false);
    let max_aniso = read_f64(config, "max_aniso", 8.0);
    let wrap_mode = match read_string(config, "wrap", "repeat").as_str() {
        "black" => ImageWrap::Black,
        "clamp" => ImageWrap::Clamp,
        _ => ImageWrap::Repeat,
    };
    let scale = read_f64(config, "scale", 1.0);
    let gamma = read_bool(config, "gamma", filename.ends_with("png"));

    TexInfo::new(filename, do_trilinear, max_aniso, wrap_mode, scale, gamma)
}

fn load_image(t: &TexInfo) -> Result<Arc<MIPMap>, String> {
    match ImageReader::open(t.filename.as_str()) {
        Ok(b) => match b.decode() {
            Ok(dy_img) => match dy_img.as_rgb8() {
                Some(img) => {
                    let width = img.width();
                    let height = img.height();
                    let res = Point2::<usize>::new(width as usize, height as usize);
                    let mut rgb_vec: Vec<Spectrum<SPECTRUM_N>> = Vec::with_capacity(res.x * res.y);
                    for y in 0..height {
                        for x in 0..width {
                            let tmp_pixel = img.get_pixel(x, y);
                            let tmp = [
                                tmp_pixel[0] as f64 / 255.0,
                                tmp_pixel[1] as f64 / 255.0,
                                tmp_pixel[2] as f64 / 255.0,
                            ];
                            rgb_vec.push(Spectrum::from_rgb(tmp, SpectrumType::Reflectance));
                        }
                    }
                    return Ok(Arc::new(MIPMap::create(
                        res,
                        &rgb_vec,
                        t.do_trilinear,
                        t.max_aniso,
                        t.wrap_mode,
                    )));
                }
                None => return Err(format!("Failed to load Image {}", t.filename)),
            },
            Err(e) => return Err(format!("Failed to load Image {} {:?}", t.filename, e)),
        },
        Err(e) => return Err(format!("Failed to load Image {} {:?}", t.filename, e)),
    }
}

fn make_texture_mapping_2d(
    mapping_config_opt: Option<&Value>,
    to_world: &Transform,
) -> Box<dyn TextureMapping2D> {
    match mapping_config_opt {
        Some(mapping_config) => {
            let mapping_type = read_string(mapping_config, "mapping", "uv");
            match mapping_type.as_str() {
                "uv" => {
                    let su = read_f64(mapping_config, "su", 1.0);
                    let sv = read_f64(mapping_config, "sv", 1.0);
                    let du = read_f64(mapping_config, "du", 1.0);
                    let dv = read_f64(mapping_config, "dv", 1.0);
                    return Box::new(UVMapping2D::new(su, sv, du, dv));
                }
                "spherical" => {
                    return Box::new(SphericalMapping2D::new(Transform::inverse(to_world)));
                }
                "cylindrical" => {
                    return Box::new(CylindricalMapping2D::new(Transform::inverse(to_world)));
                }
                "planar" => {
                    return Box::new(PlanarMapping2D::new(
                        fetch_vector3f(mapping_config, "v1", Vector3f::new(1.0, 0.0, 0.0)),
                        fetch_vector3f(mapping_config, "v2", Vector3f::new(0.0, 1.0, 0.0)),
                        read_f64(mapping_config, "udelta", 0.0),
                        read_f64(mapping_config, "vdelta", 0.0),
                    ));
                }
                _ => {
                    panic!(
                        "Unsupported Mapping Type {}, {:?}",
                        mapping_type, mapping_config
                    )
                }
            }
        }
        None => {
            return Box::new(UVMapping2D::new(1.0, 1.0, 0.0, 0.0));
        }
    }
}

fn fetch_float_texture(
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

fn fetch_float_texture_opt(
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

fn fetch_rgb_texture<T>(
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
            let material_type = read_string(mat_config, "material_type", "");
            let material_name = String::from(read_string(
                mat_config,
                "material_name",
                "DefaultMaterialName",
            ));
            match material_type.as_str() {
                "MixMaterial" => {
                    let mat1_name = read_string(mat_config, "mat1", "");
                    let mat2_name = read_string(mat_config, "mat2", "");
                    if materials_map.contains_key(mat1_name.as_str())
                        && materials_map.contains_key(mat2_name.as_str())
                    {
                        let scale = fetch_rgb_texture(mat_config, scene_global, "scale", Some(0.5));
                        materials_map.insert(
                            material_name,
                            Arc::new(MixMaterial::new(
                                scene_global.materials[mat1_name.as_str()].clone(),
                                scene_global.materials[mat1_name.as_str()].clone(),
                                scale,
                            )),
                        );
                    }
                }
                "TranslucentMaterial" => {
                    let kd = fetch_rgb_texture(mat_config, scene_global, "kd", Some(0.25));
                    let ks = fetch_rgb_texture(mat_config, scene_global, "ks", Some(0.25));
                    let roughness =
                        fetch_float_texture(mat_config, scene_global, "roughness", Some(0.1));
                    let reflect =
                        fetch_rgb_texture(mat_config, scene_global, "reflect", Some(0.25));
                    let transmit =
                        fetch_rgb_texture(mat_config, scene_global, "transmit", Some(0.25));
                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness", false);

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
                    let eta = fetch_rgb_texture(mat_config, scene_global, "eta", Some(*COPPER_N));
                    let k = fetch_rgb_texture(mat_config, scene_global, "k", Some(*COPPER_K));
                    let roughness =
                        fetch_float_texture(mat_config, scene_global, "roughness", Some(0.01));
                    let u_roughness =
                        fetch_float_texture_opt(mat_config, scene_global, "u_roughness", None);
                    let v_roughness =
                        fetch_float_texture_opt(mat_config, scene_global, "v_roughness", None);
                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness", false);
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
                    let kd = fetch_rgb_texture(mat_config, scene_global, "kd", Some(0.25));
                    let ks = fetch_rgb_texture(mat_config, scene_global, "ks", Some(0.25));
                    let roughness =
                        fetch_float_texture(mat_config, scene_global, "roughness", Some(0.1));
                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness", false);
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
                    let kr = fetch_rgb_texture(mat_config, scene_global, "kr", Some(0.9));
                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    materials_map
                        .insert(material_name, Arc::new(MirrorMaterial::new(kr, bump_map)));
                }
                "GlassMaterial" => {
                    let kr = fetch_rgb_texture(mat_config, scene_global, "kr", Some(1.0));
                    let kt = fetch_rgb_texture(mat_config, scene_global, "kt", Some(1.0));
                    let eta = fetch_float_texture(mat_config, scene_global, "eta", Some(1.5));

                    let u_roughness =
                        fetch_float_texture(mat_config, scene_global, "u_roughness", Some(0.0));
                    let v_roughness =
                        fetch_float_texture(mat_config, scene_global, "v_roughness", Some(0.0));

                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    let remap_roughness = read_bool(mat_config, "remap_roughness", false);

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
                    let kd = fetch_rgb_texture(mat_config, scene_global, "kd", Some(0.5));
                    let sigma = fetch_float_texture(mat_config, scene_global, "sigma", Some(0.0));

                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
                    materials_map.insert(
                        material_name,
                        Arc::new(MatteMaterial::new(kd, sigma, bump_map)),
                    );
                }
                "DisneyMaterial" => {
                    let color = fetch_rgb_texture(mat_config, scene_global, "color", Some(0.5));
                    let metallic =
                        fetch_float_texture(mat_config, scene_global, "metallic", Some(0.0));
                    let eta = fetch_float_texture(mat_config, scene_global, "eta", Some(1.5));
                    let roughness =
                        fetch_float_texture(mat_config, scene_global, "roughness", Some(0.5));
                    let specular_tint =
                        fetch_float_texture(mat_config, scene_global, "specular_tint", Some(0.0));
                    let anisotropic =
                        fetch_float_texture(mat_config, scene_global, "anisotropic", Some(0.0));
                    let sheen = fetch_float_texture(mat_config, scene_global, "sheen", Some(0.0));
                    let sheen_tint =
                        fetch_float_texture(mat_config, scene_global, "sheen_tint", Some(0.5));
                    let clearcoat =
                        fetch_float_texture(mat_config, scene_global, "clearcoat", Some(0.0));
                    let clearcoat_gloss =
                        fetch_float_texture(mat_config, scene_global, "clearcoat_gloss", Some(1.0));
                    let spec_trans =
                        fetch_float_texture(mat_config, scene_global, "spec_trans", Some(0.0));
                    let scatter_distance =
                        fetch_rgb_texture(mat_config, scene_global, "scatter_distance", Some(0.0));
                    let thin = read_bool(mat_config, "thin", false);
                    let flatness =
                        fetch_float_texture(mat_config, scene_global, "flatness", Some(0.0));
                    let diff_trans =
                        fetch_float_texture(mat_config, scene_global, "diff_trans", Some(1.0));
                    let bump_map =
                        fetch_float_texture_opt(mat_config, scene_global, "bump_map", None);
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
        let world_pos = fetch_point3f(light_config, "world_pos", Point3f::zero());

        let light_to_world = Transform::translate(&(Point3f::zero() - world_pos));
        let mut inside_medium = None;
        let mut outside_medium = None;
        if let Some(all_medium_config) = light_config.get("medium") {
            if let Some(inside_medium_config) = all_medium_config.get("inside") {
                inside_medium = make_medium(inside_medium_config).ok();
            }
            if let Some(inside_medium_config) = all_medium_config.get("outside") {
                outside_medium = make_medium(inside_medium_config).ok();
            }
        }

        let mi = MediumInterface {
            inside: inside_medium,
            outside: outside_medium,
        };
        match light_type.as_str() {
            "point" => {
                let i = make_spectrum(light_config, "spectrum", 1.0);
                let pl = PointLight::new(light_to_world, mi, Point3f::default(), i);
                return Arc::new(pl);
            }
            "diffuse" => {
                let lemit = make_spectrum(light_config, "spectrum", 1.0);
                let n_samples = read_i64(light_config, "n_samples", 1) as usize;
                let s: Arc<dyn Shape>;
                if let Some(shape_config) = light_config.get("light_shape") {
                    s = make_light_shape(shape_config, scene_global);
                    let area = s.area();
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

fn make_spectrum<T>(config: &Value, key: &str, default_value: T) -> Spectrum<SPECTRUM_N>
where
    T: Into<Spectrum<SPECTRUM_N>> + Copy + Clone,
{
    if let Some(spectrum_config) = config.get(key) {
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

    default_value.into()
}

fn make_light_shape(shape_config: &Value, scene_global: &SceneGlobalData) -> Arc<dyn Shape> {
    if let Some(Value::String(shape_type)) = shape_config.get("shape_type") {
        match shape_type.as_str() {
            "sphere" => {
                return Arc::new(make_sphere(shape_config));
            }
            "triangle" => {
                let obj_name = read_string(shape_config, "obj_name", "");
                let mesh = scene_global.triangle_mesh[obj_name.as_str()].clone();
                let tri_num = read_i64(shape_config, "tri_num", 0) as usize;

                return mesh[tri_num].clone();
            }
            _ => {}
        }
    }
    panic!("Failed to parse a Shape from {:?}", shape_config)
}

fn make_sphere(sphere_config: &Value) -> Sphere {
    let world_pos = fetch_point3f(sphere_config, "world_pos", Point3f::zero());
    let to_world = Transform::translate(&(Point3f::zero() - world_pos));
    let to_local = Transform::inverse(&to_world);

    let radius = read_f64(sphere_config, "radius", 1.0);
    let z_min = read_f64(sphere_config, "z_min", -radius);
    let z_max = read_f64(sphere_config, "z_max", radius);
    let phi_max = read_f64(sphere_config, "phi_max", 360.0);
    Sphere::new(to_world, to_local, radius, z_min, z_max, phi_max)
}

pub fn get_medium_scattering_properties(
    medium_config: &Value,
) -> (Spectrum<SPECTRUM_N>, Spectrum<SPECTRUM_N>) {
    let mut sigma_a = Spectrum::<SPECTRUM_N>::zero();
    let mut sigma_prime_s = Spectrum::<SPECTRUM_N>::zero();

    let mut found = false;
    if let Some(Value::String(preset_name)) = medium_config.get("preset") {
        for mss in SUBSURFACE_PARAMETER_TABLE.iter() {
            if preset_name == mss.name {
                sigma_a = Spectrum::from_rgb(mss.sigma_a, SpectrumType::Reflectance);
                sigma_prime_s = Spectrum::from_rgb(mss.sigma_prime_s, SpectrumType::Reflectance);
                found = true;
            }
        }
    }

    if !found {
        let sig_a_rgb: [f64; 3] = [0.0011, 0.0024, 0.014];
        let sig_s_rgb: [f64; 3] = [2.55, 3.21, 3.77];

        sigma_a = Spectrum::from_rgb(sig_a_rgb, SpectrumType::Reflectance);
        sigma_prime_s = Spectrum::from_rgb(sig_s_rgb, SpectrumType::Reflectance);
    }
    return (sigma_a, sigma_prime_s);
}

fn make_medium(medium_config: &Value) -> Result<Arc<dyn Medium + Send + Sync>, String> {
    let parse_err = "Failed to parse medium";
    if let Some(Value::String(medium_type)) = medium_config.get("medium_type") {
        let world_pos = fetch_point3f(medium_config, "world_pos", Point3f::zero());
        let world_to_medium =
            Transform::inverse(&Transform::translate(&(Point3f::zero() - world_pos)));
        let (sigma_a, sigma_s) = get_medium_scattering_properties(medium_config);
        let g = read_f64(medium_config, "g", 0.0);

        match medium_type.as_str() {
            "GridDensity" => {
                let nx = read_i64(medium_config, "nx", 1) as i32;
                let ny = read_i64(medium_config, "nx", 1) as i32;
                let nz = read_i64(medium_config, "nx", 1) as i32;
                let den_len = (nx * ny * nz) as usize;
                let d = read_num_array(medium_config.get("d").ok_or(parse_err)?, den_len)?;

                let p0 = fetch_point3f(medium_config, "p0", Point3f::new(0.0, 0.0, 0.0));
                let p1 = fetch_point3f(medium_config, "p0", Point3f::new(1.0, 1.0, 1.0));

                let data2medium = Transform::translate(&p0.into())
                    * Transform::scale(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
                return Ok(Arc::new(GridDensityMedium::new(
                    sigma_a,
                    sigma_s,
                    g,
                    nx,
                    ny,
                    nz,
                    world_to_medium * data2medium,
                    &d,
                )));
            }
            "Homogeneous" => {
                return Ok(Arc::new(HomogeneousMedium::new(sigma_a, sigma_s, g)));
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
