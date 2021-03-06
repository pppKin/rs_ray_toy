use image::{io::Reader as ImageReader, ImageBuffer, ImageError, Rgba};
use serde_json::Value;

use std::{collections::HashMap, error::Error, f64::INFINITY, fmt::Display, fs, sync::Arc};

use crate::{
    bvh::{BVHAccel, BVHSplitMethod},
    camera::RealisticCamera,
    film::Film,
    filters::{
        boxfilter::create_box_filter, gaussian::create_gaussian_filter,
        trianglefilter::create_triangle_filter, Filter,
    },
    geometry::{
        Bounds2f, Bounds2i, Bounds3f, Cxyz, Point2, Point2f, Point2i, Point3f, Vector2f, Vector3f,
    },
    integrator::{
        ao::*, directlighting::*, intersect_debug::*, path::*, sppm::*, volpath::*, Integrator,
        SamplerIntegratorData,
    },
    lights::{
        diffuse::DiffuseAreaLight, distant::DistantLight, infinite::InfiniteAreaLight,
        point::PointLight, Light,
    },
    material::{
        debug_material::DebugMaterial,
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
    objparser::parse_obj,
    primitives::{GeometricPrimitive, Primitive, TransformedPrimitive},
    samplers::{halton::HaltonBuilder, stratified::StratifiedBuilder, SamplerBuilder},
    scene::Scene,
    shape::{
        sphere::Sphere,
        triangle::{create_triangle_mesh, Triangle},
        Shape,
    },
    spectrum::{ISpectrum, Spectrum, SpectrumType},
    texture::{
        bilerp::BilerpTexture,
        checkerboard::{AAMethod, Checkerboard2DTexture, Checkerboard3DTexture},
        imagemap::{ImageTexture, TexInfo},
        mix::MixTexture,
        scale::ScaleTexture,
        uv::UVTexture,
        windy::WindyTexture,
        wrinkled::WrinkledTexture,
        ConstantTexture, CylindricalMapping2D, IdentityMapping3D, PlanarMapping2D,
        SphericalMapping2D, Texture, TextureMapping2D, UVMapping2D,
    },
    transform::Transform,
    SPECTRUM_N,
};

#[derive(Debug, Default)]
struct RenderProcessError {
    msg: String,
}

impl Display for RenderProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error During Render Process {}", self.msg)
    }
}

impl Error for RenderProcessError {}

struct SceneGlobalData {
    scene_config_root: String,
    float_texture: HashMap<String, Arc<dyn Texture<f64>>>,
    rgb_texture: HashMap<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>,

    materials: HashMap<String, Arc<dyn Material>>,
    triangle_mesh: HashMap<String, Vec<Arc<Triangle>>>,
}

/// read scene config file (json), create required resources, and start rendering
pub fn deploy_render(filepath: &str, save_to: &str) {
    let scene_config_root;
    let realpath = fs::canonicalize(filepath).unwrap();
    if let Some(root_path) = realpath.parent() {
        scene_config_root = String::from(root_path.to_str().unwrap());
    } else {
        scene_config_root = "./".to_string();
    }
    let scene_config_str = fs::read_to_string(realpath).unwrap();
    let scene_config: Value = serde_json::from_str(&scene_config_str).unwrap();
    let scene = make_scene(&scene_config, &scene_config_root);
    let mut inte = make_integrator(&scene_config, save_to);
    inte.render(&scene);
}

#[cfg(target_os = "windows")]
fn preprocess_filepath(root: &str, passedin: &str) -> String {
    let trimmed_passedin;
    trimmed_passedin = passedin.trim_start_matches("./").replace("/", "\\");
    format!("{}\\{}", root, trimmed_passedin)
}

#[cfg(not(target_os = "windows"))]
fn preprocess_filepath(root: &str, passedin: &str) -> String {
    let trimmed_passedin;
    // #[cfg(target_os = "windows")]
    trimmed_passedin = passedin.trim_start_matches("./").replace("\\", "/");
    format!("{}/{}", root, trimmed_passedin)
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

fn read_num_array(v: &Value, desired_length: usize) -> Result<Vec<f64>, String> {
    if let Value::Array(ary) = v {
        let length = if desired_length == 0 {
            ary.len()
        } else {
            desired_length
        };
        if ary.len() != length {
            return Err(format!(
                "Failed to parse Transform from {:?}, expected and array of {} numbers",
                ary, length
            ));
        }
        let mut a = Vec::with_capacity(length);
        for num in ary {
            if num.is_number() {
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

fn fetch_vector2f(config: &Value, key: &str, default_value: Vector2f) -> Vector2f {
    if let Some(root) = config.get(key) {
        if let Ok(nums) = read_num_array(root, 2) {
            return Vector2f::new(nums[0], nums[1]);
        }
    }
    default_value
}

fn make_to_world(root: &Value) -> Transform {
    let world_pos = fetch_point3f(root, "world_pos", Point3f::zero());
    let rotation_axis =
        fetch_vector3f(root, "rotation_axis", Vector3f::new(0.0, 0.0, 0.0)).normalize();
    let rotation_angle = read_f64(root, "rotation_angle", 0.0);
    let scale = fetch_vector3f(root, "scale", Vector3f::new(1.0, 1.0, 1.0));
    let to_world = Transform::translate(&(world_pos.into()))
        * Transform::rotate(rotation_angle, &rotation_axis)
        * Transform::scale(scale.x, scale.y, scale.z);
    to_world
}

fn make_scene(scene_config: &Value, scene_config_root: &str) -> Scene {
    // material and triangle mesh is used globally so we create them first and passed them around
    let (float_texture, rgb_texture) = make_textures(scene_config, scene_config_root);
    let mut scene_global_data = SceneGlobalData {
        scene_config_root: scene_config_root.to_string(),
        float_texture,
        rgb_texture,
        materials: HashMap::new(),
        triangle_mesh: HashMap::new(),
    };
    scene_global_data.materials = make_materials(&scene_config, &scene_global_data);
    scene_global_data.triangle_mesh = make_triangle_mesh(&scene_config, &scene_global_data);

    let aggregate = make_aggregate(&scene_config, &scene_global_data);
    eprintln!(
        "Generated aggregate with world bound {:?}",
        aggregate.world_bound()
    );
    let (lights, infinite_lights) = make_all_lights(
        &scene_config,
        &scene_global_data,
        scene_config_root,
        aggregate.world_bound(),
    );

    Scene::new(lights, infinite_lights, aggregate)
}

fn get_text_fallback<T: 'static>(
    source: &HashMap<String, Arc<dyn Texture<T>>>,
    tex_key: String,
    default_value: T,
) -> Arc<dyn Texture<T>>
where
    ConstantTexture<T>: Texture<T>,
    T: Send + Sync + Copy,
{
    if source.contains_key(&tex_key) {
        return source[&tex_key].clone();
    } else {
        return Arc::new(ConstantTexture::new(default_value));
    }
}

fn make_textures(
    scene_config: &Value,
    scene_config_root: &str,
) -> (
    HashMap<String, Arc<dyn Texture<f64>>>,
    HashMap<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>,
) {
    let mut images = HashMap::<TexInfo, Arc<MIPMap>>::new();

    let mut float_texture = HashMap::<String, Arc<dyn Texture<f64>>>::new();
    let mut rgb_texture = HashMap::<String, Arc<dyn Texture<Spectrum<SPECTRUM_N>>>>::new();
    if let Some(Value::Array(float_texture_configs)) = scene_config.get("float_texture") {
        for texture_config in float_texture_configs {
            let to_world = make_to_world(texture_config);
            let texture_type = read_string(texture_config, "texture_type", "");
            let texture_name = read_string(texture_config, "texture_name", "DefaultTextureName");
            match texture_type.as_str() {
                "MixTexture" => {
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let amount_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&float_texture, t1_name, 0.0);
                    let t2 = get_text_fallback(&float_texture, t2_name, 1.0);
                    let amount = get_text_fallback(&float_texture, amount_name, 0.5);
                    float_texture.insert(texture_name, Arc::new(MixTexture::new(t1, t2, amount)));
                }
                "BilerpTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    let v00 = read_f64(texture_config, "v00", 0.0);
                    let v01 = read_f64(texture_config, "v01", 1.0);
                    let v10 = read_f64(texture_config, "v01", 0.0);
                    let v11 = read_f64(texture_config, "v01", 1.0);
                    float_texture.insert(
                        texture_name,
                        Arc::new(BilerpTexture::new(mapping, v00, v01, v10, v11)),
                    );
                }
                "CheckerBoardTexture" => {
                    let dim = read_i64(texture_config, "dimension", 2);
                    if dim != 2 && dim != 3 {
                        eprintln!("{} dimensional checkerboard texture not supported", dim);
                        continue;
                    }
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&float_texture, t1_name, 1.0);
                    let t2 = get_text_fallback(&float_texture, t2_name, 0.0);

                    if dim == 2 {
                        let mapping =
                            make_texture_mapping_2d(texture_config.get("mapping"), &to_world);

                        // Compute _aaMethod_ for _CheckerboardTexture_
                        let aa = read_string(texture_config, "aamode", "closedform");
                        let aa_method;
                        match aa.as_str() {
                            "none" => {
                                aa_method = AAMethod::AANone;
                            }
                            _ => {
                                aa_method = AAMethod::ClosedForm;
                            }
                        }
                        float_texture.insert(
                            texture_name,
                            Arc::new(Checkerboard2DTexture::new(mapping, t1, t2, aa_method)),
                        );
                    } else {
                        // Initialize 3D texture mapping _map_ from _tp_
                        let mapping = IdentityMapping3D::new(to_world);
                        //     return new Checkerboard3DTexture<Float>(std::move(map), tex1, tex2);
                        float_texture.insert(
                            texture_name,
                            Arc::new(Checkerboard3DTexture::new(Box::new(mapping), t1, t2)),
                        );
                    }
                }
                "ScaleTexture" => {
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&float_texture, t1_name, 1.0);
                    let t2 = get_text_fallback(&float_texture, t2_name, 1.0);
                    float_texture.insert(texture_name, Arc::new(ScaleTexture::new(t1, t2)));
                }
                "WindyTexture" => {
                    let mapping = IdentityMapping3D::new(to_world);
                    float_texture
                        .insert(texture_name, Arc::new(WindyTexture::new(Box::new(mapping))));
                }
                "WrinkledTexture" => {
                    let mapping = IdentityMapping3D::new(to_world);
                    let octaves = read_i64(texture_config, "octaves", 8) as u64;
                    let omega = read_f64(texture_config, "omega", 0.5);
                    float_texture.insert(
                        texture_name,
                        Arc::new(WrinkledTexture::new(Box::new(mapping), octaves, omega)),
                    );
                }
                _ => {
                    eprintln!("Unsupported Texture Type {}", texture_type)
                }
            }
        }
    }
    if let Some(Value::Array(rgb_texture_configs)) = scene_config.get("rgb_texture") {
        for texture_config in rgb_texture_configs {
            let to_world = make_to_world(texture_config);
            let texture_type = read_string(texture_config, "texture_type", "");
            let texture_name = read_string(texture_config, "texture_name", "DefaultTextureName");
            match texture_type.as_str() {
                "MixTexture" => {
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let amount_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&rgb_texture, t1_name, Spectrum::from(0.0));
                    let t2 = get_text_fallback(&rgb_texture, t2_name, Spectrum::from(1.0));
                    let amount = get_text_fallback(&float_texture, amount_name, 0.5);
                    rgb_texture.insert(texture_name, Arc::new(MixTexture::new(t1, t2, amount)));
                }
                "UVTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    rgb_texture.insert(texture_name, Arc::new(UVTexture::new(mapping)));
                }
                "ImageTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    let tex_info = make_tex_info(texture_config, scene_config_root);
                    if images.contains_key(&tex_info) {
                        let m = images[&tex_info].clone();
                        rgb_texture.insert(texture_name, Arc::new(ImageTexture::new(mapping, m)));
                    } else {
                        if let Ok(loaded_m) = load_image(&tex_info) {
                            images.insert(tex_info, loaded_m.clone());
                            rgb_texture.insert(
                                texture_name,
                                Arc::new(ImageTexture::new(mapping, loaded_m)),
                            );
                        }
                    }
                }
                "BilerpTexture" => {
                    let mapping = make_texture_mapping_2d(texture_config.get("mapping"), &to_world);
                    let v00 = make_spectrum(texture_config, "v00", 0.0);
                    let v01 = make_spectrum(texture_config, "v01", 1.0);
                    let v10 = make_spectrum(texture_config, "v01", 0.0);
                    let v11 = make_spectrum(texture_config, "v01", 1.0);
                    rgb_texture.insert(
                        texture_name,
                        Arc::new(BilerpTexture::new(mapping, v00, v01, v10, v11)),
                    );
                }
                "CheckerBoardTexture" => {
                    let dim = read_i64(texture_config, "dimension", 2);
                    if dim != 2 && dim != 3 {
                        eprintln!("{} dimensional checkerboard texture not supported", dim);
                        continue;
                    }
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&rgb_texture, t1_name, Spectrum::from(1.0));
                    let t2 = get_text_fallback(&rgb_texture, t2_name, Spectrum::from(0.0));

                    if dim == 2 {
                        let mapping =
                            make_texture_mapping_2d(texture_config.get("mapping"), &to_world);

                        // Compute _aaMethod_ for _CheckerboardTexture_
                        let aa = read_string(texture_config, "aamode", "closedform");
                        let aa_method;
                        match aa.as_str() {
                            "none" => {
                                aa_method = AAMethod::AANone;
                            }
                            _ => {
                                aa_method = AAMethod::ClosedForm;
                            }
                        }
                        rgb_texture.insert(
                            texture_name,
                            Arc::new(Checkerboard2DTexture::new(mapping, t1, t2, aa_method)),
                        );
                    } else {
                        // Initialize 3D texture mapping _map_ from _tp_
                        let mapping = IdentityMapping3D::new(to_world);
                        rgb_texture.insert(
                            texture_name,
                            Arc::new(Checkerboard3DTexture::new(Box::new(mapping), t1, t2)),
                        );
                    }
                }
                "ScaleTexture" => {
                    let t1_name = read_string(texture_config, "t1", "ErrorTextureName");
                    let t2_name = read_string(texture_config, "t2", "ErrorTextureName");
                    let t1 = get_text_fallback(&rgb_texture, t1_name, Spectrum::from(1.0));
                    let t2 = get_text_fallback(&rgb_texture, t2_name, Spectrum::from(1.0));
                    rgb_texture.insert(texture_name, Arc::new(ScaleTexture::new(t1, t2)));
                }
                "WindyTexture" => {
                    let mapping = IdentityMapping3D::new(to_world);
                    rgb_texture
                        .insert(texture_name, Arc::new(WindyTexture::new(Box::new(mapping))));
                }
                "WrinkledTexture" => {
                    let mapping = IdentityMapping3D::new(to_world);
                    let octaves = read_i64(texture_config, "octaves", 8) as u64;
                    let omega = read_f64(texture_config, "omega", 0.5);
                    rgb_texture.insert(
                        texture_name,
                        Arc::new(WrinkledTexture::new(Box::new(mapping), octaves, omega)),
                    );
                }
                _ => {
                    eprintln!("Unsupported Texture Type {}", texture_type)
                }
            }
        }
    }
    (float_texture, rgb_texture)
}

fn make_tex_info(config: &Value, scene_config_root: &str) -> TexInfo {
    let filename = preprocess_filepath(
        scene_config_root,
        &read_string(config, "filename", "DefaultTexture"),
    );
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

fn load_image(t: &TexInfo) -> Result<Arc<MIPMap>, Box<dyn Error>> {
    let img = ImageReader::open(t.filename.as_str())?
        .decode()?
        .into_rgb8();

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
    for y in 0..res.y / 2 {
        for x in 0..res.x {
            let o1 = (y * res.x + x) as usize;
            let o2 = ((res.y - 1 - y) * res.x + x) as usize;
            rgb_vec.swap(o1, o2);
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
    if let Some(Value::String(tex_name)) = mat_config.get(tex_key) {
        if scene_global.rgb_texture.contains_key(tex_name) {
            return scene_global.rgb_texture[tex_name].clone();
        }
    }
    if let Some(d) = default_value {
        return Arc::new(ConstantTexture::new(d.into()));
    }
    panic!("Failed to fetch Texture: {}", tex_key)
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
                "Debug" => {
                    materials_map.insert(material_name, Arc::new(DebugMaterial {}));
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
    let objs_config = search_object(scene_config, "objs");
    let mut objs;
    if let Ok(Value::Array(obj_l)) = objs_config {
        objs = HashMap::with_capacity(obj_l.len());
        for obj_config in obj_l {
            let filename = read_string(obj_config, "filename", "DefaultObj");
            let obj_name = read_string(obj_config, "obj_name", "DefaultObjName");
            let to_world = make_to_world(obj_config);

            let to_local = Transform::inverse(&to_world);
            match parse_obj(&(preprocess_filepath(&scene_global.scene_config_root, &filename))) {
                Ok(result) => {
                    objs.insert(
                        obj_name,
                        create_triangle_mesh(
                            to_world,
                            to_local,
                            result.n_triangles,
                            result.n_vertices,
                            result.vertex_indices,
                            result.normal_indices,
                            result.uv_indices,
                            result.p,
                            result.n,
                            result.s,
                            result.uv,
                        ),
                    );
                }
                Err(err) => {
                    eprintln!(
                        "parse_result {} :: {:?}",
                        preprocess_filepath(&scene_global.scene_config_root, &filename),
                        err
                    );
                }
            }
        }
    } else {
        objs = HashMap::new();
    }
    objs
}

fn make_all_lights(
    scene_config: &Value,
    scene_global: &SceneGlobalData,
    scene_config_root: &str,
    world_bound: Bounds3f,
) -> (Vec<Arc<dyn Light>>, Vec<Arc<dyn Light>>) {
    let lights_config = search_object(scene_config, "lights");
    let mut lights_len = 0;
    let mut lights;
    let mut infinite_lights;
    if let Ok(Value::Array(a_l)) = lights_config {
        lights_len = a_l.len();
        lights = Vec::with_capacity(lights_len);
        for light_config in a_l {
            lights.push(make_light(
                light_config,
                scene_global,
                scene_config_root,
                world_bound,
            ));
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
            infinite_lights.push(make_light(
                light_config,
                scene_global,
                scene_config_root,
                world_bound,
            ));
        }
    } else {
        infinite_lights = vec![];
    }
    if lights_len + inf_lights_len == 0 {
        eprintln!("No lights found!")
    }

    return (lights, infinite_lights);
}

fn make_light(
    light_config: &Value,
    scene_global: &SceneGlobalData,
    scene_config_root: &str,
    world_bound: Bounds3f,
) -> Arc<dyn Light> {
    assert!(light_config.is_object());
    if let Some(Value::String(light_type)) = light_config.get("light_type") {
        let light_to_world = make_to_world(light_config);

        let mut inside_medium = None;
        let mut outside_medium = None;
        if let Some(all_medium_config) = light_config.get("medium_interface") {
            if let Some(inside_medium_config) = all_medium_config.get("inside") {
                inside_medium = make_medium(inside_medium_config).ok();
            }
            if let Some(outside_medium_config) = all_medium_config.get("outside") {
                outside_medium = make_medium(outside_medium_config).ok();
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
            "distant" => {
                let l = make_spectrum(light_config, "l", 1.0);
                let sc = make_spectrum(light_config, "scale", 1.0);
                let from = fetch_point3f(light_config, "from", Point3f::default());
                let to = fetch_point3f(light_config, "to", Point3f::new(0.0, 0.0, 1.0));
                let dir = from - to;
                return Arc::new(DistantLight::new(
                    light_to_world,
                    mi,
                    l * sc,
                    dir,
                    world_bound,
                ));
            }
            "infinite" => {
                let l = make_spectrum(light_config, "l", 1.0);
                let sc = make_spectrum(light_config, "scale", 1.0);
                let texmap = read_string(light_config, "mapname", "");

                let n_samples = read_i64(light_config, "n_samples", 1) as usize;
                return Arc::new(InfiniteAreaLight::new(
                    light_to_world,
                    l * sc,
                    &preprocess_filepath(scene_config_root, &texmap),
                    n_samples,
                    mi,
                    world_bound,
                ));
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
    let to_world = make_to_world(sphere_config);
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
        let to_world = make_to_world(medium_config);

        let world_to_medium = Transform::inverse(&to_world);
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
    let mut primitives: Vec<Arc<dyn Primitive>>;
    let aggregate_config =
        search_object(scene_config, "Aggregate").expect("No Aggregate Config Defined");
    let max_prims_in_node = read_i64(aggregate_config, "max_prims_in_node", 4) as u32;
    if let Some(Value::Array(primitives_vec)) = aggregate_config.get("primitives") {
        primitives = Vec::with_capacity(primitives_vec.len());
        for primitive_config in primitives_vec {
            match read_string(primitive_config, "primitive_type", "").as_str() {
                "sphere" => {
                    let shape = make_sphere(primitive_config);
                    let material_name =
                        read_string(primitive_config, "material_name", "DefaultMaterialName");
                    if let Some(material) = scene_global.materials.get(material_name.as_str()) {
                        // let p_area_light; we'll figure out emissive primitive later?
                        let mut inside_medium = None;
                        let mut outside_medium = None;
                        if let Some(all_medium_config) = primitive_config.get("medium_interface") {
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
                        let g = Arc::new(GeometricPrimitive::new(
                            Arc::new(shape),
                            material.clone(),
                            None,
                            mi,
                        ));
                        if let Some(Value::Array(instances_vec)) = primitive_config.get("instances")
                        {
                            primitives.reserve(instances_vec.len());
                            for instance_config in instances_vec {
                                let to_world = make_to_world(instance_config);

                                primitives
                                    .push(Arc::new(TransformedPrimitive::new(g.clone(), to_world)));
                            }
                        } else {
                            primitives.push(g.clone());
                        }
                    }
                }
                "triangle" => {
                    let obj_name = read_string(primitive_config, "obj_name", "DefaultObjName");
                    let obj_opt = scene_global.triangle_mesh.get(&obj_name);
                    let material_name =
                        read_string(primitive_config, "material_name", "DefaultMaterialName");
                    let material_opt = scene_global.materials.get(material_name.as_str());
                    if let (Some(obj), Some(material)) = (obj_opt, material_opt) {
                        let mut inside_medium = None;
                        let mut outside_medium = None;
                        if let Some(medium_interface_config) =
                            primitive_config.get("medium_interface")
                        {
                            if let Some(inside_medium_config) =
                                medium_interface_config.get("inside")
                            {
                                inside_medium = make_medium(inside_medium_config).ok();
                            }
                            if let Some(outside_medium_config) =
                                medium_interface_config.get("outside")
                            {
                                outside_medium = make_medium(outside_medium_config).ok();
                            }
                        }

                        let mi = MediumInterface {
                            inside: inside_medium,
                            outside: outside_medium,
                        };
                        let mut gs = Vec::with_capacity(obj.len());
                        for tri in obj {
                            gs.push(Arc::new(GeometricPrimitive::new(
                                tri.clone(),
                                material.clone(),
                                None,
                                mi.clone(),
                            )))
                        }

                        if let Some(Value::Array(instances_vec)) = primitive_config.get("instances")
                        {
                            primitives.reserve(instances_vec.len());
                            for instance_config in instances_vec {
                                let to_world = make_to_world(instance_config);
                                for g in &gs {
                                    primitives.push(Arc::new(TransformedPrimitive::new(
                                        g.clone(),
                                        to_world,
                                    )));
                                }
                            }
                        } else {
                            for g in &gs {
                                primitives.push(g.clone());
                            }
                        }
                    } else {
                        eprintln!(
                            "Error creating triangle instances {}:{}",
                            obj_name, material_name
                        )
                    }
                }
                unknown => {
                    eprintln!("Unsupported primitive_type! {}", unknown)
                }
            }
        }
    } else {
        eprintln!("Found 0 primitives! You're rendering nothing!");
        primitives = vec![];
    }
    Arc::new(BVHAccel::new(
        primitives,
        max_prims_in_node,
        BVHSplitMethod::HLBVH,
    ))
}

fn make_sampler(sampler_config: &Value, sample_bounds: &Bounds2i) -> Arc<dyn SamplerBuilder> {
    match read_string(sampler_config, "sampler_type", "").as_str() {
        "StratifiedSampler" => {
            let jitter = read_bool(sampler_config, "jitter", true);
            let xsamp = read_i64(sampler_config, "xsamp", 4) as u32;
            let ysamp = read_i64(sampler_config, "ysamp", 4) as u32;
            let dimension = read_i64(sampler_config, "dimension", 4) as u32;
            Arc::new(StratifiedBuilder::new(dimension, xsamp, ysamp, jitter))
        }
        "HaltonSampler" => {
            // int nsamp = params.FindOneInt("pixelsamples", 16);
            let nsamp = read_i64(sampler_config, "nsamp", 16) as u64;
            let sample_at_center = read_bool(sampler_config, "sample_at_center", false);
            Arc::new(HaltonBuilder::new(nsamp, *sample_bounds, sample_at_center))
        }
        _ => {
            panic!("Unsupported Sampler type: {:?}", sampler_config);
        }
    }
}

fn make_film(film_config: &Value, save_to: &str) -> Film {
    let xres = read_i64(film_config, "xres", 1280);
    let yres = read_i64(film_config, "yres", 720);

    let crop = Bounds2f::new(Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0));
    let scale = read_f64(film_config, "scale", 1.0);
    let diagonal = read_f64(film_config, "diagonal", 35.0); // by default we use 35mm film
    let max_sample_luminance = read_f64(film_config, "max_sample_luminance", INFINITY);
    if let Ok(filter_config) = search_object(film_config, "Filter") {
        let fltr: Filter;
        match read_string(filter_config, "filter_type", "BoxFilter").as_str() {
            "TriangleFilter" => {
                let radius = fetch_vector2f(filter_config, "radius", Vector2f::new(2.0, 2.0));
                fltr = create_triangle_filter(radius);
            }
            "GaussianFilter" => {
                let radius = fetch_vector2f(filter_config, "radius", Vector2f::new(2.0, 2.0));
                let alpha = read_f64(filter_config, "alpha", 2.0);
                fltr = create_gaussian_filter(radius, alpha);
            }
            _ => {
                let radius = fetch_vector2f(filter_config, "radius", Vector2f::new(0.5, 0.5));
                // by default create a box filter
                fltr = create_box_filter(radius);
            }
        }
        return Film::new(
            Point2i::new(xres, yres),
            diagonal,
            fltr,
            save_to.to_string(),
            crop,
            scale,
            max_sample_luminance,
        );
    } else {
        eprintln!("filter_config not found!")
    }
    panic!("Failed to create Film")
}

fn make_camera(camera_config: &Value, film: Film) -> RealisticCamera {
    let world_pos = fetch_point3f(camera_config, "world_pos", Point3f::zero());
    let cam_look = fetch_point3f(camera_config, "look", Point3f::new(1.0, 1.0, 1.0));
    let cam_up = fetch_vector3f(camera_config, "up", Vector3f::new(0.0, 0.0, 1.0));
    let to_camera = Transform::look_at(&world_pos, &cam_look, &cam_up);
    let shutter_open = read_f64(camera_config, "shutter_open", 0.0);
    let shutter_close = read_f64(camera_config, "shutter_close", 1.0);
    let aperture_diameter = read_f64(camera_config, "aperture_diameter", 1.0);
    let focus_distance = read_f64(camera_config, "focus_distance", 10.0);
    let simple_weighting = read_bool(camera_config, "simple_weighting", true);
    let lens_data = read_num_array(camera_config.get("lens_data").unwrap(), 0).unwrap();

    let mut m = None;
    if let Some(medium_config) = camera_config.get("medium") {
        if let Ok(medium) = make_medium(medium_config) {
            m = Some(medium);
        }
    }
    RealisticCamera::new(
        Transform::inverse(&to_camera),
        shutter_open,
        shutter_close,
        aperture_diameter,
        focus_distance,
        Arc::new(film),
        m,
        &lens_data,
        simple_weighting,
    )
}

fn make_integrator(scene_config: &Value, save_to: &str) -> Box<dyn Integrator> {
    if let (Ok(integrator_config), Ok(sampler_config), Ok(film_config), Ok(camera_config)) = (
        search_object(scene_config, "Integrator"),
        search_object(scene_config, "Sampler"),
        search_object(scene_config, "Film"),
        search_object(scene_config, "Camera"),
    ) {
        let film = make_film(film_config, save_to);
        let cam = make_camera(camera_config, film);
        let sampler = make_sampler(sampler_config, &cam.film.cropped_pixel_bounds);

        let pixel_bounds = Bounds2i::new(Point2i::default(), cam.film.full_resolution);
        match read_string(integrator_config, "integrator_type", "AO").as_str() {
            "DirectLighting" => {
                let strategy =
                    match read_string(integrator_config, "light_strategy", "one").as_str() {
                        "all" => LightStrategy::UniformSampleAll,
                        _ => LightStrategy::UniformSampleOne,
                    };
                let max_depth = read_i64(integrator_config, "max_depth", 5) as usize;

                return Box::new(DirectLightingIntegrator::new(
                    strategy,
                    max_depth,
                    Arc::new(SamplerIntegratorData::new(
                        Arc::new(cam),
                        sampler,
                        pixel_bounds,
                    )),
                ));
            }
            "Path" => {
                let max_depth = read_i64(integrator_config, "max_depth", 5) as usize;
                let rr_threshold = read_f64(integrator_config, "rr_threshold", 1.0);
                return Box::new(PathIntegrator::new(
                    max_depth,
                    rr_threshold,
                    Arc::new(SamplerIntegratorData::new(
                        Arc::new(cam),
                        sampler,
                        pixel_bounds,
                    )),
                ));
            }
            "Volpath" => {
                let max_depth = read_i64(integrator_config, "max_depth", 5) as usize;
                let rr_threshold = read_f64(integrator_config, "rr_threshold", 1.0);
                return Box::new(VolPathIntegrator::new(
                    max_depth,
                    rr_threshold,
                    Arc::new(SamplerIntegratorData::new(
                        Arc::new(cam),
                        sampler,
                        pixel_bounds,
                    )),
                ));
            }
            "SPPM" => {
                let radius = read_f64(integrator_config, "radius", 1.0);
                let n_iters = read_i64(integrator_config, "n_iters", 5) as usize;
                let max_depth = read_i64(integrator_config, "max_depth", 5) as usize;
                let photons_per_iter = read_i64(integrator_config, "photons_per_iter", 1) as usize;
                let write_freq = read_i64(integrator_config, "write_freq", 1 << 31) as usize;
                return Box::new(SPPMIntegrator::new(
                    Arc::new(cam),
                    radius,
                    n_iters,
                    max_depth,
                    photons_per_iter,
                    write_freq,
                ));
            }
            "Debug" => {
                let max_depth = read_i64(integrator_config, "max_depth", 5) as usize;

                return Box::new(IntersectDebugIntegrator::new(
                    max_depth,
                    Arc::new(SamplerIntegratorData::new(
                        Arc::new(cam),
                        sampler,
                        pixel_bounds,
                    )),
                ));
            }
            _ => {
                let cos_sample = read_bool(integrator_config, "cos_sample", true);
                let n_samples = read_i64(integrator_config, "n_samples", 64) as u32;
                return Box::new(AOIntegrator::new(
                    cos_sample,
                    n_samples,
                    Arc::new(SamplerIntegratorData::new(
                        Arc::new(cam),
                        sampler,
                        pixel_bounds,
                    )),
                ));
            }
        }
    }
    panic!("Failed to create Integrator")
}

pub fn write_image(filename: &str, rgb: &[f64], output_bounds: Bounds2i) -> Result<(), ImageError> {
    let resolution = output_bounds.diagonal();

    let mut img_buf = ImageBuffer::new(resolution.x as u32, resolution.y as u32);
    let mut colored_pixels = 0_usize;
    for y in 0..resolution.y {
        for x in 0..resolution.x {
            let r = rgb[(3 * (y * resolution.x + x) + 0) as usize];
            let g = rgb[(3 * (y * resolution.x + x) + 1) as usize];
            let b = rgb[(3 * (y * resolution.x + x) + 2) as usize];
            let clamped_r = (clamp_t(255.0 * gamma_correct(r) + 0.5, 0.0, 255.0)) as u8;
            let clamped_g = (clamp_t(255.0 * gamma_correct(g) + 0.5, 0.0, 255.0)) as u8;
            let clamped_b = (clamp_t(255.0 * gamma_correct(b) + 0.5, 0.0, 255.0)) as u8;
            if (clamped_r as u32 + clamped_g as u32 + clamped_b as u32) > 0 {
                colored_pixels += 1;
            }
            img_buf.put_pixel(
                x as u32,
                y as u32,
                Rgba([clamped_r, clamped_g, clamped_b, 255 as u8]),
            )
        }
    }
    eprintln!(
        "Total pixels: {}. Colored Pixels: {}",
        resolution.x * resolution.y,
        colored_pixels
    );
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
