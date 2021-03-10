use image::{ImageBuffer, ImageError, Rgba};
use serde_json::Value;

use std::{fs, sync::Arc};

use crate::{
    geometry::{Bounds2i, Cxyz},
    integrator::Integrator,
    lights::Light,
    misc::{clamp_t, gamma_correct},
    primitives::Primitive,
    scene::Scene,
    transform::Transform,
};

/// read scene config file (json), create required resources, and start rendering
pub fn deploy_render(filepath: &str, save_to: &str) {
    let scene_config_str = fs::read_to_string(filepath).unwrap();
    let scene_config: Value = serde_json::from_str(&scene_config_str).unwrap();
    let scene = make_scene(&scene_config);
    let mut inte = make_integrator(&scene_config, save_to);
    inte.render(&scene);
}

/// Search first level of value for a key
fn bsearch_object<'a>(root: &'a Value, key: &str) -> Result<&'a Value, String> {
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

fn make_xyz<'a, T>(root: &'a Value) -> Result<T, String>
where
    T: Cxyz<f64>,
{
    let mut try_vec = Vec::with_capacity(3);
    if let Value::Array(m) = root {
        if m.len() != 3 {
            return Err(format!(
                "Failed to parse Transform from {:?}, expected and array of 16 numbers",
                root
            ));
        }
        for i in 0..3 {
            if let Value::Number(n) = &m[i] {
                try_vec.push(n.as_f64().unwrap());
            } else {
                return Err(format!("Failed to parse x,y, z from {:?}", root));
            }
        }
    }
    if try_vec.len() == 3 {
        return Ok(T::from_xyz(try_vec[0], try_vec[1], try_vec[2]));
    }
    Err(format!("Failed to parse x,y, z from {:?}", root))
}

/// create a transform
fn make_transform<'a>(root: &'a Value) -> Result<Transform, String> {
    let mut try_matrix = Vec::with_capacity(16);
    if let Value::Array(m) = root {
        if m.len() != 16 {
            return Err(format!(
                "Failed to parse Transform from {:?}, expected and array of 16 numbers",
                root
            ));
        }
        for i in 0..16 {
            try_matrix.push(m[i].as_f64());
        }
    }
    let p_matrix: Option<Vec<f64>> = try_matrix.into_iter().collect();
    if let Some(m) = p_matrix {
        return Ok(Transform::new(
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15],
        ));
    }
    Err(format!("Failed to parse Transform from {:?}", root))
}

pub fn make_scene(scene_config: &Value) -> Scene {
    let (lights, infinite_lights) = make_all_lights(&scene_config);
    let aggregate = make_aggregate(&scene_config);
    Scene::new(lights, infinite_lights, aggregate)
}

pub fn make_all_lights(scene_config: &Value) -> (Vec<Arc<dyn Light>>, Vec<Arc<dyn Light>>) {
    let lights_config = bsearch_object(scene_config, "lights");
    let mut lights_len = 0;
    let mut lights;
    let mut infinite_lights;
    if let Ok(Value::Array(a_l)) = lights_config {
        lights_len = a_l.len();
        lights = Vec::with_capacity(lights_len);
        for light_config in a_l {
            lights.push(make_light(light_config));
        }
    } else {
        lights = vec![];
    }
    let infinite_lights_config = bsearch_object(scene_config, "infinite_lights");
    let mut inf_lights_len = 0;
    if let Ok(Value::Array(a_l)) = infinite_lights_config {
        inf_lights_len = a_l.len();
        infinite_lights = Vec::with_capacity(inf_lights_len);
        for light_config in a_l {
            infinite_lights.push(make_light(light_config));
        }
    } else {
        infinite_lights = vec![];
    }
    if lights_len + inf_lights_len == 0 {
        eprintln!("No lights found!")
    }

    return (lights, infinite_lights);
}

pub fn make_light(light_config: &Value) -> Arc<dyn Light> {
    assert!(light_config.is_object());
    if let Some(Value::String(light_type)) = light_config.get("light_type") {
        match light_type.as_str() {
            "point" => {}
            _ => {
                panic!("Failed to parse light {:?}", light_type);
            }
        }
    }
    panic!("Failed to parse light {:?}", light_config)
}

pub fn make_aggregate(scene_config: &Value) -> Arc<dyn Primitive> {
    todo!();
}

pub fn make_integrator(scene_config: &Value, save_to: &str) -> Box<dyn Integrator> {
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
