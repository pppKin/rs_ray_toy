use rs_ray_toy::scene;
use rs_ray_toy::scene::Scene;
use std::sync::Arc;

use std::env;
use std::str::FromStr;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];

    let _scn = Arc::new(scene::make_scene(filename));
}
