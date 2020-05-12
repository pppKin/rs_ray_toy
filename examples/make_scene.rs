use rs_ray_toy::scene;
use std::sync::Arc;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];

    let _scn = Arc::new(scene::make_scene(filename));
}
