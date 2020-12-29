use super::{fbm, Texture, TextureMapping3D};
use crate::geometry::Vector3f;

#[derive(Debug)]
pub struct WindyTexture {
    mapping: Box<dyn TextureMapping3D>,
}

impl WindyTexture {
    pub fn new(mapping: Box<dyn TextureMapping3D>) -> Self {
        Self { mapping }
    }
}

impl<T: From<f64>> Texture<T> for WindyTexture {
    fn evaluate(&self, si: &crate::interaction::SurfaceInteraction) -> T {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let p = self.mapping.map(si, &mut dpdx, &mut dpdy);
        let wind_strength = fbm(&(p * 0.1), &(dpdx * 0.1), &(dpdy * 0.1), 0.5, 3);
        let wave_height = fbm(&p, &dpdx, &dpdy, 0.5, 6);
        T::from(wind_strength.abs() * wave_height)
    }
}
