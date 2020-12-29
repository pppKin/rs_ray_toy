use super::{turbulence, Texture, TextureMapping3D};
use crate::geometry::Vector3f;

#[derive(Debug)]
pub struct WrinkledTexture {
    mapping: Box<dyn TextureMapping3D>,

    octaves: u64,
    omega: f64,
}

impl<T: From<f64>> Texture<T> for WrinkledTexture {
    fn evaluate(&self, si: &crate::interaction::SurfaceInteraction) -> T {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let p = self.mapping.map(si, &mut dpdx, &mut dpdy);
        T::from(turbulence(&p, &dpdx, &dpdy, self.omega, self.octaves))
    }
}
