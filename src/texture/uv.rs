use crate::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    rtoycore::SPECTRUM_N,
    spectrum::{ISpectrum, Spectrum, SpectrumType},
};

use super::{Texture, TextureMapping2D};

#[derive(Debug)]
pub struct UVTexture {
    mapping: Box<dyn TextureMapping2D>,
}

impl UVTexture {
    pub fn new(mapping: Box<dyn TextureMapping2D>) -> Self {
        Self { mapping }
    }
}

impl Texture<Spectrum<SPECTRUM_N>> for UVTexture {
    fn evaluate(&self, si: &SurfaceInteraction) -> Spectrum<SPECTRUM_N> {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        let rgb = [st[0] - st[0].floor(), st[1] - st[1].floor(), 0.0];
        Spectrum::from_rgb(rgb, SpectrumType::Reflectance)
    }
}
