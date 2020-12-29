use std::fmt::Debug;

use crate::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    mipmap::{ImageWrap, MIPMap},
    rtoycore::SPECTRUM_N,
    spectrum::Spectrum,
};

use super::{Texture, TextureMapping2D};

// TexInfo Declarations
#[derive(Debug, PartialEq, PartialOrd)]
pub struct TexInfo {
    filename: String,
    do_trilinear: bool,
    max_aniso: f64,

    wrap_mode: ImageWrap,
    scale: f64,
    gamma: bool,
}

#[derive(Debug)]
pub struct ImageTexture {
    mapping: Box<dyn TextureMapping2D>,
    mipmap: MIPMap,
}

impl ImageTexture {
    pub fn new(mapping: Box<dyn TextureMapping2D>, mipmap: MIPMap) -> Self {
        Self { mapping, mipmap }
    }
}

impl Texture<Spectrum<SPECTRUM_N>> for ImageTexture {
    fn evaluate(&self, si: &SurfaceInteraction) -> Spectrum<SPECTRUM_N> {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        let mem = self.mipmap.lookup_d(&st, &dstdx, &dstdy);
        mem
    }
}
