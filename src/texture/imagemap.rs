use std::fmt::Debug;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    mipmap::{ImageWrap, MIPMap},
    spectrum::Spectrum,
    SPECTRUM_N,
};

use super::{Texture, TextureMapping2D};

// TexInfo Declarations
#[derive(Debug, PartialEq, PartialOrd)]
pub struct TexInfo {
    pub filename: String,
    pub do_trilinear: bool,
    pub max_aniso: f64,

    pub wrap_mode: ImageWrap,
    pub scale: f64,
    pub gamma: bool,
}

impl Hash for TexInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.filename.hash(state);
        self.do_trilinear.hash(state);
        self.max_aniso.to_bits().hash(state);
        self.wrap_mode.hash(state);
        self.scale.to_bits().hash(state);
        self.gamma.hash(state);
    }
}

impl Eq for TexInfo {}

impl TexInfo {
    pub fn new(
        filename: String,
        do_trilinear: bool,
        max_aniso: f64,
        wrap_mode: ImageWrap,
        scale: f64,
        gamma: bool,
    ) -> Self {
        Self {
            filename,
            do_trilinear,
            max_aniso,
            wrap_mode,
            scale,
            gamma,
        }
    }
}

#[derive(Debug)]
pub struct ImageTexture {
    mapping: Box<dyn TextureMapping2D>,
    mipmap: Arc<MIPMap>,
}

impl ImageTexture {
    pub fn new(mapping: Box<dyn TextureMapping2D>, mipmap: Arc<MIPMap>) -> Self {
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
