use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use crate::{geometry::Vector2f, interaction::SurfaceInteraction};

use super::{Texture, TextureMapping2D};

#[derive(Debug)]
pub struct BilerpTexture<T: Debug> {
    mapping: Box<dyn TextureMapping2D>,
    v00: T,
    v01: T,
    v10: T,
    v11: T,
}

impl<T: Debug> BilerpTexture<T> {
    pub fn new(mapping: Box<dyn TextureMapping2D>, v00: T, v01: T, v10: T, v11: T) -> Self {
        Self {
            mapping,
            v00,
            v01,
            v10,
            v11,
        }
    }
}

impl<T> Texture<T> for BilerpTexture<T>
where
    T: Debug + Copy + Mul<f64, Output = T> + Add<Output = T> + Send + Sync,
{
    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        self.v00 * (1.0 - st[0]) * (1.0 - st[1])
            + self.v01 * (1.0 - st[0]) * st[1]
            + self.v10 * st[0] * (1.0 - st[1])
            + self.v11 * st[0] * st[1]
    }
}
