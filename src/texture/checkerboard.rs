use std::{
    ops::{Add, Mul},
    sync::Arc,
};

use crate::{
    geometry::{Vector2f, Vector3f},
    interaction::SurfaceInteraction,
};

use super::{Texture, TextureMapping2D, TextureMapping3D};

// AAMethod Declaration
#[derive(Debug)]
pub enum AAMethod {
    AANone,
    ClosedForm,
}

#[derive(Debug)]
pub struct Checkerboard2DTexture<T> {
    mapping: Box<dyn TextureMapping2D>,

    tex1: Arc<dyn Texture<T>>,
    tex2: Arc<dyn Texture<T>>,
    aa_method: AAMethod,
}

impl<T> Checkerboard2DTexture<T> {
    pub fn new(
        mapping: Box<dyn TextureMapping2D>,
        tex1: Arc<dyn Texture<T>>,
        tex2: Arc<dyn Texture<T>>,
        aa_method: AAMethod,
    ) -> Self {
        Self {
            mapping,
            tex1,
            tex2,
            aa_method,
        }
    }
}

fn bump_int(x: f64) -> f64 {
    (x / 2.0).floor() + 2.0 * ((x / 2.0 - (x / 2.0).floor() - 0.5).max(0.0))
}

impl<T> Texture<T> for Checkerboard2DTexture<T>
where
    T: std::fmt::Debug + Mul<f64, Output = T> + Add<Output = T> + Send + Sync,
{
    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();

        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        match self.aa_method {
            AAMethod::AANone => {
                if (st[0].floor() as i32 + st[1].floor() as i32) % 2 == 0 {
                    return self.tex1.evaluate(si);
                }
                return self.tex2.evaluate(si);
            }
            AAMethod::ClosedForm => {
                // Compute closed-form box-filtered _Checkerboard2DTexture_ value

                // Evaluate single check if filter is entirely inside one of them
                let ds = dstdx.abs().max_comp();
                let dt = dstdy.abs().max_comp();
                let s0 = st[0] - ds;
                let s1 = st[0] + ds;
                let t0 = st[1] - dt;
                let t1 = st[1] + dt;

                if s0.floor() == s1.floor() && t0.floor() == t1.floor() {
                    // Point sample _Checkerboard2DTexture_
                    if (st[0].floor() as i32 + st[1].floor() as i32) % 2 == 0 {
                        return self.tex1.evaluate(si);
                    } else {
                        return self.tex2.evaluate(si);
                    }
                }
                // Apply box filter to checkerboard region
                let sint = (bump_int(s1) - bump_int(s0)) / (2.0 * ds);
                let tint = (bump_int(t1) - bump_int(t0)) / (2.0 * dt);
                let mut area2 = sint + tint - 2.0 * sint * tint;
                if ds > 1.0 || dt > 1.0 {
                    area2 = 0.5;
                }

                self.tex1.evaluate(si) * (1.0 - area2) + self.tex2.evaluate(si) * area2
            }
        }
    }
}

#[derive(Debug)]
pub struct Checkerboard3DTexture<T: std::fmt::Debug> {
    mapping: Box<dyn TextureMapping3D>,

    tex1: Arc<dyn Texture<T>>,
    tex2: Arc<dyn Texture<T>>,
}

impl<T: std::fmt::Debug> Checkerboard3DTexture<T> {
    pub fn new(
        mapping: Box<dyn TextureMapping3D>,
        tex1: Arc<dyn Texture<T>>,
        tex2: Arc<dyn Texture<T>>,
    ) -> Self {
        Self {
            mapping,
            tex1,
            tex2,
        }
    }
}

impl<T: std::fmt::Debug + Send + Sync> Texture<T> for Checkerboard3DTexture<T> {
    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let p = self.mapping.map(si, &mut dpdx, &mut dpdy);

        if (p.x.floor() + p.y.floor() + p.z.floor()) as i32 % 2 == 0 {
            return self.tex1.evaluate(si);
        } else {
            return self.tex2.evaluate(si);
        }
    }
}
