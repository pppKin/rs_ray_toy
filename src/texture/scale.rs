use std::{ops::Mul, sync::Arc};

use crate::interaction::SurfaceInteraction;

use super::Texture;

#[derive(Debug)]
pub struct ScaleTexture<T, U>
where
    T: Mul<U, Output = T>,
{
    t1: Arc<dyn Texture<T>>,
    t2: Arc<dyn Texture<U>>,
}

impl<T, U> ScaleTexture<T, U>
where
    T: Mul<U, Output = T>,
{
    pub fn new(t1: Arc<dyn Texture<T>>, t2: Arc<dyn Texture<U>>) -> Self {
        Self { t1, t2 }
    }
}

impl<T, U> Texture<T> for ScaleTexture<T, U>
where
    T: std::fmt::Debug,
    U: std::fmt::Debug,
    T: Mul<U, Output = T>,
{
    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        self.t1.evaluate(si) * self.t2.evaluate(si)
    }
}
