use std::{ops::Add, ops::Mul, sync::Arc};

use crate::interaction::SurfaceInteraction;

use super::Texture;

#[derive(Debug)]
pub struct MixTexture<T>
where
    T: Add<Output = T> + std::fmt::Debug,
{
    t1: Arc<dyn Texture<T>>,
    t2: Arc<dyn Texture<T>>,

    amount: Arc<dyn Texture<f64>>,
}

impl<T> MixTexture<T>
where
    T: Add<Output = T> + std::fmt::Debug,
{
    pub fn new(
        t1: Arc<dyn Texture<T>>,
        t2: Arc<dyn Texture<T>>,
        amount: Arc<dyn Texture<f64>>,
    ) -> Self {
        Self { t1, t2, amount }
    }
}

impl<T> Texture<T> for MixTexture<T>
where
    T: std::fmt::Debug + Mul<f64, Output = T> + Add<Output = T> + Send + Sync,
{
    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let amt = self.amount.evaluate(si);
        self.t1.evaluate(si) as T * (1.0 - amt) + self.t2.evaluate(si) as T * amt
    }
}
