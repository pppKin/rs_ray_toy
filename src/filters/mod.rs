use std::{fmt::Debug, sync::Arc};

use crate::geometry::{Point2f, Vector2f};

pub trait IFilter: Debug + Send + Sync {
    fn if_evaluate(&self, p: &Point2f, r: &FilterRadius) -> f64;
}

#[derive(Default, Debug, Copy, Clone)]
pub struct FilterRadius {
    pub radius: Vector2f,
    pub inv_radius: Vector2f,
}

impl FilterRadius {
    pub fn new(radius: Vector2f) -> Self {
        Self {
            radius,
            inv_radius: Vector2f::new(1_f64 / radius.x, 1_f64 / radius.y),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Filter {
    f: Arc<dyn IFilter>,
    pub r: FilterRadius,
}

impl Filter {
    pub fn new(f: Arc<dyn IFilter>, r: FilterRadius) -> Self {
        Self { f, r }
    }

    pub fn evaluate(&self, p: &Point2f) -> f64 {
        self.f.if_evaluate(p, &self.r)
    }
}

pub mod boxfilter;
pub mod gaussian;
pub mod trianglefilter;
