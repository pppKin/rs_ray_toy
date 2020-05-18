use crate::geometry::{Point2f, Vector2f};

pub trait IFilter {
    fn evaluate(&mut self, p: &Point2f, r: &mut FilterRadius) -> f64;
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

#[derive(Default, Debug, Copy, Clone)]
pub struct Filter<T> {
    f: T,
    pub r: FilterRadius,
}

impl<T> Filter<T> {
    pub fn new(f: T, r: FilterRadius) -> Self {
        Self { f, r }
    }
}

impl<T> Filter<T>
where
    T: IFilter,
{
    pub fn evaluate(&mut self, p: &Point2f) -> f64 {
        self.f.evaluate(p, &mut self.r)
    }
}

pub mod boxfilter;
pub mod gaussian;
pub mod trianglefilter;
