use super::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleFltr {}

pub type TriangleFilter = Filter<TriangleFltr>;

impl IFilter for TriangleFltr {
    fn evaluate(&mut self, p: &Point2f, r: &mut FilterRadius) -> f64 {
        f64::max(0_f64, r.radius.x - p.x.abs()) * f64::max(0_f64, r.radius.y - p.y.abs())
    }
}

pub fn create_triangle_filter(radius: Vector2f) -> TriangleFilter {
    TriangleFilter::new(TriangleFltr::default(), FilterRadius::new(radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_filter() {
        let mut t = create_triangle_filter(Vector2f::new(0.5, 0.5));
        let s = t.evaluate(&Point2f::new(0.1, 0.2));
        println!("{:?}", s);
    }
}