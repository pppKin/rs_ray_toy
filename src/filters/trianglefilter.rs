use super::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleFltr {}

impl IFilter for TriangleFltr {
    fn if_evaluate(&self, p: &Point2f, r: &FilterRadius) -> f64 {
        f64::max(0_f64, r.radius.x - p.x.abs()) * f64::max(0_f64, r.radius.y - p.y.abs())
    }
}

pub fn create_triangle_filter(radius: Vector2f) -> Filter {
    Filter::new(Arc::new(TriangleFltr::default()), FilterRadius::new(radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_filter() {
        let t = create_triangle_filter(Vector2f::new(0.5, 0.5));
        let s = t.evaluate(&Point2f::new(0.1, 0.2));
        println!("{:?}", s);
    }
}
