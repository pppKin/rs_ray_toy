use super::*;
#[derive(Default, Debug, Copy, Clone)]
pub struct BoxFltr {
    // not to be confused with Box
}
pub type BoxFilter = Filter<BoxFltr>;

impl IFilter for BoxFltr {
    fn evaluate(&mut self, _p: &Point2f, _r: &mut FilterRadius) -> f64 {
        1_f64
    }
}

pub fn create_box_filter(radius: Vector2f) -> BoxFilter {
    BoxFilter::new(BoxFltr::default(), FilterRadius::new(radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxfilter() {
        let mut f = create_box_filter(Vector2f::new(0.5, 0.5));
        let s = f.evaluate(&Point2f::new(0.1, 0.2));
        println!("{:?}", s);
    }
}
