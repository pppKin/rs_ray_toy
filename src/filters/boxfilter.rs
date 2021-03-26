use super::*;
#[derive(Default, Debug, Copy, Clone)]
pub struct BoxFltr {
    // not to be confused with Box
}

impl IFilter for BoxFltr {
    fn if_evaluate(&self, _p: &Point2f, _r: &FilterRadius) -> f64 {
        1_f64
    }
}

pub fn create_box_filter(radius: Vector2f) -> Filter {
    Filter::new(Arc::new(BoxFltr::default()), FilterRadius::new(radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxfilter() {
        let f = create_box_filter(Vector2f::new(0.5, 0.5));
        let s = f.evaluate(&Point2f::new(0.1, 0.2));
        println!("{:?}", s);
    }
}
