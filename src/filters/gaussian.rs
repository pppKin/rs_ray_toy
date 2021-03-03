use super::*;
#[derive(Default, Debug, Copy, Clone)]
pub struct Gaussian {
    alpha: f64,
    exp_x: f64,
    exp_y: f64,
}

impl Gaussian {
    pub fn new(alpha: f64, exp_x: f64, exp_y: f64) -> Self {
        Self {
            alpha,
            exp_x,
            exp_y,
        }
    }

    fn gaussian_func(&self, d: f64, expv: f64) -> f64 {
        f64::max(0_f64, f64::exp(-self.alpha * d * d) - expv)
    }
}

pub type GaussianFilter = Filter<Gaussian>;

impl IFilter for Gaussian {
    fn if_evaluate(&self, p: &Point2f, _r: &FilterRadius) -> f64 {
        // Gaussian(p.x, expX) * Gaussian(p.y, expY)
        self.gaussian_func(p.x, self.exp_x) * self.gaussian_func(p.y, self.exp_y)
    }
}

pub fn create_gaussian_filter(radius: Vector2f, alpha: f64) -> GaussianFilter {
    GaussianFilter::new(
        Arc::new(Gaussian::new(
            alpha,
            f64::exp(-alpha * radius.x * radius.x),
            f64::exp(-alpha * radius.y * radius.y),
        )),
        FilterRadius::new(radius),
    )
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_gaussian_filter() {
//         let mut f = create_gaussian_filter(Vector2f::new(0.5, 0.5), 1_f64);
//         let s = f.evaluate(&Point2f::new(0.0, 0.0));
//         println!("{:?}", s);
//     }
// }
