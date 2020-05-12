use crate::geometry::Cxyz;
use crate::misc::clamp_t;
use image::Rgba;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
#[derive(Debug, Default, Copy, Clone)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    pub fn new() -> Self {
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }
    pub fn to_pixel(&self) -> Rgba<u8> {
        return Rgba([
            (clamp_t(self.r, 0.0, 1.0) * 255.0) as u8,
            (clamp_t(self.g, 0.0, 1.0) * 255.0) as u8,
            (clamp_t(self.b, 0.0, 1.0) * 255.0) as u8,
            255 as u8,
        ]);
    }
}

impl Cxyz for Color {
    fn to_xyz(&self) -> (f64, f64, f64) {
        return (self.r, self.g, self.b);
    }
    fn from_xyz(x: f64, y: f64, z: f64) -> Color {
        return Color { r: x, g: y, b: z };
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Color {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl AddAssign<Color> for Color {
    fn add_assign(&mut self, rhs: Color) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Color {
        Color {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl MulAssign<f64> for Color {
    fn mul_assign(&mut self, rhs: f64) {
        self.r *= rhs;
        self.g *= rhs;
        self.b *= rhs;
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Color {
        Color {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl MulAssign<Color> for Color {
    fn mul_assign(&mut self, rhs: Color) {
        self.r *= rhs.r;
        self.g *= rhs.g;
        self.b *= rhs.b;
    }
}

impl Div<f64> for Color {
    type Output = Color;
    fn div(self, rhs: f64) -> Color {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        Color {
            r: self.r * inv,
            g: self.g * inv,
            b: self.b * inv,
        }
    }
}

impl DivAssign<f64> for Color {
    fn div_assign(&mut self, rhs: f64) {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        self.r *= inv;
        self.g *= inv;
        self.b *= inv;
    }
}

impl Neg for Color {
    type Output = Color;
    fn neg(self) -> Color {
        Color {
            r: -self.r,
            g: -self.g,
            b: -self.b,
        }
    }
}

impl SubAssign<f64> for Color {
    fn sub_assign(&mut self, rhs: f64) {
        self.r -= rhs;
        self.g -= rhs;
        self.b -= rhs;
    }
}

impl Sub<f64> for Color {
    type Output = Color;
    fn sub(self, rhs: f64) -> Color {
        Color {
            r: self.r - rhs,
            g: self.g - rhs,
            b: self.b - rhs,
        }
    }
}

impl Index<u8> for Color {
    type Output = f64;
    #[inline]
    fn index(&self, index: u8) -> &f64 {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}

impl IndexMut<u8> for Color {
    #[inline]
    fn index_mut(&mut self, index: u8) -> &mut f64 {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}
