use crate::color::Color;
use crate::geometry::Point3f;
pub struct Light {
    pub position: Point3f,
    pub color: Color,
    pub kind: String,
}
