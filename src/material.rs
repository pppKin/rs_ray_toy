// type Material struct {
// 	color                                                              Color
// 	difuseCol, specularCol, specularD, reflectionCol, transmitCol, IOR float64
// }

// func (m Material) String() string {
// 	return fmt.Sprintf("<Mat: %s %.2f %.2f %.2f %.2f %.2f %.2f>", m.color.String(), m.difuseCol, m.specularCol, m.specularD, m.reflectionCol, m.transmitCol, m.IOR)
// }
use crate::color::Color;

#[derive(Debug, Default, Copy, Clone)]
pub struct Material {
    pub color: Color,
    pub diffuse_col: f64,
    pub specular_col: f64,
    pub specular_d: f64,
    pub reflection_col: f64,
    pub transmit_col: f64,
    pub ior: f64,
}
