use crate::{
    medium::MediumOpArc,
    misc::{clamp_t, float_nearly_equal, gamma, lerp, next_float_down, next_float_up},
    MAX_DIST,
};
use std::{
    f64::{consts::PI, INFINITY},
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub type Point2f = Point2<f64>;
pub type Point2i = Point2<i64>;
pub type Point3f = Point3<f64>;
pub type Point3i = Point3<i64>;
pub type Vector2f = Vector2<f64>;
pub type Vector2i = Vector2<i64>;
pub type Vector3f = Vector3<f64>;
pub type Vector3i = Vector3<i64>;
pub type Normal3f = Normal3<f64>;

#[derive(Debug, Default, Copy, Clone)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Normal3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Bounds2f = Bounds2<f64>;
pub type Bounds2i = Bounds2<i64>;
pub type Bounds3f = Bounds3<f64>;
pub type Bounds3i = Bounds3<i64>;

#[derive(Debug, Default, Copy, Clone)]
pub struct Bounds2<T> {
    pub p_min: Point2<T>,
    pub p_max: Point2<T>,
}

#[derive(Debug, Copy, Clone)]
pub struct Bounds3<T> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

#[derive(Debug, Clone)]
pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,
    pub t_max: f64, // MAX_DIST
    pub time: f64,
    pub medium: MediumOpArc,
}

#[derive(Default, Debug, Clone)]
pub struct RayDifferential {
    pub ray: Ray,
    pub has_differentials: bool,
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_direction: Vector3f,
    pub ry_direction: Vector3f,
}
// bool hasDifferentials;
// Point3f rxOrigin, ryOrigin;
// Vector3f rxDirection, ryDirection;

pub trait IntersectP {
    fn intersect_p(&self, r: &Ray) -> bool;
}

pub trait Cxyz<T: Copy>
where
    Self: Sized,
{
    fn to_xyz(&self) -> (T, T, T);
    fn from_xyz(x: T, y: T, z: T) -> Self;
}

/// Product of the Euclidean magnitudes of the two vectors and the
/// cosine of the angle between them. A return value of zero means
/// both vectors are orthogonal, a value if one means they are
/// codirectional.
pub fn dot3<T: Copy + Mul<T, Output = T> + Add<T, Output = T>>(
    a: &impl Cxyz<T>,
    b: &impl Cxyz<T>,
) -> T {
    let (x1, y1, z1) = a.to_xyz();
    let (x2, y2, z2) = b.to_xyz();
    (x1 * x2) + (y1 * y2) + (z1 * z2)
}

pub fn abs_dot3(a: &impl Cxyz<f64>, b: &impl Cxyz<f64>) -> f64 {
    dot3(a, b).abs()
}

// Point2 Methods

impl<T> Point2<T> {
    pub fn new(x: T, y: T) -> Point2<T> {
        Point2::<T> { x: x, y: y }
    }
}

impl Point2<f64> {
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

impl From<Point2f> for Point2i {
    fn from(p: Point2f) -> Self {
        Self::new(p.x as i64, p.y as i64)
    }
}

impl From<Point2i> for Point2f {
    fn from(p: Point2i) -> Self {
        Self::new(p.x as f64, p.y as f64)
    }
}

pub fn convert_pnt2<T, U>(incoming: Point2<T>) -> Point2<U>
where
    T: Into<U>,
{
    Point2::<U>::new(incoming.x.into(), incoming.y.into())
}

impl<T> PartialEq for Point2<T>
where
    T: PartialEq,
{
    fn eq(&self, rhs: &Point2<T>) -> bool {
        self.x == rhs.x && self.y == rhs.y
    }
}

impl<T> Add<Point2<T>> for Point2<T>
where
    T: Add<T, Output = T>,
{
    type Output = Point2<T>;
    fn add(self, rhs: Point2<T>) -> Point2<T> {
        Point2::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T> Add<T> for Point2<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Point2<T>;
    fn add(self, rhs: T) -> Point2<T> {
        Point2::<T> {
            x: self.x + rhs,
            y: self.y + rhs,
        }
    }
}

impl<T> Add<Vector2<T>> for Point2<T>
where
    T: Add<T, Output = T>,
{
    type Output = Point2<T>;
    fn add(self, rhs: Vector2<T>) -> Point2<T> {
        Point2::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T> Sub<Point2<T>> for Point2<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Vector2<T>;
    fn sub(self, rhs: Point2<T>) -> Vector2<T> {
        Vector2::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T> Sub<Vector2<T>> for Point2<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Point2<T>;
    fn sub(self, rhs: Vector2<T>) -> Point2<T> {
        Point2::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T> Sub<T> for Point2<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Point2<T>;
    fn sub(self, rhs: T) -> Point2<T> {
        Point2::<T> {
            x: self.x - rhs,
            y: self.y - rhs,
        }
    }
}

impl<T> Mul<T> for Point2<T>
where
    T: Copy + Mul<T, Output = T>,
{
    type Output = Point2<T>;
    fn mul(self, rhs: T) -> Point2<T> {
        Point2::<T> {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T> Index<u8> for Point2<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: u8) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}

impl<T> IndexMut<u8> for Point2<T> {
    #[inline]
    fn index_mut(&mut self, index: u8) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}

/// Apply floor operation component-wise.
pub fn pnt2_floor(p: Point2<f64>) -> Point2<f64> {
    Point2 {
        x: p.x.floor(),
        y: p.y.floor(),
    }
}

/// Apply ceil operation component-wise.
pub fn pnt2_ceil(p: Point2<f64>) -> Point2<f64> {
    Point2 {
        x: p.x.ceil(),
        y: p.y.ceil(),
    }
}

/// Apply std::cmp::min operation component-wise.
pub fn pnt2_min_pnt2<T>(pa: Point2<T>, pb: Point2<T>) -> Point2<T>
where
    T: Ord,
{
    Point2 {
        x: std::cmp::min(pa.x, pb.x),
        y: std::cmp::min(pa.y, pb.y),
    }
}

/// Apply std::cmp::max operation component-wise.
pub fn pnt2_max_pnt2<T>(pa: Point2<T>, pb: Point2<T>) -> Point2<T>
where
    T: Ord,
{
    Point2 {
        x: std::cmp::max(pa.x, pb.x),
        y: std::cmp::max(pa.y, pb.y),
    }
}

/// Given a bounding box and a point, the **bnd2_union_pnt2()**
/// function returns a new bounding box that encompasses that point as
/// well as the original box.
pub fn bnd2_union_pnt2(b: &Bounds2<f64>, p: Point2<f64>) -> Bounds2<f64> {
    let p_min: Point2<f64> = Point2::<f64> {
        x: b.p_min.x.min(p.x),
        y: b.p_min.y.min(p.y),
    };
    let p_max: Point2<f64> = Point2::<f64> {
        x: b.p_max.x.max(p.x),
        y: b.p_max.y.max(p.y),
    };
    Bounds2 { p_min, p_max }
}

/// Pads the bounding box by a constant factor in both dimensions.
pub fn bnd2_expand(b: &Bounds2f, delta: f64) -> Bounds2f {
    Bounds2f {
        p_min: b.p_min - Vector2f { x: delta, y: delta },
        p_max: b.p_max + Vector2f { x: delta, y: delta },
    }
}

// Point3 Methods
impl<T> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Point3::<T> { x, y, z }
    }
}

pub trait PointMin {
    fn min(&self, other: &Self) -> Self;
}

impl<T> PointMin for Point3<T>
where
    T: PartialOrd + Copy,
{
    fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.x < other.x { self.x } else { other.x },
            if self.y < other.y { self.y } else { other.y },
            if self.z < other.z { self.z } else { other.z },
        )
    }
}

impl<T> PointMin for Point2<T>
where
    T: PartialOrd + Copy,
{
    fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.x < other.x { self.x } else { other.x },
            if self.y < other.y { self.y } else { other.y },
        )
    }
}

pub trait PointMax {
    fn max(&self, other: &Self) -> Self;
}

impl<T> PointMax for Point3<T>
where
    T: PartialOrd + Copy,
{
    fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.x > other.x { self.x } else { other.x },
            if self.y > other.y { self.y } else { other.y },
            if self.z > other.z { self.z } else { other.z },
        )
    }
}

impl<T> PointMax for Point2<T>
where
    T: PartialOrd + Copy,
{
    fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.x > other.x { self.x } else { other.x },
            if self.y > other.y { self.y } else { other.y },
        )
    }
}

impl Point3<f64> {
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Point3<i64> {
    pub fn from_pnt3f(p: &Point3f) -> Self {
        Self::new(p.x as i64, p.y as i64, p.z as i64)
    }
}

impl<T: Copy> Cxyz<T> for Point3<T> {
    fn to_xyz(&self) -> (T, T, T) {
        return (self.x, self.y, self.z);
    }
    fn from_xyz(x: T, y: T, z: T) -> Self {
        return Self { x: x, y: y, z: z };
    }
}

impl PartialEq for Point3f {
    fn eq(&self, rhs: &Point3f) -> bool {
        nearly_equal(self, rhs)
    }
}

impl<T> AddAssign<Point3<T>> for Point3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Point3<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T> Add<Point3<T>> for Point3<T>
where
    T: Add<T, Output = T>,
{
    type Output = Point3<T>;
    fn add(self, rhs: Point3<T>) -> Point3<T> {
        Point3::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign<T> for Point3<T>
where
    T: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: T) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl<T> Add<T> for Point3<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Point3<T>;
    fn add(self, rhs: T) -> Point3<T> {
        Point3::<T> {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl<T> Add<Vector3<T>> for Point3<T>
where
    T: Add<T, Output = T>,
{
    type Output = Point3<T>;
    fn add(self, rhs: Vector3<T>) -> Point3<T> {
        Point3::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign<Vector3<T>> for Point3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Vector3<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T> Sub<T> for Point3<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Point3<T>;
    fn sub(self, rhs: T) -> Point3<T> {
        Point3::<T> {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs,
        }
    }
}

impl<T> Sub<Point3<T>> for Point3<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Vector3<T>;
    fn sub(self, rhs: Point3<T>) -> Vector3<T> {
        Vector3::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Sub<Vector3<T>> for Point3<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Point3<T>;
    fn sub(self, rhs: Vector3<T>) -> Point3<T> {
        Point3::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Mul<T> for Point3<T>
where
    T: Copy + Mul<T, Output = T>,
{
    type Output = Point3<T>;
    fn mul(self, rhs: T) -> Point3<T>
    where
        T: Copy + Mul<T, Output = T>,
    {
        Point3::<T> {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> MulAssign<T> for Point3<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Point3<f64> {
    type Output = Point3<f64>;
    fn div(self, rhs: f64) -> Point3<f64> {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        Point3::<f64> {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl DivAssign<f64> for Point3<f64> {
    fn div_assign(&mut self, rhs: f64) {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl<T> Index<u8> for Point3<T> {
    type Output = T;
    fn index(&self, index: u8) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl<T> IndexMut<u8> for Point3<T> {
    fn index_mut(&mut self, index: u8) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl<T> Neg for Point3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Point3<T>;
    fn neg(self) -> Point3<T> {
        Point3::<T> {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Permute the coordinate values according to the povided
/// permutation.
pub fn pnt3_permute<T>(v: &Point3<T>, x: usize, y: usize, z: usize) -> Point3<T>
where
    T: Copy,
{
    let v3: [T; 3] = [v.x, v.y, v.z];
    let xp: T = v3[x];
    let yp: T = v3[y];
    let zp: T = v3[z];
    Point3::<T> {
        x: xp,
        y: yp,
        z: zp,
    }
}

/// Interpolate linearly between two provided points.
pub fn pnt3_lerp(t: f64, p0: &Point3f, p1: &Point3f) -> Point3f {
    *p0 * (1.0 as f64 - t) as f64 + *p1 * t
}

/// Apply floor operation component-wise.
pub fn pnt3_floor(p: &Point3<f64>) -> Point3<f64> {
    Point3 {
        x: p.x.floor(),
        y: p.y.floor(),
        z: p.z.floor(),
    }
}

/// Apply ceil operation component-wise.
pub fn pnt3_ceil(p: &Point3<f64>) -> Point3<f64> {
    Point3 {
        x: p.x.ceil(),
        y: p.y.ceil(),
        z: p.z.ceil(),
    }
}

/// Apply abs operation component-wise.
pub fn pnt3_abs(p: &Point3<f64>) -> Point3<f64> {
    Point3 {
        x: p.x.abs(),
        y: p.y.abs(),
        z: p.z.abs(),
    }
}

/// The distance between two points is the length of the vector
/// between them.
pub fn pnt3_distance(p1: &Point3<f64>, p2: &Point3<f64>) -> f64 {
    (*p1 - *p2).length()
}

/// The distance squared between two points is the length of the
/// vector between them squared.
pub fn pnt3_distance_squared(p1: &Point3<f64>, p2: &Point3<f64>) -> f64 {
    (*p1 - *p2).length_squared()
}

/// When tracing spawned rays leaving the intersection point p, we
/// offset their origins enough to ensure that they are past the
/// boundary of the error box and thus won't incorrectly re-intersect
/// the surface.
pub fn pnt3_offset_ray_origin(
    p: &Point3f,
    p_error: &Vector3f,
    n: &Normal3f,
    w: &Vector3f,
) -> Point3f {
    //     f64 d = Dot(Abs(n), pError);
    let d: f64 = dot3(&nrm_abs(n), p_error);
    // #ifdef PBRT_FLOAT_AS_DOUBLE
    //     // We have tons of precision; for now bump up the offset a bunch just
    //     // to be extra sure that we start on the right side of the surface
    //     // (In case of any bugs in the epsilons code...)
    //     d *= 1024.;
    // #endif
    let mut offset: Vector3f = Vector3f::from(*n) * d;
    if dot3(w, n) < 0.0 as f64 {
        offset = -offset;
    }
    let mut po: Point3f = *p + offset;
    // round offset point _po_ away from _p_
    for i in 0..3 {
        if offset[i] > 0.0 as f64 {
            po[i] = next_float_up(po[i]);
        } else if offset[i] < 0.0 as f64 {
            po[i] = next_float_down(po[i]);
        }
    }
    po
}

// Vector2 Methods
impl<T> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2::<T> { x, y }
    }
}

impl<T> Vector2<T>
where
    T: PartialOrd + Copy + From<i32> + Neg<Output = T>,
{
    pub fn max_comp(&self) -> T {
        if self.x > self.y {
            return self.x;
        } else {
            return self.y;
        }
    }
    pub fn abs(&self) -> Self {
        let x = if self.x < T::from(0) { -self.x } else { self.x };
        let y = if self.y < T::from(0) { -self.y } else { self.y };
        Self::new(x, y)
    }
}

impl Vector2<f64> {
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
    pub fn length_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2)
    }
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl<T> Index<u8> for Vector2<T> {
    type Output = T;
    fn index(&self, index: u8) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}

impl<T> IndexMut<u8> for Vector2<T> {
    fn index_mut(&mut self, index: u8) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Check failed: i >= 0 && i <= 1"),
        }
    }
}

impl AddAssign<Vector2<f64>> for Vector2<f64> {
    fn add_assign(&mut self, rhs: Vector2<f64>) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Add<Vector2<f64>> for Vector2<f64> {
    type Output = Vector2<f64>;
    fn add(self, rhs: Vector2<f64>) -> Vector2<f64> {
        Vector2::<f64> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl SubAssign<Vector2<f64>> for Vector2<f64> {
    fn sub_assign(&mut self, rhs: Vector2<f64>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Sub<Vector2<f64>> for Vector2<f64> {
    type Output = Vector2<f64>;
    fn sub(self, rhs: Vector2<f64>) -> Vector2<f64> {
        Vector2::<f64> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl MulAssign<f64> for Vector2<f64> {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f64> for Vector2<f64> {
    type Output = Vector2<f64>;
    fn div(self, rhs: f64) -> Vector2<f64> {
        assert_ne!(rhs, 0.0);
        let inv: f64 = 1.0 / rhs;
        Vector2::<f64> {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

impl DivAssign<f64> for Vector2<f64> {
    fn div_assign(&mut self, rhs: f64) {
        assert_ne!(rhs, 0.0);
        let inv: f64 = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
    }
}

impl<T> Neg for Vector2<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Vector2<T>;
    fn neg(self) -> Vector2<T> {
        Vector2::<T> {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<T> From<Vector2<T>> for Point2<T> {
    fn from(v: Vector2<T>) -> Self {
        Point2::<T> { x: v.x, y: v.y }
    }
}

pub fn vec2_dot(v1: &Vector2<f64>, v2: &Vector2<f64>) -> f64 {
    return v1.x * v2.x + v1.y * v2.y;
}

// Vector3 methods
impl<T> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3::<T> { x, y, z }
    }
}
impl Vector3<f64> {
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    pub fn abs(&self) -> Vector3<f64> {
        Vector3::<f64> {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
    pub fn length_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl Vector3<f64> {
    /// Compute a new vector pointing in the same direction but with unit
    /// length.
    #[inline]
    pub fn normalize(&self) -> Vector3<f64> {
        if self.length() == 0.0 {
            *self
        } else {
            *self / self.length()
        }
    }
}

impl<T: Copy> Cxyz<T> for Vector3<T> {
    fn to_xyz(&self) -> (T, T, T) {
        return (self.x, self.y, self.z);
    }
    fn from_xyz(x: T, y: T, z: T) -> Self {
        return Self { x: x, y: y, z: z };
    }
}

impl PartialEq for Vector3f {
    fn eq(&self, rhs: &Vector3f) -> bool {
        nearly_equal(self, rhs)
    }
}

impl AddAssign<Vector3<f64>> for Vector3<f64> {
    fn add_assign(&mut self, rhs: Vector3<f64>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Add for Vector3<f64> {
    type Output = Vector3<f64>;
    fn add(self, rhs: Vector3<f64>) -> Vector3<f64> {
        Vector3::<f64> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl SubAssign<Vector3<f64>> for Vector3<f64> {
    fn sub_assign(&mut self, rhs: Vector3<f64>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Sub for Vector3<f64> {
    type Output = Vector3<f64>;
    fn sub(self, rhs: Vector3<f64>) -> Vector3<f64> {
        Vector3::<f64> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vector3<f64> {
    type Output = Vector3<f64>;
    #[inline]
    fn mul(self, rhs: f64) -> Vector3<f64> {
        Vector3::<f64> {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<f64> for Vector3<f64> {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Vector3<f64> {
    type Output = Vector3<f64>;
    fn div(self, rhs: f64) -> Vector3<f64> {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        Vector3::<f64> {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl DivAssign<f64> for Vector3<f64> {
    fn div_assign(&mut self, rhs: f64) {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl<T> Neg for Vector3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Vector3<T>;
    fn neg(self) -> Vector3<T> {
        Vector3::<T> {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Index<u8> for Vector3<f64> {
    type Output = f64;
    fn index(&self, index: u8) -> &f64 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl IndexMut<u8> for Vector3<f64> {
    fn index_mut(&mut self, index: u8) -> &mut f64 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl<T> From<Point3<T>> for Vector3<T> {
    #[inline]
    fn from(p: Point3<T>) -> Self {
        Vector3::<T> {
            x: p.x,
            y: p.y,
            z: p.z,
        }
    }
}

impl<T> From<Normal3<T>> for Vector3<T> {
    fn from(n: Normal3<T>) -> Self {
        Vector3::<T> {
            x: n.x,
            y: n.y,
            z: n.z,
        }
    }
}

pub fn convert_pnt3<T, U>(incoming: Point3<T>) -> Point3<U>
where
    T: Into<U>,
{
    Point3::<U>::new(incoming.x.into(), incoming.y.into(), incoming.z.into())
}

/// Given two vectors in 3D, the cross product is a vector that is
/// perpendicular to both of them.
pub fn cross(a: &impl Cxyz<f64>, b: &impl Cxyz<f64>) -> Vector3f {
    let (v1x, v1y, v1z) = a.to_xyz();
    let (v2x, v2y, v2z) = b.to_xyz();
    Vector3f::new(
        (v1y * v2z) - (v1z * v2y),
        (v1z * v2x) - (v1x * v2z),
        (v1x * v2y) - (v1y * v2x),
    )
}

/// Return the largest coordinate value.
pub fn max_component(v: &impl Cxyz<f64>) -> f64 {
    let (x, y, z) = v.to_xyz();
    x.max(y).max(z)
}

/// Return the index of the component with the largest value.
pub fn vec3_max_dimension(v: &Vector3<f64>) -> usize {
    if v.x > v.y {
        if v.x > v.z {
            0_usize
        } else {
            2_usize
        }
    } else if v.y > v.z {
        1_usize
    } else {
        2_usize
    }
}

/// Permute the coordinate values according to the povided
/// permutation.
pub fn vec3_permute(v: &Vector3<f64>, x: usize, y: usize, z: usize) -> Vector3<f64> {
    let v3: [f64; 3] = [v.x, v.y, v.z];
    let xp: f64 = v3[x];
    let yp: f64 = v3[y];
    let zp: f64 = v3[z];
    Vector3::<f64> {
        x: xp,
        y: yp,
        z: zp,
    }
}

/// Construct a local coordinate system given only a single 3D vector.
#[inline]
pub fn vec3_coordinate_system(v1: &Vector3f, v2: &mut Vector3f, v3: &mut Vector3f) {
    if v1.x.abs() > v1.y.abs() {
        *v2 = Vector3f {
            x: -v1.z,
            y: 0.0 as f64,
            z: v1.x,
        } / (v1.x * v1.x + v1.z * v1.z).sqrt();
    } else {
        *v2 = Vector3f {
            x: 0.0 as f64,
            y: v1.z,
            z: -v1.y,
        } / (v1.y * v1.y + v1.z * v1.z).sqrt();
    }
    *v3 = cross(v1, &*v2);
}

/// Calculate appropriate direction vector from two angles.
pub fn spherical_direction(sin_theta: f64, cos_theta: f64, phi: f64) -> Vector3f {
    Vector3f {
        x: sin_theta * phi.cos(),
        y: sin_theta * phi.sin(),
        z: cos_theta,
    }
}

/// Take three basis vectors representing the x, y, and z axes and
/// return the appropriate direction vector with respect to the
/// coordinate frame defined by them.
pub fn spherical_direction_vec3(
    sin_theta: f64,
    cos_theta: f64,
    phi: f64,
    x: &Vector3f,
    y: &Vector3f,
    z: &Vector3f,
) -> Vector3f {
    *x * (sin_theta * phi.cos()) + *y * (sin_theta * phi.sin()) + *z * cos_theta
}

/// Conversion of a direction to spherical angles. Note that
/// **spherical_theta()** assumes that the vector **v** has been
/// normalized before being passed in.
pub fn spherical_theta(v: &Vector3f) -> f64 {
    clamp_t(v.z, -1.0 as f64, 1.0 as f64).acos()
}

/// Conversion of a direction to spherical angles.
pub fn spherical_phi(v: &Vector3f) -> f64 {
    let p: f64 = v.y.atan2(v.x);
    if p < 0.0 as f64 {
        p + 2.0 as f64 * PI
    } else {
        p
    }
}

// Normal3 Methods

impl Normal3<f64> {
    /// Compute a new normal pointing in the same direction but with unit
    /// length.
    #[inline]
    pub fn normalize(&self) -> Normal3<f64> {
        *self / self.length()
    }
}

impl<T> Normal3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Normal3::<T> { x, y, z }
    }
    pub fn to_vec3(self) -> Vector3<T> {
        Vector3::<T> {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<T: Copy> Cxyz<T> for Normal3<T> {
    fn to_xyz(&self) -> (T, T, T) {
        return (self.x, self.y, self.z);
    }
    fn from_xyz(x: T, y: T, z: T) -> Self {
        return Self { x: x, y: y, z: z };
    }
}

impl<T> Add for Normal3<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Normal3<T>;
    fn add(self, rhs: Normal3<T>) -> Normal3<T> {
        Normal3::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> Sub for Normal3<T>
where
    T: Copy + Sub<T, Output = T>,
{
    type Output = Normal3<T>;
    fn sub(self, rhs: Normal3<T>) -> Normal3<T> {
        Normal3::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Mul<T> for Normal3<T>
where
    T: Copy + Mul<T, Output = T>,
{
    type Output = Normal3<T>;
    fn mul(self, rhs: T) -> Normal3<T>
    where
        T: Copy + Mul<T, Output = T>,
    {
        Normal3::<T> {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> MulAssign<T> for Normal3<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T> Neg for Normal3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Normal3<T>;
    fn neg(self) -> Normal3<T> {
        Normal3::<T> {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> Index<u8> for Normal3<T> {
    type Output = T;
    fn index(&self, index: u8) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl<T> IndexMut<u8> for Normal3<T> {
    fn index_mut(&mut self, index: u8) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Check failed: i >= 0 && i <= 2"),
        }
    }
}

impl Normal3<f64> {
    pub fn length_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl PartialEq for Normal3f {
    fn eq(&self, rhs: &Normal3f) -> bool {
        nearly_equal(self, rhs)
    }
}

// work around bug
// https://github.com/rust-lang/rust/issues/40395
impl Div<f64> for Normal3<f64> {
    type Output = Normal3<f64>;
    fn div(self, rhs: f64) -> Normal3<f64> {
        assert_ne!(rhs, 0.0 as f64);
        let inv: f64 = 1.0 as f64 / rhs;
        Normal3::<f64> {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl<T> From<Vector3<T>> for Normal3<T> {
    fn from(v: Vector3<T>) -> Self {
        Normal3::<T> {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

/// Return normal with the absolute value of each coordinate.
pub fn nrm_abs(n: &Normal3<f64>) -> Normal3<f64> {
    Normal3::<f64> {
        x: n.x.abs(),
        y: n.y.abs(),
        z: n.z.abs(),
    }
}

/// Flip a surface normal so that it lies in the same hemisphere as a
/// given vector/normal.
pub fn faceforward(n: &Normal3f, v: &impl Cxyz<f64>) -> Normal3f {
    if dot3(n, v) < 0.0 as f64 {
        -(*n)
    } else {
        *n
    }
}

impl<T> Bounds2<T> {
    pub fn new(p1: Point2<T>, p2: Point2<T>) -> Self
    where
        T: Copy + PartialOrd,
    {
        let p_min: Point2<T> = Point2::<T> {
            x: if p1.x > p2.x { p2.x } else { p1.x },
            y: if p1.y > p2.y { p2.y } else { p1.y },
        };
        let p_max: Point2<T> = Point2::<T> {
            x: if p1.x > p2.x { p1.x } else { p2.x },
            y: if p1.y > p2.y { p1.y } else { p2.y },
        };
        Bounds2::<T> { p_min, p_max }
    }
    pub fn diagonal(&self) -> Vector2<T>
    where
        T: Copy + Sub<T, Output = T>,
    {
        self.p_max - self.p_min
    }
    pub fn area(&self) -> T
    where
        T: Copy + Sub<T, Output = T> + Mul<T, Output = T>,
    {
        let d: Vector2<T> = self.p_max - self.p_min;
        d.x * d.y
    }
    /// Determine if a given point is inside the bounding box.
    pub fn inside(pt: &Point2<T>, b: &Bounds2<T>) -> bool
    where
        T: PartialOrd,
    {
        pt.x >= b.p_min.x && pt.x <= b.p_max.x && pt.y >= b.p_min.y && pt.y <= b.p_max.y
    }

    /// Is a 2D point inside a 2D bound?
    pub fn inside_exclusive(pt: &Point2<T>, b: &Bounds2<T>) -> bool
    where
        T: PartialOrd,
    {
        pt.x >= b.p_min.x && pt.x < b.p_max.x && pt.y >= b.p_min.y && pt.y < b.p_max.y
    }
    pub fn union(b: &Self, p: &Point2<T>) -> Self
    where
        Point2<T>: PointMin + PointMax,
    {
        let p_min = b.p_min.min(p);
        let p_max = b.p_max.max(p);
        Bounds2 { p_min, p_max }
    }
    pub fn union_bnd(b: &Self, other: &Self) -> Self
    where
        Point2<T>: PointMin + PointMax,
    {
        let p_min = b.p_min.min(&other.p_min);
        let p_max = b.p_max.max(&other.p_max);
        Bounds2 { p_min, p_max }
    }
    pub fn expand(&self, delta: T) -> Self
    where
        T: Copy + PartialOrd + PartialEq + Add + Sub,
        Point2<T>: Add<T, Output = Point2<T>> + Sub<T, Output = Point2<T>>,
    {
        Self::new(self.p_min - delta, self.p_max - delta)
    }
}

impl Bounds2<f64> {
    pub fn lerp(&self, t: &Point2f) -> Point2f {
        Point2f {
            x: lerp(t.x, self.p_min.x, self.p_max.x),
            y: lerp(t.y, self.p_min.y, self.p_max.y),
        }
    }
    pub fn offset(&self, p: Point2f) -> Vector2f {
        let mut o: Vector2f = p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x /= self.p_max.x - self.p_min.x;
        }
        if self.p_max.y > self.p_min.y {
            o.y /= self.p_max.y - self.p_min.y;
        }
        o
    }
}

pub fn convert_bnd2<T, U>(incoming: Bounds2<T>) -> Bounds2<U>
where
    U: From<T> + Copy + std::cmp::PartialOrd,
{
    Bounds2::<U>::new(convert_pnt2(incoming.p_min), convert_pnt2(incoming.p_max))
}

impl From<Bounds2i> for Bounds2f {
    fn from(b: Bounds2i) -> Self {
        Self::new(b.p_min.into(), b.p_max.into())
    }
}

pub struct Bounds2Iterator<'a> {
    p: Point2i,
    bounds: &'a Bounds2i,
}

impl<'a> Iterator for Bounds2Iterator<'a> {
    type Item = Point2i;

    fn next(&mut self) -> Option<Point2i> {
        self.p.x += 1;
        if self.p.x == self.bounds.p_max.x {
            self.p.x = self.bounds.p_min.x;
            self.p.y += 1;
        }
        if self.p.y == self.bounds.p_max.y {
            None
        } else {
            Some(self.p)
        }
    }
}

impl<'a> IntoIterator for &'a Bounds2i {
    type Item = Point2i;
    type IntoIter = Bounds2Iterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Bounds2Iterator {
            // need to start 1 before p_min.x as next() will be called
            // to get the first element
            p: Point2i {
                x: self.p_min.x - 1,
                y: self.p_min.y,
            },
            bounds: self,
        }
    }
}

/// The intersection of two bounding boxes can be found by computing
/// the maximum of their two respective minimum coordinates and the
/// minimum of their maximum coordinates.
pub fn bnd2_intersect_bnd2<T>(b1: &Bounds2<T>, b2: &Bounds2<T>) -> Bounds2<T>
where
    T: Copy + Ord,
{
    Bounds2::<T> {
        p_min: Point2::<T> {
            x: std::cmp::max(b1.p_min.x, b2.p_min.x),
            y: std::cmp::max(b1.p_min.y, b2.p_min.y),
        },
        p_max: Point2::<T> {
            x: std::cmp::min(b1.p_max.x, b2.p_max.x),
            y: std::cmp::min(b1.p_max.y, b2.p_max.y),
        },
    }
}

// work around bug
// https://github.com/rust-lang/rust/issues/40395
impl Default for Bounds3<f64> {
    fn default() -> Bounds3<f64> {
        let min_num: f64 = std::f64::MIN;
        let max_num: f64 = std::f64::MAX;
        // Bounds3f
        Bounds3::<f64> {
            p_min: Point3f {
                x: max_num,
                y: max_num,
                z: max_num,
            },
            p_max: Point3f {
                x: min_num,
                y: min_num,
                z: min_num,
            },
        }
    }
}

impl<T: Copy> Bounds3<T> {
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self
    where
        T: Copy + PartialOrd,
    {
        let p_min: Point3<T> = Point3::<T> {
            x: if p1.x > p2.x { p2.x } else { p1.x },
            y: if p1.y > p2.y { p2.y } else { p1.y },
            z: if p1.z > p2.z { p2.z } else { p1.z },
        };
        let p_max: Point3<T> = Point3::<T> {
            x: if p1.x > p2.x { p1.x } else { p2.x },
            y: if p1.y > p2.y { p1.y } else { p2.y },
            z: if p1.z > p2.z { p1.z } else { p2.z },
        };
        Bounds3::<T> { p_min, p_max }
    }
    pub fn corner(&self, corner: u8) -> Point3<T>
    where
        T: Copy,
    {
        // assert!(corner >= 0_u8);
        assert!(corner < 8_u8);
        let x: T;
        if corner & 1 == 0 {
            x = self.p_min.x;
        } else {
            x = self.p_max.x;
        }
        let y: T;
        if corner & 2 == 0 {
            y = self.p_min.y;
        } else {
            y = self.p_max.y;
        }
        let z: T;
        if corner & 4 == 0 {
            z = self.p_min.z;
        } else {
            z = self.p_max.z;
        }
        Point3::<T> { x, y, z }
    }
    pub fn diagonal(&self) -> Vector3<T>
    where
        T: Copy + Sub<T, Output = T>,
    {
        self.p_max - self.p_min
    }
    pub fn surface_area(&self) -> T
    where
        T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T>,
    {
        let d: Vector3<T> = self.diagonal();
        // 2 * (d.x * d.y + d.x * d.z + d.y * d.z)
        let r: T = d.x * d.y + d.x * d.z + d.y * d.z;
        r + r // avoid '2 *'
    }
    pub fn maximum_extent(&self) -> u8
    where
        T: Copy + std::cmp::PartialOrd + Sub<T, Output = T>,
    {
        let d: Vector3<T> = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0_u8
        } else if d.y > d.z {
            1_u8
        } else {
            2_u8
        }
    }
    pub fn offset(&self, p: &Point3<T>) -> Vector3<T>
    where
        T: Copy + std::cmp::PartialOrd + Sub<T, Output = T> + DivAssign<T>,
    {
        let mut o: Vector3<T> = *p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x /= self.p_max.x - self.p_min.x;
        }
        if self.p_max.y > self.p_min.y {
            o.y /= self.p_max.y - self.p_min.y;
        }
        if self.p_max.z > self.p_min.z {
            o.z /= self.p_max.z - self.p_min.z;
        }
        o
    }
    pub fn bounding_sphere(b: &Bounds3f, center: &mut Point3f, radius: &mut f64) {
        let p_min: Point3f = b.p_min as Point3f;
        let p_max: Point3f = b.p_max as Point3f;
        let sum: Point3f = p_min + p_max;
        *center = sum / 2.0;
        let center_copy: Point3f = *center as Point3f;
        let is_inside: bool = Bounds3f::inside(&center_copy, b);
        if is_inside {
            *radius = pnt3_distance(&center_copy, &p_max);
        } else {
            *radius = 0.0;
        }
    }
    /// Determine if a given point is inside the bounding box.
    pub fn inside(pt: &Point3<T>, b: &Bounds3<T>) -> bool
    where
        T: PartialOrd,
    {
        pt.x >= b.p_min.x
            && pt.x <= b.p_max.x
            && pt.y >= b.p_min.y
            && pt.y <= b.p_max.y
            && pt.z >= b.p_min.z
            && pt.z <= b.p_max.z
    }
    /// Is a 2D point inside a 2D bound?
    pub fn inside_exclusive(pt: &Point3<T>, b: &Bounds3<T>) -> bool
    where
        T: PartialOrd,
    {
        pt.x >= b.p_min.x
            && pt.x < b.p_max.x
            && pt.y >= b.p_min.y
            && pt.y < b.p_max.y
            && pt.z > b.p_min.z
            && pt.z < b.p_max.z
    }
    pub fn union(b: &Self, p: &Point3<T>) -> Self
    where
        Point3<T>: PointMin + PointMax,
    {
        let p_min = b.p_min.min(p);
        let p_max = b.p_max.max(p);
        Bounds3 { p_min, p_max }
    }
    pub fn union_bnd(b: &Self, other: &Self) -> Self
    where
        Point3<T>: PointMin + PointMax,
    {
        let p_min = b.p_min.min(&other.p_min);
        let p_max = b.p_max.max(&other.p_max);
        Bounds3 { p_min, p_max }
    }
    pub fn expand(&self, delta: T) -> Self
    where
        T: Copy + PartialOrd + PartialEq + Add + Sub,
        Point3<T>: Add<T, Output = Point3<T>> + Sub<T, Output = Point3<T>>,
    {
        Self::new(self.p_min - delta, self.p_max - delta)
    }
}

impl Bounds3<f64> {
    // pub fn new(p1: Point3<f64>, p2: Point3<f64>) -> Self {
    //     let p_min: Point3<f64> = Point3::<f64> {
    //         x: p1.x.min(p2.x),
    //         y: p1.y.min(p2.y),
    //         z: p1.z.min(p2.z),
    //     };
    //     let p_max: Point3<f64> = Point3::<f64> {
    //         x: p1.x.max(p2.x),
    //         y: p1.y.max(p2.y),
    //         z: p1.z.max(p2.z),
    //     };
    //     Bounds3::<f64> { p_min, p_max }
    // }
    pub fn lerp(&self, t: &Point3f) -> Point3f {
        Point3f {
            x: lerp(t.x, self.p_min.x as f64, self.p_max.x as f64),
            y: lerp(t.y, self.p_min.y as f64, self.p_max.y as f64),
            z: lerp(t.z, self.p_min.z as f64, self.p_max.z as f64),
        }
    }
    pub fn intersect_b(&self, r: &Ray, hitt0: &mut f64, hitt1: &mut f64) -> bool {
        let mut t0 = 0.0;
        let mut t1 = MAX_DIST;
        for i in 0..3 {
            // update interval for _i_th bounding box slab
            let inv_ray_dir = 1.0 / r.d[i];
            let mut t_near = (self.p_min[i] - r.o[i]) * inv_ray_dir;
            let mut t_far = (self.p_max[i] - r.o[i]) * inv_ray_dir;
            // update parametric interval from slab intersection $t$ values
            if t_near > t_far {
                std::mem::swap(&mut t_near, &mut t_far);
            }
            // update _t_far_ to ensure robust ray--bounds intersection
            t_far *= 1.0 + 2.0 * gamma(3);
            if t_near > t0 {
                t0 = t_near;
            }
            if t_far < t1 {
                t1 = t_far;
            }
            if t0 > t1 {
                return false;
            }
        }
        *hitt0 = t0;
        *hitt1 = t1;
        true
    }
    pub fn intersect_p(&self, ray: &Ray, inv_dir: &Vector3f, dir_is_neg: &[u8; 3]) -> bool {
        // check for ray intersection against $x$ and $y$ slabs
        let mut t_min = (self[dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
        let mut t_max = (self[1_u8 - dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
        let ty_min = (self[dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
        let mut ty_max = (self[1_u8 - dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
        // update _t_max_ and _ty_max_ to ensure robust bounds intersection
        t_max *= 1.0 + 2.0 * gamma(3);
        ty_max *= 1.0 + 2.0 * gamma(3);
        if t_min > ty_max || ty_min > t_max {
            return false;
        }
        if ty_min > t_min {
            t_min = ty_min;
        }
        if ty_max < t_max {
            t_max = ty_max;
        }
        // check for ray intersection against $z$ slab
        let tz_min = (self[dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
        let mut tz_max = (self[1_u8 - dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
        // update _tz_max_ to ensure robust bounds intersection
        tz_max *= 1.0 + 2.0 * gamma(3);
        if t_min > tz_max || tz_min > t_max {
            return false;
        }
        if tz_min > t_min {
            t_min = tz_min;
        }
        if tz_max < t_max {
            t_max = tz_max;
        }
        (t_min < ray.t_max) && (t_max > 0.0)
    }
}

impl PartialEq for Bounds3f {
    fn eq(&self, rhs: &Bounds3f) -> bool {
        nearly_equal(&self.p_min, &rhs.p_min) && nearly_equal(&self.p_max, &rhs.p_max)
    }
}

impl<T> Index<u8> for Bounds3<T> {
    type Output = Point3<T>;
    #[inline]
    fn index(&self, i: u8) -> &Point3<T> {
        match i {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Invalid index!"),
        }
    }
}

pub fn convert_bnd3<T, U>(incoming: Bounds3<T>) -> Bounds3<U>
where
    U: From<T> + Copy + std::cmp::PartialOrd,
{
    Bounds3::<U>::new(convert_pnt3(incoming.p_min), convert_pnt3(incoming.p_max))
}

impl Default for Ray {
    fn default() -> Self {
        Ray {
            o: Point3f::default(),
            d: Vector3f::default(),
            t_max: INFINITY,
            time: 0.0,
            medium: None,
        }
    }
}

impl Ray {
    pub fn new(o: Point3f, d: Vector3f, t_max: f64, time: f64, medium: MediumOpArc) -> Ray {
        Ray {
            o,
            d: d.normalize(),
            t_max,
            time,
            medium,
        }
    }
    pub fn new_od(o: Point3f, d: Vector3f) -> Ray {
        Ray {
            o,
            d: d.normalize(),
            t_max: INFINITY,
            time: 0.0,
            medium: None,
        }
    }
    pub fn position(&self, t: f64) -> Point3f {
        self.o + self.d * t
    }
}

impl RayDifferential {
    pub fn new(
        ray: Ray,
        has_differentials: bool,
        rx_origin: Point3f,
        ry_origin: Point3f,
        rx_direction: Vector3f,
        ry_direction: Vector3f,
    ) -> Self {
        Self {
            ray,
            has_differentials,
            rx_origin,
            ry_origin,
            rx_direction,
            ry_direction,
        }
    }

    pub fn scale_differentials(&mut self, s: f64) {
        self.rx_origin = self.ray.o + (self.rx_origin - self.ray.o) * s;
        self.ry_origin = self.ray.o + (self.ry_origin - self.ray.o) * s;
        self.rx_direction = self.ray.d + (self.rx_direction - self.ray.d) * s;
        self.ry_direction = self.ray.d + (self.ry_direction - self.ray.d) * s;
    }
}

impl From<Ray> for RayDifferential {
    fn from(r: Ray) -> Self {
        Self::new(
            r,
            false,
            Point3f::default(),
            Point3f::default(),
            Vector3f::default(),
            Vector3f::default(),
        )
    }
}

pub fn nearly_equal(a: &impl Cxyz<f64>, b: &impl Cxyz<f64>) -> bool {
    let (ax, ay, az) = a.to_xyz();
    let (bx, by, bz) = b.to_xyz();
    float_nearly_equal(ax, bx) && float_nearly_equal(ay, by) && float_nearly_equal(az, bz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SMALL;

    #[test]
    fn test_pnt3() {
        let c = Point3f::new(0.0, 0.0, 0.0);
        let _d = c + 1.0;
    }

    #[test]
    fn test_vec3() {
        let sv = Vector3f {
            x: 3.0,
            y: 4.0,
            z: -5.0,
        };

        assert!(sv.length_squared() == 50.0);

        let e = Vector3f {
            x: 8.1,
            y: 10.8,
            z: -13.5,
        };
        println!("{:?}\n{:?}", (sv * 2.7), e);

        assert!(&(sv * 2.7) == &e);

        // dot product cross product
        let dp = dot3(&sv, &e);
        assert!((dp - 135.0).abs() <= SMALL);

        let cv = Vector3f::new(9.6, -12.4, 3.7);
        assert!(&cross(&sv, &cv) == &Vector3f::new(-47.2, -59.1, -75.6))
    }

    #[test]
    fn test_bound3() {
        let a = Bounds3f::new(
            Point3f::new(0.0, -10.0, 5.0),
            Point3f::new(-10.0, 20.0, 10.0),
        );

        let d = Point3f::new(-15.0, 10.0, 30.0);

        // EXPECT_EQ(Bounds3f(Point3f(-15, -10, 5), Point3f(0, 20, 30)), e);
        let e = Bounds3f::union(&a, &d);
        let r = Bounds3f {
            p_min: Point3f::new(-15.0, -10.0, 5.0),
            p_max: Point3f::new(0.0, 20.0, 30.0),
        };
        assert!(&r == &e);

        let mut bs_c: Point3f = Point3f::default();
        let mut bs_r: f64 = 0.0;
        Bounds3f::bounding_sphere(&a, &mut bs_c, &mut bs_r);
        assert!(&bs_c == &Point3f::new(-5.0, 5.0, 7.5));

        nearly_equal(&a.p_min, &a.p_max);
    }
}
