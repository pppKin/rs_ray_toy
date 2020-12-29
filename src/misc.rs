use crate::{
    geometry::{Point2f, Vector2f, Vector3f},
    rtoycore::MACHINE_EPSILON,
};
use rand::{prelude::ThreadRng, Rng};
use std::{
    f64::consts::PI,
    fs::File,
    io::{self, BufRead},
    ops::{
        Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul,
        MulAssign, Not, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    path::Path,
    rc::Rc,
    sync::Arc,
};

pub const PI_OVER_2: f64 = 1.570_796_326_794_896_619_23;
pub const PI_OVER_4: f64 = 0.785_398_163_397_448_309_61;
pub const ONE_MINUS_EPSILON: f64 = 1.0 - MACHINE_EPSILON;

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *f64* to *u64*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn float_to_bits(f: f64) -> u64 {
    // uint64_t ui;
    // memcpy(&ui, &f, sizeof(double));
    // return ui;
    let rui: u64;
    unsafe {
        let ui: u64 = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Error propagation.
#[inline]
pub fn gamma(n: i64) -> f64 {
    (n as f64 * MACHINE_EPSILON) / (1.0 - n as f64 * MACHINE_EPSILON)
}

/// Is used to write sRGB-compatible 8-bit image files.
#[inline]
pub fn gamma_correct(value: f64) -> f64 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 as f64 * value.powf((1.0 / 2.4) as f64) - 0.055
    }
}

/// Convert from angles expressed in degrees to radians.
pub fn radians(deg: f64) -> f64 {
    (PI / 180.0) * deg
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *u64* to *f64*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn bits_to_float(ui: u64) -> f64 {
    // float f;
    // memcpy(&f, &ui, sizeof(uint32_t));
    // return f;
    let rf: f64;
    unsafe {
        let f: f64 = std::mem::transmute_copy(&ui);
        rf = f;
    }
    rf
}

/// Bump a floating-point value up to the next greater representable
/// floating-point value.
pub fn next_float_up(v: f64) -> f64 {
    if v.is_infinite() && v > 0.0 {
        v
    } else {
        let new_v = if v == -0.0 { 0.0 } else { v };
        let mut ui: u64 = float_to_bits(new_v);
        if new_v >= 0.0 {
            ui += 1;
        } else {
            ui -= 1;
        }
        bits_to_float(ui)
    }
}

/// Bump a floating-point value down to the next smaller representable
/// floating-point value.
pub fn next_float_down(v: f64) -> f64 {
    if v.is_infinite() && v < 0.0 {
        v
    } else {
        let new_v = if v == 0.0 { -0.0 } else { v };
        let mut ui: u64 = float_to_bits(new_v);
        if new_v > 0.0 {
            ui -= 1;
        } else {
            ui += 1;
        }
        bits_to_float(ui)
    }
}

/// Clamp the given value *val* to lie between the values *low* and *high*.
pub fn clamp_t<T>(val: T, low: T, high: T) -> T
where
    T: PartialOrd,
{
    let r: T;
    if val < low {
        r = low;
    } else if val > high {
        r = high;
    } else {
        r = val;
    }
    r
}

/// CommonNum and CommonLogicalNum allows us to write some convenient generic helper function
// using inline here allows rustc to replace all these function calls with just plain number.
// rustc is smart!
pub trait CommonNum:
    Sized
    + Copy
    + Default
    + From<u8>
    + Add<Self, Output = Self>
    + AddAssign
    + Sub<Self, Output = Self>
    + SubAssign
    + Mul<Self, Output = Self>
    + MulAssign
    + Div<Self, Output = Self>
    + DivAssign
    + PartialEq
    + PartialOrd
{
    #[inline]
    fn num_one() -> Self {
        Self::from(1_u8)
    }
    #[inline]
    fn two() -> Self {
        Self::num_one() + Self::num_one()
    }
    #[inline]
    fn four() -> Self {
        Self::two() * Self::two()
    }
    #[inline]
    fn eight() -> Self {
        Self::four() * Self::two()
    }
    #[inline]
    fn sixteen() -> Self {
        Self::eight() * Self::two()
    }
    #[inline]
    fn thirtytwo() -> Self {
        Self::sixteen() * Self::two()
    }
}

impl CommonNum for i32 {
    // #[inline]
    // fn num_one() -> Self {
    //     1
    // }
}

impl CommonNum for i64 {
    // #[inline]
    // fn num_one() -> Self {
    //     1
    // }
}

impl CommonNum for f64 {
    // #[inline]
    // fn num_one() -> Self {
    //     1.0
    // }
}

impl CommonNum for usize {
    // #[inline]
    // fn num_one() -> Self {
    //     1
    // }
}

impl CommonNum for u32 {
    // #[inline]
    // fn num_one() -> Self {
    //     1
    // }
}

impl CommonNum for u64 {
    // #[inline]
    // fn num_one() -> Self {
    //     1
    // }
}

/// CommonNum and CommonLogicalNum allows us to write some convenient generic helper function
pub trait CommonLogicalNum:
    CommonNum
    + Shr<Self, Output = Self>
    + Shl<Self, Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign
    + BitXor<Self, Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    + ShrAssign
    + ShlAssign
{
}

impl CommonLogicalNum for usize {}
impl CommonLogicalNum for u32 {}
impl CommonLogicalNum for u64 {}
impl CommonLogicalNum for i32 {}
impl CommonLogicalNum for i64 {}

/// Interpolate linearly between two provided values.
pub fn lerp<T>(t: f64, a: T, b: T) -> T
where
    T: Mul<f64, Output = T> + Add<Output = T>,
{
    a * (1_f64 - t) + b * t
}

/// Find solution(s) of the quadratic equation at<sup>2</sup> + bt + c = 0.
pub fn quadratic(a: f64, b: f64, c: f64, t0: &mut f64, t1: &mut f64) -> bool {
    // find quadratic discriminant
    let discrim: f64 = (b as f64) * (b as f64) - 4.0 * (a as f64) * (c as f64);
    if discrim < 0.0 {
        false
    } else {
        let root_discrim: f64 = discrim.sqrt();
        // compute quadratic _t_ values
        let q = if b < 0.0 {
            -0.5 * (b as f64 - root_discrim)
        } else {
            -0.5 * (b as f64 + root_discrim)
        };
        *t0 = q as f64 / a;
        *t1 = c / q as f64;
        if *t0 > *t1 {
            std::mem::swap(&mut (*t0), &mut (*t1))
        }
        true
    }
}

/// Uniformly distribute samples over a unit disk.
pub fn concentric_sample_disk(u: Point2f) -> Point2f {
    // map uniform random numbers to $[-1,1]^2$
    let u_offset: Point2f = u * 2.0 as f64 - Vector2f { x: 1.0, y: 1.0 };
    // handle degeneracy at the origin
    if u_offset.x == 0.0 as f64 && u_offset.y == 0.0 as f64 {
        return Point2f::default();
    }
    // apply concentric mapping to point
    let theta: f64;
    let r: f64;
    if u_offset.x.abs() > u_offset.y.abs() {
        r = u_offset.x;
        theta = PI_OVER_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y);
    }
    Point2f {
        x: theta.cos(),
        y: theta.sin(),
    } * r
}

pub fn float_nearly_equal(a: f64, b: f64) -> bool {
    let a_abs = a.abs();
    let b_abs = b.abs();
    let diff = (a - b).abs();

    if a == b {
        return true;
    }
    if a == 0.0 || b == 0.0 || ((a_abs + b_abs) < std::f64::MIN_POSITIVE) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (MACHINE_EPSILON * std::f64::MIN_POSITIVE);
    } else {
        // use relative error
        return diff / (a_abs + b_abs).min(std::f64::MAX) < MACHINE_EPSILON;
    }
}

/// Cosine-weighted hemisphere sampling using Malley's method.
#[inline]
pub fn cosine_sample_hemisphere(u: Point2f) -> Vector3f {
    let d: Point2f = concentric_sample_disk(u);
    let z: f64 = (0.0 as f64).max(1.0 - d.x * d.x - d.y * d.y).sqrt();
    Vector3f { x: d.x, y: d.y, z }
}

pub fn erf_inv(x: f64) -> f64 {
    let clamped_x: f64 = clamp_t(x, -0.99999, 0.99999);
    let mut w: f64 = -((1.0 as f64 - clamped_x) * (1.0 as f64 + clamped_x)).ln();
    let mut p: f64;
    if w < 5.0 as f64 {
        w -= 2.5 as f64;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_640 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
    } else {
        w = w.sqrt() - 3.0 as f64;
        p = -0.000_200_214_257;
        p = 0.000_100_950_558 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_44 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
    }
    p * clamped_x
}

pub fn erf(x: f64) -> f64 {
    // constants
    let a1: f64 = 0.254_829_592;
    let a2: f64 = -0.284_496_736;
    let a3: f64 = 1.421_413_741;
    let a4: f64 = -1.453_152_027;
    let a5: f64 = 1.061_405_429;
    let p: f64 = 0.327_591_1;
    // save the sign of x
    let sign = if x < 0.0 as f64 { -1.0 } else { 1.0 };
    let x: f64 = x.abs();
    // A&S formula 7.1.26
    let t: f64 = 1.0 as f64 / (1.0 as f64 + p * x);
    let y: f64 = 1.0 as f64 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Uniformly sample rays in a full sphere. Choose a direction.
pub fn uniform_sample_sphere(u: Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = (0.0 as f64).max(1.0 - z * z).sqrt();
    let phi = 2.0 * PI * u[1];
    Vector3f {
        x: r * phi.cos(),
        y: r * phi.sin(),
        z,
    }
}

/// Uniformly sample rays in a hemisphere. Choose a direction.
pub fn uniform_sample_hemisphere(u: &Point2f) -> Vector3f {
    let z: f64 = u[0];
    let r: f64 = (0.0 as f64).max(1.0 as f64 - z * z).sqrt();
    let phi: f64 = 2.0 as f64 * PI * u[1];
    Vector3f {
        x: r * phi.cos(),
        y: r * phi.sin(),
        z,
    }
}

/// Uniformly sample rays in a hemisphere. Probability density
/// function (PDF).
#[inline]
pub fn uniform_hemisphere_pdf() -> f64 {
    1.0 / (2.0 * PI)
}

pub fn round_up_pow2<T>(v: T) -> T
where
    T: CommonLogicalNum,
{
    let mut v = v;
    v = v - T::num_one();
    v |= v >> T::num_one();
    v |= v >> T::two();
    v |= v >> T::four();
    v |= v >> T::eight();
    v |= v >> T::sixteen();
    v + T::num_one()
}

/// Randomly permute an array of *count* sample values, each of which
/// has *n_dimensions* dimensions.
pub fn shuffle<T>(samp: &mut [T], count: u32, n_dimensions: u32, rng: &mut ThreadRng) {
    for i in 0..count {
        let other = i + rng.gen_range(0, count - i);
        for j in 0..n_dimensions {
            samp.swap(
                (n_dimensions * i + j) as usize,
                (n_dimensions * other + j) as usize,
            );
        }
    }
}

pub fn latin_hypercube(samples: &mut [Point2f], n_samples: u32, rng: &mut ThreadRng) {
    let n_dim: usize = 2;
    // generate LHS samples along diagonal
    let inv_n_samples: f64 = 1.0 as f64 / n_samples as f64;
    for i in 0..n_samples {
        for j in 0..n_dim {
            let sj: f64 = (i as f64 + (rng.gen_range(0.0, ONE_MINUS_EPSILON))) * inv_n_samples;
            if j == 0 {
                samples[i as usize].x = sj.min(ONE_MINUS_EPSILON);
            } else {
                samples[i as usize].y = sj.min(ONE_MINUS_EPSILON);
            }
        }
    }
    // permute LHS samples in each dimension
    for i in 0..n_dim {
        for j in 0..n_samples {
            let other: u32 = j as u32 + rng.gen_range(0, (n_samples - j) as u32);
            if i == 0 {
                let tmp = samples[j as usize].x;
                samples[j as usize].x = samples[other as usize].x;
                samples[other as usize].x = tmp;
            } else {
                let tmp = samples[j as usize].y;
                samples[j as usize].y = samples[other as usize].y;
                samples[other as usize].y = tmp;
            }
            // samples.swap(
            //     (n_dim * j + i) as usize,
            //     (n_dim * other + i) as usize,
            // );
        }
    }
}

/// Computes the remainder of a/b. Provides the behavior that the
/// modulus of a negative number is always positive.
pub fn mod_t<T>(a: T, b: T) -> T
where
    T: Copy
        + PartialOrd
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Default,
{
    let result: T = a - (a / b) * b;
    let zero = T::default();
    if result < zero {
        result + b
    } else {
        result
    }
}

/// Helper function which emulates the behavior of std::upper_bound().
pub fn find_interval<T, P>(size: T, pred: P) -> T
where
    T: CommonLogicalNum,
    P: Fn(T) -> bool,
{
    let mut first = T::default();
    let mut len = size;
    while len > T::default() {
        let half = len >> T::num_one();
        let middle = first + half;
        // bisect range based on value of _pred_ at _middle_
        if pred(middle) {
            first = middle + T::num_one();
            len -= half + T::num_one();
        } else {
            len = half;
        }
    }
    clamp_t(first - T::num_one(), T::default(), size - T::two())
}

pub fn copy_option_arc<T: ?Sized>(opa: &Option<Arc<T>>) -> Option<Arc<T>> {
    match opa {
        Some(a) => {
            return Some(Arc::clone(a));
        }
        None => {
            return None;
        }
    }
}

pub fn copy_option_rc<T: ?Sized>(opa: &Option<Rc<T>>) -> Option<Rc<T>> {
    match opa {
        Some(a) => {
            return Some(Rc::clone(a));
        }
        None => {
            return None;
        }
    }
}

pub fn is_power_of_2<T>(v: T) -> bool
where
    T: CommonLogicalNum,
{
    v != T::default() && (v & (v - T::num_one())) == T::default()
}
