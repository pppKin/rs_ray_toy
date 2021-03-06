use crate::MACHINE_EPSILON;
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
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub const SHADOW_EPSILON: f64 = 0.0001;
pub const ONE_MINUS_EPSILON: f64 = 1.0 - MACHINE_EPSILON;
pub const INV_PI: f64 = 0.318_309_886_183_790_671_54;
pub const INV_2_PI: f64 = 0.159_154_943_091_895_335_77;
pub const INV_4_PI: f64 = 0.079_577_471_545_947_667_88;
pub const PI_OVER_2: f64 = 1.570_796_326_794_896_619_23;
pub const PI_OVER_4: f64 = 0.785_398_163_397_448_309_61;
pub const SQRT_2: f64 = 1.414_213_562_373_095_048_80;
pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn float_to_bits(f: f64) -> u64 {
    f.to_bits()
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

pub fn bits_to_float(ui: u64) -> f64 {
    f64::from_bits(ui)
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
    let discrim: f64 = (b) * (b) - 4.0 * (a) * (c);
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

#[derive(Debug, Default)]
pub struct AtomicF64 {
    v: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        Self {
            v: AtomicU64::new(value.to_bits()),
        }
    }
    pub fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(self.v.load(order))
    }
    pub fn store(&self, val: f64, order: Ordering) {
        self.v.store(val.to_bits(), order)
    }
    pub fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        let prev_val = self.v.fetch_add(val.to_bits(), order);
        f64::from_bits(prev_val)
    }
}
