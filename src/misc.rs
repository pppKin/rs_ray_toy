use crate::core::MACHINE_EPSILON;
use crate::geometry::{Point2f, Vector2f};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
pub const PI_OVER_2: f64 = 1.570_796_326_794_896_619_23;
pub const PI_OVER_4: f64 = 0.785_398_163_397_448_309_61;

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *f64* to *u32*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn float_to_bits(f: f64) -> u32 {
    // uint64_t ui;
    // memcpy(&ui, &f, sizeof(double));
    // return ui;
    let rui: u32;
    unsafe {
        let ui: u32 = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Error propagation.
#[inline]
pub fn gamma(n: i32) -> f64 {
    (n as f64 * MACHINE_EPSILON) / (1.0 - n as f64 * MACHINE_EPSILON)
}

/// Convert from angles expressed in degrees to radians.
pub fn radians(deg: f64) -> f64 {
    (PI / 180.0) * deg
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *u32* to *f64*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn bits_to_float(ui: u32) -> f64 {
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
        let mut ui: u32 = float_to_bits(new_v);
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
        let mut ui: u32 = float_to_bits(new_v);
        if new_v > 0.0 {
            ui -= 1;
        } else {
            ui += 1;
        }
        bits_to_float(ui)
    }
}

/// Clamp the given value *val* to lie between the values *low* and *high*.
#[inline]
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

/// Interpolate linearly between two provided values.
pub fn lerp(t: f64, a: f64, b: f64) -> f64 {
    let one: f64 = 1.0;
    a * (one - t) + b * t
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
