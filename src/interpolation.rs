use crate::misc::find_interval;

/// Importance sampling of 2D functions via spline interpolants.
pub fn sample_catmull_rom_2d(
    nodes1: &[f64],
    nodes2: &[f64],
    values: &[f64],
    cdf: &[f64],
    alpha: f64,
    u: f64,
    fval: Option<&mut f64>,
    pdf: Option<&mut f64>,
) -> f64 {
    let size2: i32 = nodes2.len() as i32;
    // local copy
    let mut u: f64 = u;
    // determine offset and coefficients for the _alpha_ parameter
    let mut offset: i32 = 0;
    let mut weights: [f64; 4] = [0.0 as f64; 4];
    if !catmull_rom_weights(nodes1, alpha, &mut offset, &mut weights) {
        return 0.0 as f64;
    }
    // define a lambda function to interpolate table entries
    let interpolate = |array: &[f64], idx: i32| -> f64 {
        let mut value: f64 = 0.0;
        for (i, weight) in weights.iter().enumerate() {
            if *weight != 0.0 as f64 {
                let index: i32 = (offset + i as i32) * size2 + idx;
                assert!(index >= 0);
                value += array[index as usize] * *weight;
            }
        }
        value
    };
    // map _u_ to a spline interval by inverting the interpolated _cdf_
    let maximum: f64 = interpolate(cdf, size2 - 1_i32);
    u *= maximum;
    let idx: i32 = find_interval(size2, |index| interpolate(cdf, index) <= u);
    // look up node positions and interpolated function values
    let f0: f64 = interpolate(values, idx);
    let f1: f64 = interpolate(values, idx + 1);
    assert!(idx >= 0);
    let x0: f64 = nodes2[idx as usize];
    let x1: f64 = nodes2[(idx + 1) as usize];
    let width: f64 = x1 - x0;
    // re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;
    // approximate derivatives using finite differences of the interpolant
    let d0: f64;
    let d1: f64;
    if idx > 0_i32 {
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[(idx - 1) as usize]);
    } else {
        d0 = f1 - f0;
    }
    if (idx + 2) < size2 {
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[(idx + 2) as usize] - x0);
    } else {
        d1 = f1 - f0;
    }

    // invert definite integral over spline segment and return solution

    // set initial guess for $t$ by importance sampling a linear interpolant
    let mut t = if f0 != f1 {
        (f0 - (0.0 as f64)
            .max(f0 * f0 + 2.0 as f64 * u * (f1 - f0))
            .sqrt())
            / (f0 - f1)
    } else {
        u / f0
    };
    let mut a: f64 = 0.0;
    let mut b: f64 = 1.0;
    let mut f_hat;
    let mut fhat;
    loop {
        // fall back to a bisection step when _t_ is out of bounds
        if !(t >= a && t <= b) {
            t = 0.5 as f64 * (a + b);
        }
        // evaluate target function and its derivative in Horner form
        f_hat = t
            * (f0
                + t * (0.5 as f64 * d0
                    + t * ((1.0 as f64 / 3.0 as f64) * (-2.0 as f64 * d0 - d1) + f1 - f0
                        + t * (0.25 as f64 * (d0 + d1) + 0.5 as f64 * (f0 - f1)))));
        fhat = f0
            + t * (d0
                + t * (-2.0 as f64 * d0 - d1
                    + 3.0 as f64 * (f1 - f0)
                    + t * (d0 + d1 + 2.0 as f64 * (f0 - f1))));
        // stop the iteration if converged
        if (f_hat - u).abs() < 1e-6 as f64 || b - a < 1e-6 as f64 {
            break;
        }
        // update bisection bounds using updated _t_
        if (f_hat - u) < 0.0 as f64 {
            a = t;
        } else {
            b = t;
        }
        // perform a Newton step
        t -= (f_hat - u) / fhat;
    }
    // return the sample position and function value
    if let Some(fval) = fval {
        *fval = fhat;
    }
    if let Some(pdf) = pdf {
        *pdf = fhat / maximum;
    }
    x0 + width * t
}

pub fn catmull_rom_weights(
    nodes: &[f64],
    x: f64,
    offset: &mut i32,
    weights: &mut [f64; 4],
) -> bool {
    // return _false_ if _x_ is out of bounds
    if !(x >= *nodes.first().unwrap() && x <= *nodes.last().unwrap()) {
        return false;
    }
    // search for the interval _idx_ containing _x_
    let idx: i32 = find_interval(nodes.len() as i32, |index| nodes[index as usize] <= x);
    *offset = idx - 1;
    assert!(idx >= 0);
    let x0: f64 = nodes[idx as usize];
    let x1: f64 = nodes[(idx + 1) as usize];
    // compute the $t$ parameter and powers
    let t: f64 = (x - x0) / (x1 - x0);
    let t2: f64 = t * t;
    let t3: f64 = t2 * t;
    // compute initial node weights $w_1$ and $w_2$
    weights[1] = 2.0 as f64 * t3 - 3.0 as f64 * t2 + 1.0 as f64;
    weights[2] = -2.0 as f64 * t3 + 3.0 as f64 * t2;
    // compute first node weight $w_0$
    if idx > 0_i32 {
        let w0: f64 = (t3 - 2.0 as f64 * t2 + t) * (x1 - x0) / (x1 - nodes[(idx - 1) as usize]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        let w0: f64 = t3 - 2.0 as f64 * t2 + t;
        weights[0] = 0.0 as f64;
        weights[1] -= w0;
        weights[2] += w0;
    }
    // compute last node weight $w_3$
    if (idx + 2) < nodes.len() as i32 {
        let w3: f64 = (t3 - t2) * (x1 - x0) / (nodes[(idx + 2) as usize] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        let w3: f64 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0 as f64;
    }
    true
}

pub fn integrate_catmull_rom(
    n: i32,
    x: &[f64],
    offset: usize,
    values: &[f64],
    cdf: &mut Vec<f64>,
) -> f64 {
    let mut sum: f64 = 0.0;
    cdf[offset] = 0.0 as f64;
    for i in 0..(n - 1) as usize {
        // look up $x_i$ and function values of spline segment _i_
        let x0: f64 = x[i];
        let x1: f64 = x[i + 1];
        let f0: f64 = values[offset + i];
        let f1: f64 = values[offset + i + 1];
        let width: f64 = x1 - x0;
        // approximate derivatives using finite differences
        let d0: f64;
        let d1: f64;
        if i > 0 {
            d0 = width * (f1 - values[offset + i - 1]) / (x1 - x[i - 1]);
        } else {
            d0 = f1 - f0;
        }
        if i + 2 < n as usize {
            d1 = width * (values[offset + i + 2] - f0) / (x[i + 2] - x0);
        } else {
            d1 = f1 - f0;
        }
        // keep a running sum and build a cumulative distribution function
        sum += ((d0 - d1) * (1.0 as f64 / 12.0 as f64) + (f0 + f1) * 0.5 as f64) * width;
        cdf[offset + i + 1] = sum;
    }
    sum
}
