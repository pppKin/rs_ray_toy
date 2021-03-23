use crate::{
    geometry::{Normal3f, Point2f, Point3f, Vector3f},
    misc::read_lines,
};
use std::str::FromStr;
use std::{error::Error, fmt::Display, str::SplitWhitespace};

#[derive(Debug, Default, Clone)]
pub struct ParseObjError {
    filename: String,
    err_line: u32,
    element_type: String,
    desc: String,
}

impl ParseObjError {
    pub fn new(filename: String, err_line: u32, element_type: String, desc: String) -> Self {
        Self {
            filename,
            err_line,
            element_type,
            desc,
        }
    }
}

impl Display for ParseObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error Parsing File: {}\n at line {} with element type {}, desc: {}",
            self.filename, self.err_line, self.element_type, self.desc
        )
    }
}

impl Error for ParseObjError {}

pub struct ParseResult {
    pub n_triangles: usize,
    pub n_vertices: usize,
    pub vertex_indices: Vec<usize>,
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub s: Vec<Vector3f>,
    pub uv: Vec<Point2f>,
}

impl ParseResult {
    pub fn new(
        vertex_indices: Vec<usize>,
        p: Vec<Point3f>,
        n: Vec<Normal3f>,
        s: Vec<Vector3f>,
        uv: Vec<Point2f>,
    ) -> Self {
        Self {
            n_triangles: (vertex_indices.len() / 3),
            n_vertices: p.len(),
            vertex_indices,
            p,
            n,
            s,
            uv,
        }
    }
}

pub fn parse_obj(filename: &str) -> Result<ParseResult, Box<dyn Error>> {
    let mut err = ParseObjError::default();
    err.filename = filename.to_string();
    let lines = read_lines(filename)?;

    let mut vertex_indices = vec![];
    let mut p = vec![];
    let mut n = vec![];
    let mut s = vec![];
    let mut uv = vec![];

    for line_r in lines {
        err.err_line += 1;
        err.element_type = "unknown".to_string();
        let tmp = line_r?;
        let mut sp = tmp.split_whitespace();
        match sp.next() {
            Some("v") => {
                // geometric vertices
                match make_vertex(&mut sp) {
                    Ok(vertex_p) => {
                        p.push(vertex_p);
                    }
                    Err(desc) => {
                        err.desc = desc;
                        return Err(Box::new(err));
                    }
                }
            }
            Some("vt") => match make_uv(&mut sp) {
                Ok(uv_co) => {
                    uv.push(uv_co);
                }
                Err(desc) => {
                    err.desc = desc;
                    return Err(Box::new(err));
                }
            },
            Some("vn") => {
                // geometric vertices
                match make_vertex(&mut sp) {
                    Ok(tmp_p) => {
                        n.push(Normal3f::new(tmp_p.x, tmp_p.y, tmp_p.z));
                    }
                    Err(desc) => {
                        err.desc = desc;
                        return Err(Box::new(err));
                    }
                }
            }
            Some("f") => {
                // geometric vertices
                match make_face(&mut sp) {
                    Ok((t_idx1, t_idx2, t_idx3)) => {
                        vertex_indices.push(t_idx1);
                        vertex_indices.push(t_idx2);
                        vertex_indices.push(t_idx3);
                    }
                    Err(desc) => {
                        err.desc = desc;
                        return Err(Box::new(err));
                    }
                }
            }
            unknown => {
                eprintln!("Unsupported Element {:?}", unknown)
            }
        }
    }
    Ok(ParseResult::new(vertex_indices, p, n, s, uv))
}

fn make_vertex(sp: &mut SplitWhitespace) -> Result<Point3f, String> {
    let v1 = f64::from_str(sp.next().ok_or("Failed to get v1")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v2 = f64::from_str(sp.next().ok_or("Failed to get v2")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v3 = f64::from_str(sp.next().ok_or("Failed to get v3")?)
        .or_else(|e| return Err(e.to_string()))?;
    Ok(Point3f::new(v1, v2, v3))
}

fn make_uv(sp: &mut SplitWhitespace) -> Result<Point2f, String> {
    let u = f64::from_str(sp.next().ok_or("Failed to get v1")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v = f64::from_str(sp.next().unwrap_or_default()).or_else(|e| return Err(e.to_string()))?;
    Ok(Point2f::new(u, v))
}

fn make_face(sp: &mut SplitWhitespace) -> Result<(usize, usize, usize), String> {
    let v1 = usize::from_str(sp.next().ok_or("Failed to get v1")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v2 = usize::from_str(sp.next().ok_or("Failed to get v2")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v3 = usize::from_str(sp.next().ok_or("Failed to get v3")?)
        .or_else(|e| return Err(e.to_string()))?;
    Ok((v1, v2, v3))
}
