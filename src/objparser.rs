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
    pub normal_indices: Vec<usize>,
    pub uv_indices: Vec<usize>,
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub s: Vec<Vector3f>,
    pub uv: Vec<Point2f>,
}

impl ParseResult {
    pub fn new(
        vertex_indices: Vec<usize>,
        normal_indices: Vec<usize>,
        uv_indices: Vec<usize>,
        p: Vec<Point3f>,
        n: Vec<Normal3f>,
        s: Vec<Vector3f>,
        uv: Vec<Point2f>,
    ) -> Self {
        let n_triangles = vertex_indices.len() / 3;
        let n_vertices = p.len();
        if uv_indices.len() > 0 {
            assert!(vertex_indices.len() == uv_indices.len())
        }
        if normal_indices.len() > 0 {
            assert!(vertex_indices.len() == normal_indices.len())
        }
        Self {
            n_triangles,
            n_vertices,
            vertex_indices,
            normal_indices,
            uv_indices,
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

    let s = vec![];

    let mut n = vec![];
    let mut uv = vec![];

    let mut normal_indices = vec![];
    let mut uv_indices = vec![];
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
                    Ok((f1, f2, f3, f4_opt)) => {
                        if let (Some(v1), Some(v2), Some(v3)) = (f1.0, f2.0, f3.0) {
                            vertex_indices.push(v1);
                            vertex_indices.push(v2);
                            vertex_indices.push(v3);
                            if let (Some(vt1), Some(vt2), Some(vt3)) = (f1.1, f2.1, f3.1) {
                                if !uv.is_empty()
                                    && vt1 < uv.len()
                                    && vt2 < uv.len()
                                    && vt3 < uv.len()
                                {
                                    uv_indices.push(vt1);
                                    uv_indices.push(vt2);
                                    uv_indices.push(vt3);
                                }
                            }

                            if let (Some(vn1), Some(vn2), Some(vn3)) = (f1.2, f2.2, f3.2) {
                                if !n.is_empty() && vn1 < n.len() && vn2 < n.len() && vn3 < n.len()
                                {
                                    normal_indices.push(vn1);
                                    normal_indices.push(vn2);
                                    normal_indices.push(vn3);
                                }
                            }

                            if let Some(f4) = f4_opt {
                                if let Some(v4) = f4.0 {
                                    vertex_indices.push(v2);
                                    vertex_indices.push(v3);
                                    vertex_indices.push(v4);
                                }
                                if let (Some(vt2), Some(vt3), Some(vt4)) = (f2.1, f3.1, f4.1) {
                                    if !uv.is_empty()
                                        && vt2 < uv.len()
                                        && vt3 < uv.len()
                                        && vt4 < uv.len()
                                    {
                                        uv_indices.push(vt2);
                                        uv_indices.push(vt3);
                                        uv_indices.push(vt4);
                                    }
                                }
                                if let (Some(vn2), Some(vn3), Some(vn4)) = (f2.2, f3.2, f4.2) {
                                    if !n.is_empty()
                                        && vn2 < n.len()
                                        && vn3 < n.len()
                                        && vn4 < n.len()
                                    {
                                        normal_indices.push(vn2);
                                        normal_indices.push(vn3);
                                        normal_indices.push(vn4);
                                    }
                                }
                            }
                        }
                    }
                    Err(desc) => {
                        err.desc = desc;
                        return Err(Box::new(err));
                    }
                }
            }
            Some("#") => {
                // comment line
            }
            unknown => {
                eprintln!("ParseObjError: unsupported Element {:?}", unknown)
            }
        }
    }
    eprintln!(
        "Parse {} succeed! {} verteices, {} triangles!",
        filename,
        p.len(),
        vertex_indices.len() / 3
    );
    Ok(ParseResult::new(
        vertex_indices,
        normal_indices,
        uv_indices,
        p,
        n,
        s,
        uv,
    ))
}

fn make_vertex(sp: &mut SplitWhitespace) -> Result<Point3f, String> {
    let v1 = f64::from_str(sp.next().ok_or("ParseObjError: Failed to get v1")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v2 = f64::from_str(sp.next().ok_or("ParseObjError: Failed to get v2")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v3 = f64::from_str(sp.next().ok_or("ParseObjError: Failed to get v3")?)
        .or_else(|e| return Err(e.to_string()))?;
    Ok(Point3f::new(v1, v2, v3))
}

fn make_uv(sp: &mut SplitWhitespace) -> Result<Point2f, String> {
    let u = f64::from_str(sp.next().ok_or("ParseObjError: Failed to get v1")?)
        .or_else(|e| return Err(e.to_string()))?;
    let v = f64::from_str(sp.next().unwrap_or_default()).or_else(|e| return Err(e.to_string()))?;
    Ok(Point2f::new(u, v))
}

fn parse_face_element(f_str: &str) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut tmp = vec![];
    for idx in f_str.split("/") {
        if let Ok(i) = usize::from_str(idx) {
            tmp.push(Some(i - 1));
        } else {
            tmp.push(None)
        }
    }
    tmp.resize(3, None);
    (tmp[0], tmp[1], tmp[2])
}

fn make_face(
    sp: &mut SplitWhitespace,
) -> Result<
    (
        (Option<usize>, Option<usize>, Option<usize>),
        (Option<usize>, Option<usize>, Option<usize>),
        (Option<usize>, Option<usize>, Option<usize>),
        Option<(Option<usize>, Option<usize>, Option<usize>)>,
    ),
    String,
> {
    if let (Some(v1), Some(v2), Some(v3)) = (sp.next(), sp.next(), sp.next()) {
        let v1_element = parse_face_element(v1);
        let v2_element = parse_face_element(v2);
        let v3_element = parse_face_element(v3);
        let v4_element;
        if let Some(v4) = sp.next() {
            v4_element = Some(parse_face_element(v4));
        } else {
            v4_element = None;
        }
        return Ok((v1_element, v2_element, v3_element, v4_element));
    } else {
        return Err("ParseObjError: Failed to get face element".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::triangle::create_triangle_mesh;
    use crate::transform::Transform;
    #[test]
    fn test_parse_obj() {
        let r = parse_obj("../example.obj");
        match r {
            Ok(result) => {
                // println!("result : {} triangles", result.n_triangles);
                let tm = create_triangle_mesh(
                    Transform::default(),
                    Transform::default(),
                    result.n_triangles,
                    result.n_vertices,
                    result.vertex_indices,
                    result.normal_indices,
                    result.uv_indices,
                    result.p,
                    result.n,
                    result.s,
                    result.uv,
                );
                println!("result : {} triangles", tm.len());
            }
            Err(err) => {
                panic!("{}", err.to_string())
            }
        }
    }
}
