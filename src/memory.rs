use std::ops::{Index, IndexMut, Not};

// the BlockedArray template implements a generic 2D array of values, with the items ordered in memory using a blocked memory layout,
// The array is subdivided into square blocks of a small fixed size that is a power of 2.
//  Each block is laid out row by row
// This organization substantially improves the memory coherence of 2D array referencesin practice
// and requires only a small amount of additional computation to determine the memory address for a particular position (Lam, Rothberg, and Wolf 1991).
// TODO: benchmark this?

pub const LOG_BLOCK_SIZE: usize = 2;
pub const BLOCK_SIZE: usize = 1 << LOG_BLOCK_SIZE;

pub fn ba_round_up(x: usize) -> usize {
    (x + BLOCK_SIZE - 1) & (!(BLOCK_SIZE - 1))
}
pub fn ba_offset(a: usize) -> usize {
    a >> LOG_BLOCK_SIZE
}
pub fn ba_block(a: usize) -> usize {
    a & (BLOCK_SIZE - 1)
}

#[derive(Debug, Default, Clone)]
pub struct BlockedArray<T: Clone + Copy + From<f64>> {
    data: Vec<T>,
    u_res: usize,
    v_res: usize,
    u_blocks: usize,
}

impl<T: Clone + From<f64> + Copy> BlockedArray<T> {
    // let mut ba = Self::new(u_res, v_res);
    // for u in 0..u_res {
    //     for v in 0..v_res {
    //         ba[(u, v)] = d[v * u_res + u].clone();
    //     }
    // }
    // ba
    pub fn new(data: Option<Vec<T>>, u_res: usize, v_res: usize) -> Self {
        let u_blocks = ba_round_up(u_res) >> LOG_BLOCK_SIZE;
        let tmp_v = vec![T::from(0.0); ba_round_up(u_res) * ba_round_up(v_res)];
        let mut ba = BlockedArray {
            data: tmp_v,
            u_res,
            v_res,
            u_blocks,
        };
        if let Some(d) = data {
            for u in 0..u_res {
                for v in 0..v_res {
                    ba[(u, v)] = d[v * u_res + u]
                }
            }
        }
        ba
    }
    pub fn u_size(&self) -> usize {
        self.u_res
    }
    pub fn v_size(&self) -> usize {
        self.v_res
    }
}

impl<T> Not for BlockedArray<T>
where
    T: Not<Output = T> + Copy + From<f64>,
{
    type Output = BlockedArray<T>;

    fn not(self) -> Self::Output {
        let mut v = vec![];
        for value in &self.data {
            v.push(!*value);
        }
        Self::Output::new(Some(v), self.u_size(), self.v_size())
    }
}

impl<T: Clone + From<f64> + Copy> Index<(usize, usize)> for BlockedArray<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (u, v) = index;
        let bu = ba_block(u);
        let bv = ba_block(v);
        let ou = ba_offset(u);
        let ov = ba_offset(v);
        let mut offset = BLOCK_SIZE * BLOCK_SIZE * (self.u_blocks * bv + bu);
        offset += BLOCK_SIZE * ov + ou;
        &self.data[offset]
    }
}

impl<T: Clone + From<f64> + Copy> IndexMut<(usize, usize)> for BlockedArray<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (u, v) = index;
        let bu = ba_block(u);
        let bv = ba_block(v);
        let ou = ba_offset(u);
        let ov = ba_offset(v);
        let mut offset = BLOCK_SIZE * BLOCK_SIZE * (self.u_blocks * bv + bu);
        offset += BLOCK_SIZE * ov + ou;
        &mut self.data[offset]
    }
}
