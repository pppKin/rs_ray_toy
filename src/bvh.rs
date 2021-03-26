use crate::{
    geometry::{Bounds3f, IntersectP, Point3f, Ray, Vector3f},
    interaction::SurfaceInteraction,
    primitives::Primitive,
};

use std::{
    rc::Rc,
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};
// // BVHAccel Utility Functions
fn left_shift3(x: u32) -> u32 {
    let mut x = x; //make a local mut copy
    assert!(x <= (1 << 10));
    if x == (1 << 10) {
        x -= 1;
    }
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

fn encode_mortoncode3(v: &Vector3f) -> u32 {
    assert!(v.x >= 0.0);
    assert!(v.y >= 0.0);
    assert!(v.z >= 0.0);
    (left_shift3(v.z as u32) << 2) | (left_shift3(v.y as u32) << 1) | left_shift3(v.x as u32)
}

#[derive(Clone, Debug, Default)]
struct BucketInfo {
    count: u32,
    bounds: Bounds3f,
}

#[derive(Clone, Debug, Default)]
struct BVHBuildNode {
    bounds: Bounds3f,
    children: [Option<Rc<BVHBuildNode>>; 2],
    split_axis: u32,
    first_prime_offset: u32,
    n_primitives: u32,
}

impl BVHBuildNode {
    fn init_leaf(&mut self, first: u32, n: u32, b: &Bounds3f) {
        self.first_prime_offset = first;
        self.n_primitives = n;
        self.bounds = *b;
        self.children = [None, None]; // initialize only when necessarry
    }
    fn init_interior(&mut self, axis: u32, c0: BVHBuildNode, c1: BVHBuildNode) {
        self.bounds = Bounds3f::union_bnd(&c0.bounds, &c1.bounds);
        self.children = [Some(Rc::new(c0)), Some(Rc::new(c1))];
        self.split_axis = axis;
        self.n_primitives = 0;
    }
}

struct BVHPrimitiveInfo {
    primitive_number: u32,
    bounds: Bounds3f,
    centroid: Point3f,
}

// For example, consider a 2D coordinate(x,y)where the bits of x and y are denoted by xi and yi. The corresponding Morton-coded value is
// ...y3x3y2x2y1x1y0x0
// We use 10 bits for each of thex, y, and z dimensions, giving a total of 30 bits forthe Morton code.
#[derive(Default, Clone, Copy)]
struct MortonPrimitive {
    primitive_index: u32,
    morton_code: u32,
}

struct LBVHTreelet {
    start_index: u32,
    n_primitives: u32,
    build_nodes: BVHBuildNode,
}

impl LBVHTreelet {
    fn new(start_index: u32, n_primitives: u32, build_nodes: BVHBuildNode) -> Self {
        LBVHTreelet {
            start_index,
            n_primitives,
            build_nodes,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct LinearBVHNode {
    bounds: Bounds3f,
    offset: u32,
    n_primitives: u32,
    axis: u32,
}

pub enum BVHSplitMethod {
    SAH,
    HLBVH,
}

pub struct BVHAccel {
    max_prims_in_node: u32,
    split_method: BVHSplitMethod,
    primitives: Vec<Arc<dyn Primitive>>,
    nodes: Vec<LinearBVHNode>,
}

impl IntersectP for BVHAccel {
    fn intersect_p(&self, r: &Ray) -> bool {
        if !self.nodes.is_empty() {
            let inv_dir = Vector3f::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
            let dir_is_neg: [u8; 3] = [
                (inv_dir.x < 0.0) as u8,
                (inv_dir.y < 0.0) as u8,
                (inv_dir.z < 0.0) as u8,
            ];
            // int nodesToVisit[64];
            let mut nodes_to_visit: [usize; 64] = [0; 64];
            let mut to_visit_offset = 0;
            let mut current_node_index = 0;
            loop {
                let node = &self.nodes[current_node_index];
                if node.bounds.intersect_p(r, &inv_dir, &dir_is_neg) {
                    // Process BVH node _node_ for traversal
                    if node.n_primitives > 0 {
                        for i in 0..node.n_primitives {
                            if self.primitives[(node.offset + i) as usize].intersect_p(r) {
                                return true;
                            }
                        }
                        if to_visit_offset == 0 {
                            break;
                        }
                        to_visit_offset -= 1;
                        current_node_index = nodes_to_visit[to_visit_offset as usize];
                    } else {
                        if dir_is_neg[node.axis as usize] > 0 {
                            // second child first
                            nodes_to_visit[to_visit_offset] = current_node_index + 1;
                            to_visit_offset += 1;
                            current_node_index = node.offset as usize;
                        } else {
                            nodes_to_visit[to_visit_offset] = node.offset as usize;
                            to_visit_offset += 1;
                            current_node_index += 1;
                        }
                    }
                } else {
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                }
            }
        }
        false
    }
}

impl Primitive for BVHAccel {
    fn world_bound(&self) -> Bounds3f {
        if !self.nodes.is_empty() {
            return self.nodes[0].bounds;
        }
        Bounds3f::default()
    }
    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let mut hit = false;
        if !self.nodes.is_empty() {
            let inv_dir = Vector3f::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
            let dir_is_neg: [u8; 3] = [
                (inv_dir.x < 0.0) as u8,
                (inv_dir.y < 0.0) as u8,
                (inv_dir.z < 0.0) as u8,
            ];
            // int nodesToVisit[64];
            let mut nodes_to_visit: [usize; 64] = [0; 64];
            let mut to_visit_offset = 0;
            let mut current_node_index = 0;
            loop {
                let node = &self.nodes[current_node_index];
                if node.bounds.intersect_p(r, &inv_dir, &dir_is_neg) {
                    // Process BVH node _node_ for traversal
                    if node.n_primitives > 0 {
                        for i in 0..node.n_primitives {
                            if self.primitives[(node.offset + i) as usize].intersect(r, si) {
                                hit = true;
                            }
                        }
                        if to_visit_offset == 0 {
                            break;
                        }
                        to_visit_offset -= 1;
                        current_node_index = nodes_to_visit[to_visit_offset as usize];
                    } else {
                        if dir_is_neg[node.axis as usize] > 0 {
                            // second child first
                            nodes_to_visit[to_visit_offset] = current_node_index + 1;
                            to_visit_offset += 1;
                            current_node_index = node.offset as usize;
                        } else {
                            nodes_to_visit[to_visit_offset] = node.offset as usize;
                            to_visit_offset += 1;
                            current_node_index += 1;
                        }
                    }
                } else {
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                }
            }
        }
        hit
    }
}

// Recall that a radix sort differs from most sorting algorithms
// in that it isn’t based oncomparing pairs of values but rather is based on bucketing items
// based on some key.
// Radix sort can be used to sort integer values by sorting them one digit at a time,
// going from the right most digit to the leftmost.
// Especially with binary values, it’s worth sorting multiple digits at a time;
// doing so reduces the total number of passes taken over the data.In the implementation here,
// bitsPerPasssets the number of bits processed per pass; withthe value 6, we have 5 passes to sort the 30 bits
fn radix_sort(v: &mut Vec<MortonPrimitive>) {
    let mut tmp_vector = Vec::<MortonPrimitive>::with_capacity(v.len());
    for _ in 0..v.len() {
        tmp_vector.push(MortonPrimitive::default());
    }
    let bits_per_pass = 6;
    let n_bits = 30;
    // "Radix sort bitsPerPass must evenly divide nBits");
    assert_eq!(n_bits % bits_per_pass, 0);
    let n_passes = n_bits / bits_per_pass;
    for pass in 0..n_passes {
        // Perform one pass of radix sort, sorting _bitsPerPass_ bits
        let low_bit = pass * bits_per_pass;

        // Set in and out vector pointers for radix sort pass
        let in_v: &mut Vec<MortonPrimitive>;
        let out_v: &mut Vec<MortonPrimitive>;
        if pass & 1 > 0 {
            in_v = &mut tmp_vector;
            out_v = v;
        } else {
            in_v = v;
            out_v = &mut tmp_vector;
        }
        // Count number of zero bits in array for current radix sort bit
        let n_buckets = 1 << bits_per_pass;
        let mut bucket_count = vec![0; n_buckets];
        let bit_mask = (1 << bits_per_pass) - 1;
        for mp_idx in 0..in_v.len() {
            let mp = in_v[mp_idx]; //Copied here?
            let bucket = (mp.morton_code >> low_bit) & bit_mask;
            // assert!(bucket >= 0);
            assert!(bucket < n_buckets as u32);
            bucket_count[bucket as usize] += 1;
        }

        // Compute starting index in output array for each bucket
        let mut out_index = vec![0; n_buckets];
        out_index[0] = 0;
        for i in 1..n_buckets {
            out_index[i] = out_index[i - 1] + bucket_count[i - 1];
        }

        // Store sorted values in output array
        for mp_idx in 0..in_v.len() {
            let mp = in_v[mp_idx]; //Copied here?
            let bucket = (mp.morton_code >> low_bit) & bit_mask;
            let out_idx = out_index[bucket as usize] as usize;
            (*out_v)[out_idx] = mp;
            out_index[bucket as usize] += 1;
        }
    }

    // Copy final result from _tempVector_, if needed
    if (n_passes & 1) > 0 {
        std::mem::swap(v, &mut tmp_vector);
    }
}

impl BVHAccel {
    pub fn new(
        p: Vec<Arc<dyn Primitive>>,
        max_prims_in_node: u32,
        split_method: BVHSplitMethod,
    ) -> Self {
        let mut bvhaccel = BVHAccel {
            max_prims_in_node,
            split_method,
            primitives: p,
            nodes: vec![],
        };
        // Build BVH from _primitives_
        assert!(bvhaccel.primitives.len() > 0);

        // Initialize _primitiveInfo_ array for primitives
        let mut primitive_info_v =
            Vec::<BVHPrimitiveInfo>::with_capacity(bvhaccel.primitives.len());
        for i in 0..bvhaccel.primitives.len() {
            let b = bvhaccel.primitives[i].world_bound();
            primitive_info_v.push(BVHPrimitiveInfo {
                primitive_number: i as u32,
                bounds: b,
                centroid: (b.p_min + b.p_max) * 0.5,
            })
        }
        let primitive_info = Arc::new(primitive_info_v);

        // Build BVH tree for primitives using _primitiveInfo_
        let mut total_nodes = 0;
        let mut ordered_prims: Vec<Option<Arc<dyn Primitive>>> =
            vec![None; bvhaccel.primitives.len()];
        let root;
        match bvhaccel.split_method {
            BVHSplitMethod::HLBVH => {
                root = bvhaccel.hlbvh_build(
                    Arc::clone(&primitive_info),
                    &mut total_nodes,
                    &mut ordered_prims,
                );
            }
            BVHSplitMethod::SAH => unimplemented!(),
        }
        bvhaccel.primitives = vec![];
        for p_opt in ordered_prims {
            match p_opt {
                Some(pr) => {
                    bvhaccel.primitives.push(pr);
                }
                None => {}
            }
        }
        bvhaccel.nodes = vec![LinearBVHNode::default(); total_nodes as usize];
        let mut offset = 0;
        bvhaccel.flattern_bvh(&root, &mut offset);
        assert_eq!(total_nodes, offset);
        bvhaccel
    }

    fn hlbvh_build(
        &self,
        primitive_info: Arc<Vec<BVHPrimitiveInfo>>,
        total_nodes: &mut u32,
        ordered_prims: &mut Vec<Option<Arc<dyn Primitive>>>,
    ) -> BVHBuildNode {
        // Compute bounding box of all primitive centroids
        //     Bounds3f bounds;
        let mut bounds = Bounds3f::default();
        for pi in &*primitive_info {
            bounds = Bounds3f::union(&bounds, &pi.centroid);
        }

        let morton_prims = Arc::new(Mutex::new(vec![
            MortonPrimitive::default();
            primitive_info.len()
        ]));
        let (done_tx, done_rx): (
            Sender<(u32, MortonPrimitive)>,
            Receiver<(u32, MortonPrimitive)>,
        ) = mpsc::channel();
        let mut encode_morton_hns = vec![];
        let primitive_info_len = primitive_info.len();
        let portion = (primitive_info_len / 4) as u32;
        // Compute Morton indices of primitives
        for worker_count in 0..4 {
            let d_tx = done_tx.clone();
            let worker_idx = worker_count;
            let start = (worker_idx * portion) as usize;
            let mut end = ((worker_idx + 1) * portion) as usize;
            if worker_idx == 3 {
                end = primitive_info_len;
            }
            let pis = Arc::clone(&primitive_info);
            encode_morton_hns.push(thread::spawn(move || {
                for i in start..end {
                    let pi = &pis[i]; //borrow it here
                    let mut mp = MortonPrimitive::default();
                    let morton_bits = 10;
                    let morton_scale = (1 << morton_bits) as f64;
                    mp.primitive_index = pi.primitive_number;
                    let centroid_offset = bounds.offset(&pi.centroid);
                    mp.morton_code = encode_mortoncode3(&(centroid_offset * morton_scale));
                    d_tx.send((i as u32, mp)).unwrap();
                }
            }));
        }

        drop(done_tx);
        let e_morton_prims = Arc::clone(&morton_prims);
        encode_morton_hns.push(thread::spawn(move || loop {
            match done_rx.recv() {
                Ok((idx, mp)) => {
                    let mut mps = e_morton_prims.lock().unwrap();
                    mps[idx as usize] = mp;
                }
                Err(_e) => {
                    break;
                }
            }
        }));

        for hn in encode_morton_hns {
            hn.join().unwrap();
        }
        // After all morton_prims have been calculated and stored, return back to this thread

        // Radix sort primitive Morton indices
        let mut mps = morton_prims.lock().unwrap();
        radix_sort(&mut mps);

        // Create LBVH treelets at bottom of BVH

        // Find intervals of primitives for each treelet
        let mut treelets_to_build: Vec<LBVHTreelet> = vec![];
        let mut start = 0;
        // 0 based indexing
        for end in 1..(mps.len() + 1) {
            let mask = 0b00111111111111000000000000000000;
            if end == mps.len()
                || ((mps[start].morton_code & mask) != (mps[end].morton_code & mask))
            {
                // Add entry to _treeletsToBuild_ for this treelet
                let n_primitives = end - start;
                let _max_bvh_nodes = 2 * n_primitives;
                let mut nodes: BVHBuildNode = BVHBuildNode::default();
                nodes.n_primitives = n_primitives as u32;
                treelets_to_build.push(LBVHTreelet::new(start as u32, n_primitives as u32, nodes));
                start = end;
            }
        }

        // Create LBVHs for treelets in parallel
        let (total_tx, total_rx): (Sender<u32>, Receiver<u32>) = mpsc::channel();
        let mut total: u32 = 0;

        let ordered_prim_offset = Arc::new(AtomicU32::new(0));
        ordered_prims.resize(self.primitives.len(), None);
        // TODO: Use ParallelFor
        for i in 0..treelets_to_build.len() {
            let total_tx_c = total_tx.clone();
            let mut nodes_created = 0;
            let first_bit_index = 29 - 12;
            let tr = &mut treelets_to_build[i];
            tr.build_nodes = self.emit_lbvh(
                &tr.build_nodes,
                Arc::clone(&primitive_info),
                &mps[tr.start_index as usize..(tr.start_index + tr.n_primitives) as usize],
                tr.n_primitives,
                &mut nodes_created,
                ordered_prims,
                Arc::clone(&ordered_prim_offset),
                first_bit_index,
            );

            total_tx_c.send(nodes_created).unwrap();
        }
        drop(total_tx);
        loop {
            match total_rx.recv() {
                Ok(t) => {
                    total += t;
                }
                Err(_e) => {
                    break;
                }
            }
        }
        *total_nodes = total;

        // Create and return SAH BVH from LBVH treelets
        let mut finished_treelets = Vec::<BVHBuildNode>::with_capacity(treelets_to_build.len());
        for treelet in treelets_to_build {
            finished_treelets.push(treelet.build_nodes);
        }
        let finished_treelets_len = finished_treelets.len() as u32;
        return self.build_upper_sah(
            &mut finished_treelets,
            0,
            finished_treelets_len,
            total_nodes,
        );
    }

    fn emit_lbvh(
        &self,
        build_nodes: &BVHBuildNode,
        primitive_info: Arc<Vec<BVHPrimitiveInfo>>,
        morton_prims: &[MortonPrimitive],
        n_primitives: u32,
        total_nodes: &mut u32,
        ordered_prims: &mut Vec<Option<Arc<dyn Primitive>>>,
        ordered_prim_offset: Arc<AtomicU32>,
        bit_index: i64,
    ) -> BVHBuildNode {
        assert!(n_primitives > 0);
        if bit_index == -1 || n_primitives < self.max_prims_in_node {
            // Create and return leaf node of LBVH treelet
            *total_nodes += 1;
            let mut node = BVHBuildNode::default();
            let mut bounds = Bounds3f::default();
            let first_prime_offset = ordered_prim_offset.fetch_add(n_primitives, Ordering::SeqCst);
            for i in 0..n_primitives {
                let primitive_index = morton_prims[i as usize].primitive_index;
                let p = &self.primitives[primitive_index as usize];
                let tmp_arc_p = Arc::clone(&p);
                ordered_prims[(first_prime_offset + i) as usize] = Some(tmp_arc_p);
                bounds = Bounds3f::union_bnd(&bounds, &p.world_bound());
            }
            node.init_leaf(first_prime_offset, n_primitives, &bounds);
            return node;
        } else {
            // Advance to next subtree level if there's no LBVH split for this bit
            let mask = 1 << bit_index;
            if (morton_prims[0].morton_code & mask)
                == (morton_prims[(n_primitives - 1) as usize].morton_code & mask)
            {
                return self.emit_lbvh(
                    build_nodes,
                    primitive_info,
                    morton_prims,
                    n_primitives,
                    total_nodes,
                    ordered_prims,
                    ordered_prim_offset,
                    bit_index - 1,
                );
            } else {
            }

            // Find LBVH split point for this dimension
            let mut search_start = 0;
            let mut search_end = n_primitives - 1;
            // a binary search efficiently finds the dividing point
            // where the bitIndexth bit goes from 0 to 1 in the current set of primitives.
            while search_start + 1 != search_end {
                assert!(search_start != search_end);
                let mid = (search_start + search_end) / 2;
                if (morton_prims[search_start as usize].morton_code & mask)
                    == (morton_prims[mid as usize].morton_code & mask)
                {
                    search_start = mid;
                } else {
                    search_end = mid;
                }
            }
            let split_offset = search_end;
            assert!(split_offset <= (n_primitives - 1));
            assert!(
                (morton_prims[(split_offset - 1) as usize].morton_code & mask)
                    != (morton_prims[split_offset as usize].morton_code & mask)
            );

            // Create and return interior LBVH node
            *total_nodes += 1;
            let mut node = BVHBuildNode::default();
            let c0 = self.emit_lbvh(
                build_nodes,
                Arc::clone(&primitive_info),
                morton_prims,
                split_offset,
                total_nodes,
                ordered_prims,
                Arc::clone(&ordered_prim_offset),
                bit_index - 1,
            );
            let c1 = self.emit_lbvh(
                build_nodes,
                primitive_info,
                morton_prims,
                n_primitives - split_offset,
                total_nodes,
                ordered_prims,
                ordered_prim_offset,
                bit_index - 1,
            );
            let axis = bit_index % 3;
            node.init_interior(axis as u32, c0, c1);
            return node;
        }
    }

    fn build_upper_sah(
        &self,
        treelet_roots: &mut Vec<BVHBuildNode>,
        start: u32,
        end: u32,
        total_nodes: &mut u32,
    ) -> BVHBuildNode {
        // CHECK_LT(start, end);
        assert!(start < end);
        let n_nodes = end - start;
        // int nNodes = end - start;
        if n_nodes == 1 {
            return treelet_roots[start as usize].clone();
        }
        *total_nodes += 1;
        let mut node = BVHBuildNode::default();
        // Compute bounds of all nodes under this HLBVH node
        let mut bounds = Bounds3f::default();
        for i in start..end {
            bounds = Bounds3f::union_bnd(&bounds, &treelet_roots[i as usize].bounds);
        }

        // Compute bound of HLBVH node centroids, choose split dimension _dim_
        let mut centroid_bounds = Bounds3f::default();
        for i in start..end {
            let t = &treelet_roots[i as usize];
            let centroid = (t.bounds.p_min + t.bounds.p_max) * 0.5;
            centroid_bounds = Bounds3f::union(&centroid_bounds, &centroid);
        }

        let dim = centroid_bounds.maximum_extent();
        // FIXME: if this hits, what do we need to do?
        // Make sure the SAH split below does something... ?
        assert!(centroid_bounds.p_max[dim] != centroid_bounds.p_min[dim]);
        // Allocate _BucketInfo_ for SAH partition buckets
        let n_buckets = 12;

        let mut buckets = vec![BucketInfo::default(); n_buckets];
        // Initialize _BucketInfo_ for HLBVH SAH partition buckets
        for i in start..end {
            let t = &treelet_roots[i as usize];
            let centroid = (t.bounds.p_min[dim] + t.bounds.p_max[dim]) * 0.5;
            let mut b = ((n_buckets as f64)
                * ((centroid - centroid_bounds.p_min[dim])
                    / (centroid_bounds.p_max[dim] - centroid_bounds.p_min[dim])))
                as usize;

            if b == n_buckets {
                b = n_buckets - 1;
            }
            assert!(b < n_buckets);
            buckets[b].count += 1;
            buckets[b].bounds = Bounds3f::union_bnd(&buckets[b].bounds, &t.bounds);
        }

        // Compute costs for splitting after each bucket
        // Float cost[nBuckets - 1];
        let mut costs = vec![0_f64; n_buckets - 1];
        for i in 0..n_buckets - 1 {
            let mut b0 = Bounds3f::default();
            let mut b1 = Bounds3f::default();
            let mut count0 = 0;
            let mut count1 = 0;
            for j in 0..i {
                b0 = Bounds3f::union_bnd(&b0, &buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for j in i + 1..n_buckets {
                b1 = Bounds3f::union_bnd(&b1, &buckets[j].bounds);
                count1 += buckets[j].count;
            }
            costs[i] = 0.125
                + (count0 as f64 * b0.surface_area() + count1 as f64 * b1.surface_area())
                    / bounds.surface_area();
        }

        // Find bucket to split at that minimizes SAH metric
        let mut min_cost = costs[0];
        let mut min_cost_bucket = 0;
        for i in 1..n_buckets - 1 {
            if costs[i] < min_cost {
                min_cost = costs[i];
                min_cost_bucket = i;
            }
        }

        // Split nodes and create interior HLBVH SAH node
        let mid = treelet_roots[start as usize..end as usize]
            .iter_mut()
            .partition_in_place(|node| {
                let centroid = (node.bounds.p_min[dim] + node.bounds.p_max[dim]) * 0.5;
                let mut b = (n_buckets as f64
                    * ((centroid - centroid_bounds.p_min[dim])
                        / (centroid_bounds.p_max[dim] - centroid_bounds.p_min[dim])))
                    as usize;
                if b == n_buckets {
                    b = n_buckets - 1;
                }
                assert!(b < n_buckets);
                b <= min_cost_bucket
            }) as u32
            + start;
        assert!(mid > start);
        assert!(mid < end);

        node.init_interior(
            dim as u32,
            self.build_upper_sah(treelet_roots, start, mid, total_nodes),
            self.build_upper_sah(treelet_roots, mid, end, total_nodes),
        );

        node
    }

    fn flattern_bvh(&mut self, node: &BVHBuildNode, offset: &mut u32) -> u32 {
        let my_offset = *offset;
        (*offset) += 1;
        self.nodes[my_offset as usize].bounds = node.bounds;
        if node.n_primitives > 0 {
            assert!(node.children[0].is_none());
            assert!(node.children[1].is_none());
            assert!(node.n_primitives < (2 << 15));
            self.nodes[my_offset as usize].offset = node.first_prime_offset;
            self.nodes[my_offset as usize].n_primitives = node.n_primitives;
        } else {
            // Create interior flattened BVH node
            self.nodes[my_offset as usize].axis = node.split_axis;
            self.nodes[my_offset as usize].n_primitives = 0;

            if node.children[0].is_some() && node.children[1].is_some() {
                let c0 = Rc::clone(&(node.children[0].as_ref().unwrap()));
                let c1 = Rc::clone(&(node.children[1].as_ref().unwrap()));
                self.flattern_bvh(&*c0, offset);
                self.nodes[my_offset as usize].offset = self.flattern_bvh(&*c1, offset);
            }
        }
        my_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::pnt3_lerp;
    // use crate::lights::*;
    // use crate::material::*;
    // use crate::primitives::*;
    // use crate::shape::triangle::*;
    // use crate::transform::*;
    use rand::prelude::*;
    #[test]
    fn test_morton_sort() {
        let mut rng = rand::thread_rng();

        let mut pi_vec = Vec::<BVHPrimitiveInfo>::with_capacity(50);
        for idx in 0..10 {
            let p1 = Point3f::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-20.0..5.0),
                rng.gen_range(6.0..50.0),
            );
            let p2 = Point3f::new(
                rng.gen_range(11.0..54.0),
                rng.gen_range(6.0..33.0),
                rng.gen_range(51.0..88.0),
            );
            let bd = Bounds3f::new(p1, p2);
            let pi = BVHPrimitiveInfo {
                primitive_number: idx,
                bounds: bd,
                centroid: pnt3_lerp(rng.gen(), &p1, &p2),
            };
            pi_vec.push(pi);
        }
        let mut bnds = Bounds3f::default();
        for pi in &pi_vec {
            bnds = Bounds3f::union_bnd(&bnds, &pi.bounds);
        }
        let mut mp_v = vec![];
        for pi in &pi_vec {
            let centroid_offset = bnds.offset(&pi.centroid) * ((1 << 10) as f64);
            let mc = encode_mortoncode3(&centroid_offset);
            mp_v.push(MortonPrimitive {
                primitive_index: pi.primitive_number,
                morton_code: mc,
            })
        }

        radix_sort(&mut mp_v);
    }

    // #[test]
    // fn test_tri_bvh() {
    //     let mut rng = rand::thread_rng();

    //     let obj_to_world = Transform::default();
    //     let world_to_obj = Transform::default();
    //     let mut vertices = vec![];
    //     for _i in 0..134 {
    //         vertices.push(Point3f::new(
    //             rng.gen_range(-100.0, 100.0),
    //             rng.gen_range(-100.0, 100.0),
    //             rng.gen_range(-100.0, 100.0),
    //         ));
    //     }
    //     let mesh = create_triangle_mesh(
    //         obj_to_world,
    //         world_to_obj,
    //         120,
    //         132,
    //         vec![
    //             0, 1, 2, 1, 0, 3, 4, 2, 1, 2, 4, 5, 6, 5, 4, 5, 6, 7, 8, 7, 6, 7, 8, 9, 10, 9, 8,
    //             9, 10, 11, 12, 11, 10, 11, 12, 13, 14, 13, 12, 13, 14, 15, 16, 15, 14, 15, 16, 17,
    //             18, 17, 16, 17, 18, 19, 20, 19, 18, 19, 20, 21, 22, 23, 24, 23, 22, 25, 26, 24, 23,
    //             24, 26, 27, 28, 27, 26, 27, 28, 29, 30, 29, 28, 29, 30, 31, 32, 31, 30, 31, 32, 33,
    //             34, 33, 32, 33, 34, 35, 36, 35, 34, 35, 36, 37, 38, 37, 36, 37, 38, 39, 40, 39, 38,
    //             39, 40, 41, 42, 41, 40, 41, 42, 43, 44, 45, 46, 45, 44, 47, 48, 46, 45, 46, 48, 49,
    //             50, 49, 48, 49, 50, 51, 52, 51, 50, 51, 52, 53, 54, 53, 52, 53, 54, 55, 56, 55, 54,
    //             55, 56, 57, 58, 57, 56, 57, 58, 59, 60, 59, 58, 59, 60, 61, 62, 61, 60, 61, 62, 63,
    //             64, 63, 62, 63, 64, 65, 66, 67, 68, 67, 66, 69, 70, 68, 67, 68, 70, 71, 72, 71, 70,
    //             71, 72, 73, 74, 73, 72, 73, 74, 75, 76, 75, 74, 75, 76, 77, 78, 77, 76, 77, 78, 79,
    //             80, 79, 78, 79, 80, 81, 82, 81, 80, 81, 82, 83, 84, 83, 82, 83, 84, 85, 86, 85, 84,
    //             85, 86, 87, 88, 89, 90, 89, 88, 91, 92, 90, 89, 90, 92, 93, 94, 93, 92, 93, 94, 95,
    //             96, 95, 94, 95, 96, 97, 98, 97, 96, 97, 98, 99, 100, 99, 98, 99, 100, 101, 102,
    //             101, 100, 101, 102, 103, 104, 103, 102, 103, 104, 105, 106, 105, 104, 105, 106,
    //             107, 108, 107, 106, 107, 108, 109, 110, 111, 112, 111, 110, 113, 114, 112, 111,
    //             112, 114, 115, 116, 115, 114, 115, 116, 117, 118, 117, 116, 117, 118, 119, 120,
    //             119, 118, 119, 120, 121, 122, 121, 120, 121, 122, 123, 124, 123, 122, 123, 124,
    //             125, 126, 125, 124, 125, 126, 127, 128, 127, 126, 127, 128, 129, 130, 129, 128,
    //             129, 130, 131,
    //         ],
    //         vertices,
    //         vec![],
    //         vec![],
    //         vec![],
    //     );

    //     let mut pris = vec![];
    //     for tri_a in mesh {
    //         let tri = Arc::clone(&tri_a);
    //         pris.push(Arc::new(GeometricPrimitive::new(
    //             tri,
    //             Arc::new(Material::default()),
    //             Arc::new(Light::default()),
    //         )) as Arc<dyn Primitive>);
    //     }
    //     let bvhaccel = BVHAccel::new(pris, 8, BVHSplitMethod::HLBVH);
    //     assert!(bvhaccel.intersect_p(&Ray::new_od(
    //         Point3f::new(
    //             rng.gen_range(-10.0, 10.0),
    //             rng.gen_range(-10.0, 10.0),
    //             rng.gen_range(-10.0, 10.0),
    //         ),
    //         Vector3f::new(
    //             rng.gen_range(-1.0, 1.0),
    //             rng.gen_range(-1.0, 1.0),
    //             rng.gen_range(-1.0, 1.0),
    //         )
    //     )));
    //     println!(
    //         "bvhaccel.nodes.len() {} {}",
    //         bvhaccel.nodes.len(),
    //         &bvhaccel.nodes[0].n_primitives
    //     );
    // }
}
