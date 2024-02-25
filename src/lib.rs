use memmap2::MmapMut;
use num_traits::Float;
use std::{
    fmt::Debug,
    fs::{File, OpenOptions},
    path::Path,
    vec,
};
type NodeID = usize;
type Swid = u128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    L2,
    IP,
}
fn get_distance<F: Float + Debug + Default>(a: &[F], b: &[F], space: Distance) -> F {
    #[cfg(target_feature = "avx")]
    const STRIDE: usize = 8;
    #[cfg(not(target_feature = "avx"))]
    const STRIDE: usize = 4;
    let chunks_a = a.chunks_exact(STRIDE);
    let chunks_b = b.chunks_exact(STRIDE);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    match space {
        Distance::Euclidean => {
            let mut sum = [F::zero(); STRIDE];
            for (a, b) in chunks_a.zip(chunks_b) {
                for i in 0..STRIDE {
                    let diff = a[i] - b[i];
                    sum[i] = diff.mul_add(diff, sum[i]);
                }
            }
            let mut sum = sum.iter().fold(F::zero(), |acc, &x| acc + x);
            for i in 0..rem_a.len() {
                let diff = rem_a[i] - rem_b[i];
                sum = diff.mul_add(diff, sum);
            }
            sum.sqrt()
        }
        Distance::Cosine => {
            let mut dot = [F::zero(); STRIDE];
            let mut xx = [F::zero(); STRIDE];
            let mut yy = [F::zero(); STRIDE];
            for (a, b) in chunks_a.zip(chunks_b) {
                for i in 0..STRIDE {
                    dot[i] = a[i].mul_add(b[i], dot[i]);
                    xx[i] = a[i].mul_add(a[i], xx[i]);
                    yy[i] = b[i].mul_add(b[i], yy[i]);
                }
            }
            let mut dot = dot.iter().fold(F::zero(), |acc, &x| acc + x);
            let mut xx = xx.iter().fold(F::zero(), |acc, &x| acc + x);
            let mut yy = yy.iter().fold(F::zero(), |acc, &x| acc + x);
            for i in 0..rem_a.len() {
                dot = rem_a[i].mul_add(rem_b[i], dot);
                xx = rem_a[i].mul_add(rem_a[i], xx);
                yy = rem_b[i].mul_add(rem_b[i], yy);
            }

            //handle 0 vectors
            if xx * yy <= F::zero() {
                return F::zero();
            }

            F::one() - dot / (xx * yy).sqrt()
        }
        Distance::L2 => {
            let mut sum = [F::zero(); STRIDE];
            for (a, b) in chunks_a.zip(chunks_b) {
                for i in 0..STRIDE {
                    let diff = a[i] - b[i];
                    sum[i] = diff.mul_add(diff, sum[i]);
                }
            }
            let mut sum = sum.iter().fold(F::zero(), |acc, &x| acc + x);
            for i in 0..rem_a.len() {
                let diff = rem_a[i] - rem_b[i];
                sum = diff.mul_add(diff, sum);
            }
            sum
        }
        Distance::IP => {
            let mut dot = [F::zero(); STRIDE];
            for (a, b) in chunks_a.zip(chunks_b) {
                for i in 0..STRIDE {
                    dot[i] = a[i].mul_add(b[i], dot[i]);
                }
            }
            let mut dot = dot.iter().fold(F::zero(), |acc, &x| acc + x);
            for i in 0..rem_a.len() {
                dot = rem_a[i].mul_add(rem_b[i], dot);
            }
            dot
        }
    }
}

#[derive(Debug)]
pub enum Store<T> {
    Mmap((File, MmapMut)),
    Vec(Vec<T>),
}
#[derive(Debug)]
pub struct VPTree<'a, F: Float + Debug + Default> {
    pub nodes: Vec<Node<F>>,
    pub dimensions: usize,
    pub vector_layer: &'a mut [F],
    pub swid_layer: &'a mut [Swid],
    pub vector_store: Store<F>,
    pub swid_store: Store<Swid>,
    space: Distance,
    top_node: Option<NodeID>,
}

#[derive(Debug)]
pub enum Node<F: Float + Debug + Default> {
    Branch1 {
        left_vector: NodeID,
        left_next: NodeID,
    },
    Branch2 {
        left_vector: NodeID,
        left_next: NodeID,
        right_vector: NodeID,
        right_next: NodeID,
        distance: F,
    },
    Leaf1 {
        left_vector: NodeID,
    },
    Leaf2 {
        left_vector: NodeID,
        right_vector: NodeID,
        distance: F,
    },
}
impl<F: Float + Debug + Default> Node<F> {
    fn vector(&self) -> NodeID {
        match self {
            Node::Branch1 { left_vector, .. }
            | Node::Branch2 { left_vector, .. }
            | Node::Leaf1 { left_vector, .. }
            | Node::Leaf2 { left_vector, .. } => *left_vector,
        }
    }
}

impl<'a, F: Float + Debug + Default> VPTree<'a, F> {
    pub fn new(space: Distance, dimensions: usize) -> VPTree<'a, F> {
        VPTree {
            nodes: Vec::new(),
            dimensions,
            vector_layer: &mut [],
            swid_layer: &mut [],
            vector_store: Store::Vec(Vec::new()),
            swid_store: Store::Vec(Vec::new()),
            space,
            top_node: None,
        }
    }
    pub fn new_with_store(
        space: Distance,
        dimensions: usize,
        vector_store: &Path,
        swid_store: &Path,
    ) -> VPTree<'a, F> {
        let vector_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(vector_store)
            .unwrap();
        let swid_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(swid_store)
            .unwrap();
        let mut vector_mmap = unsafe { MmapMut::map_mut(&vector_file).unwrap() };
        let mut swid_mmap = unsafe { MmapMut::map_mut(&swid_file).unwrap() };
        let vector_layer = unsafe {
            std::slice::from_raw_parts_mut(
                vector_mmap.as_mut_ptr() as *mut F,
                vector_mmap.len() / std::mem::size_of::<F>(),
            )
        };
        let swid_layer = unsafe {
            std::slice::from_raw_parts_mut(
                swid_mmap.as_mut_ptr() as *mut Swid,
                swid_mmap.len() / std::mem::size_of::<Swid>(),
            )
        };
        let mut tree: VPTree<'_, F> = VPTree {
            nodes: Vec::new(),
            dimensions,
            vector_layer: &mut [],
            swid_layer: &mut [],
            vector_store: Store::Vec(Vec::new()),
            swid_store: Store::Vec(Vec::new()),
            space,
            top_node: None,
        };
        swid_layer
            .iter()
            .zip(vector_layer.chunks(dimensions))
            .for_each(|(swid, vector)| {
                tree.insert(vector, *swid);
            });
        tree.vector_store = Store::Mmap((vector_file, vector_mmap));
        tree.swid_store = Store::Mmap((swid_file, swid_mmap));
        tree.vector_layer = vector_layer;
        tree.swid_layer = swid_layer;
        tree
    }
    fn get_vector(&self, vector_id: NodeID) -> &[F] {
        self.vector_layer
            .chunks(self.dimensions)
            .nth(vector_id)
            .unwrap()
    }
    fn get_closest_leaf(&self, q: &[F]) -> Option<Vec<NodeID>> {
        self.top_node?;
        let mut current_node = self.top_node.unwrap();
        let mut current_distance = get_distance(
            q,
            self.get_vector(self.nodes[current_node].vector()),
            self.space,
        );
        let mut parent_chain = vec![current_node];
        loop {
            (current_distance, current_node) = match self.nodes[current_node] {
                Node::Leaf1 { .. } | Node::Leaf2 { .. } => return Some(parent_chain),
                Node::Branch2 {
                    left_next,
                    right_vector,
                    right_next,
                    ..
                } => std::cmp::min_by(
                    (
                        get_distance(q, self.get_vector(right_vector), self.space),
                        right_next,
                    ),
                    (current_distance, left_next),
                    |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal),
                ),
                Node::Branch1 { left_next, .. } => (current_distance, left_next),
            };
            parent_chain.push(current_node);
        }
    }
    fn resize(&mut self, n: isize) {
        let new_len = if n > 0 {
            self.vector_layer.len() + n.unsigned_abs() * self.dimensions
        } else {
            self.vector_layer.len() - n.unsigned_abs() * self.dimensions
        };
        match self.vector_store {
            Store::Mmap((ref file, ref mut mmap)) => {
                mmap.flush().unwrap();
                let bytes = new_len * std::mem::size_of::<F>();
                file.set_len(bytes as u64).unwrap();

                *mmap = unsafe { MmapMut::map_mut(file).unwrap() };

                self.vector_layer =
                    unsafe { std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut F, new_len) };
            }
            Store::Vec(ref mut vec) => {
                vec.resize(new_len, F::zero());
                self.vector_layer =
                    unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr(), new_len) };
            }
        }
        let new_len = if n > 0 {
            self.swid_layer.len() + n.unsigned_abs()
        } else {
            self.swid_layer.len() - n.unsigned_abs()
        };
        match self.swid_store {
            Store::Mmap((ref file, ref mut mmap)) => {
                mmap.flush().unwrap();
                let bytes = new_len * std::mem::size_of::<Swid>();
                file.set_len(bytes as u64).unwrap();
                *mmap = unsafe { MmapMut::map_mut(file).unwrap() };
                self.swid_layer = unsafe {
                    std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut Swid, new_len)
                };
            }
            Store::Vec(ref mut vec) => {
                vec.resize(new_len, Swid::default());
                self.swid_layer =
                    unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr(), new_len) };
            }
        }
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) {
        let swid_id = self.swid_layer.len();
        let vector_id = swid_id;
        self.resize(1);
        self.swid_layer[swid_id] = swid;
        self.vector_layer[(vector_id * self.dimensions)..((vector_id + 1) * self.dimensions)]
            .copy_from_slice(q);
        let closest_leaf = self.get_closest_leaf(q);
        match closest_leaf {
            Some(leaf_chain) => {
                self.push_child(vector_id, None, leaf_chain);
            }
            None => {
                self.top_node = Some(0);
                self.nodes.push(Node::Leaf1 {
                    left_vector: vector_id,
                })
            }
        }
    }
    fn push_child(
        &mut self,
        new_vector_id: NodeID,
        new_id: Option<NodeID>,
        mut chain: Vec<NodeID>,
    ) {
        let id = chain.pop().unwrap();
        let new_child_distance = get_distance(
            self.get_vector(self.nodes[id].vector()),
            self.get_vector(new_vector_id),
            self.space,
        );
        let (new_distance, new_center, new_center_next) = match self.nodes[id] {
            Node::Leaf1 { left_vector } => {
                self.nodes[id] = Node::Leaf2 {
                    left_vector,
                    right_vector: new_vector_id,
                    distance: new_child_distance,
                };
                return;
            }
            Node::Branch1 {
                left_vector,
                left_next,
            } => {
                self.nodes[id] = Node::Branch2 {
                    left_vector,
                    left_next,
                    right_vector: new_vector_id,
                    right_next: new_id.unwrap(),
                    distance: new_child_distance,
                };
                return;
            }
            Node::Leaf2 {
                left_vector,
                right_vector,
                distance,
            } => {
                if new_child_distance < distance {
                    self.nodes[id] = Node::Leaf2 {
                        left_vector,
                        right_vector: new_vector_id,
                        distance: new_child_distance,
                    };
                    (distance, right_vector, None)
                } else {
                    (new_child_distance, new_vector_id, None)
                }
            }
            Node::Branch2 {
                left_vector,
                left_next,
                right_vector,
                right_next,
                distance,
            } => {
                if new_child_distance < distance {
                    self.nodes[id] = Node::Branch2 {
                        left_vector,
                        left_next,
                        right_vector: new_vector_id,
                        right_next: new_id.unwrap(),
                        distance: new_child_distance,
                    };
                    (distance, right_vector, Some(right_next))
                } else {
                    (new_child_distance, new_vector_id, new_id)
                }
            }
        };
        let new_center_id = self.nodes.len();
        self.nodes.push(match self.nodes[id] {
            Node::Branch2 { .. } => Node::Branch1 {
                left_vector: new_center,
                left_next: new_center_next.unwrap(),
            },
            Node::Leaf2 { .. } => Node::Leaf1 {
                left_vector: new_center,
            },
            _ => unreachable!(),
        });
        if chain.is_empty() {
            let new_parent_id = self.nodes.len();
            self.nodes.push(Node::Branch2 {
                left_vector: self.nodes[id].vector(),
                left_next: id,
                right_vector: new_center,
                right_next: new_center_id,
                distance: new_distance,
            });
            self.top_node = Some(new_parent_id);
        } else {
            self.push_child(new_center, Some(new_center_id), chain);
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        self.knn_with_filter(q, k, |_| true)
    }
    pub fn knn_with_filter(
        &self,
        q: &[F],
        k: usize,
        filter: fn((Swid, F)) -> bool,
    ) -> Vec<(Swid, F)> {
        let mut result: Vec<(u128, F)> = Vec::with_capacity(k);
        let mut current_id = self.top_node.unwrap();
        let mut current_distance = get_distance(
            q,
            self.get_vector(self.nodes[current_id].vector()),
            self.space,
        );
        let mut stack: Vec<(usize, F)> = Vec::with_capacity(k);
        while result.len() < k {
            match self.nodes[current_id] {
                Node::Leaf1 { left_vector, .. } => {
                    let tuple = (self.swid_layer[left_vector], current_distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                }
                Node::Branch1 { left_next, .. } => {
                    stack.push((left_next, current_distance));
                }
                Node::Leaf2 {
                    right_vector,
                    left_vector,
                    ..
                } => {
                    let tuple = (self.swid_layer[left_vector], current_distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                    let distance = get_distance(q, self.get_vector(right_vector), self.space);
                    let tuple = (self.swid_layer[right_vector], distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                }
                Node::Branch2 {
                    left_next,
                    right_next,
                    right_vector,
                    ..
                } => {
                    stack.push((left_next, current_distance));
                    let distance = get_distance(q, self.get_vector(right_vector), self.space);
                    stack.push((right_next, distance));
                }
            }
            stack.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            (current_id, current_distance) = match stack.pop() {
                Some(x) => x,
                None => break,
            };
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        result
    }
    pub fn remove(&mut self, swid_to_remove: Swid) -> Result<(), ()> {
        match self
            .swid_layer
            .iter()
            .position(|&swid| swid == swid_to_remove)
        {
            Some(swid_id) => {
                let last = self.swid_layer.len() - 1;
                let mut last_vector = self.vector_layer[last * self.dimensions..].to_owned();
                self.swid_layer.swap(swid_id, last);
                self.vector_layer[swid_id * self.dimensions..(swid_id + 1) * self.dimensions]
                    .swap_with_slice(last_vector.as_mut_slice());
                self.resize(-1);
            }
            None => return Err(()),
        }
        let mut new_tree = VPTree::new(self.space, self.dimensions);
        self.swid_layer
            .iter()
            .zip(self.vector_layer.chunks(self.dimensions))
            .for_each(|(swid, vector)| {
                new_tree.insert(vector, *swid);
            });
        self.nodes = new_tree.nodes;
        self.top_node = new_tree.top_node;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp_tree() {
        let mut vptree = VPTree::<f64>::new(Distance::Euclidean, 2);
        vptree.insert(&[3.0, 3.0], 4);
        vptree.insert(&[4.0, 4.0], 352);
        vptree.insert(&[5.0, 5.0], 43);
        vptree.insert(&[6.0, 6.0], 41);
        vptree.insert(&[7.0, 7.0], 35);
        vptree.insert(&[8.0, 8.0], 52);
        vptree.insert(&[9.0, 9.0], 42);
        vptree.insert(&[0.0, 0.0], 32);
        vptree.insert(&[1.0, 1.0], 222);
        vptree.insert(&[2.0, 2.0], 567);
        //dbg!(&vptree);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (32, 0.0),
                (222, std::f64::consts::SQRT_2),
                (567, 2.0 * std::f64::consts::SQRT_2)
            ]
        );
        vptree.remove(32);
        vptree.remove(4);
        //dbg!(&vptree);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (222, std::f64::consts::SQRT_2),
                (567, 2.0 * std::f64::consts::SQRT_2),
                (352, 4.0 * std::f64::consts::SQRT_2)
            ]
        );
    }
    const BENCH_DIMENSIONS: usize = 300;
    #[test]
    fn test_10000() {
        use microbench::*;
        let mut vptree = VPTree::<f32>::new(Distance::Euclidean, BENCH_DIMENSIONS);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "insert", || {
            for i in 0..10000 {
                let vector = vec![i as f32; BENCH_DIMENSIONS];
                vptree.insert(&vector, i);
            }
            vptree = VPTree::<f32>::new(Distance::Euclidean, BENCH_DIMENSIONS);
        });
        for i in 0..10000 {
            let vector = vec![i as f32; BENCH_DIMENSIONS];
            vptree.insert(&vector, i);
        }
        microbench::bench(&bench_options, "knn_topk1", || {
            for i in 0..10000 {
                let vector = vec![i as f32; BENCH_DIMENSIONS];
                vptree.knn(&vector, 1);
            }
        });
        microbench::bench(&bench_options, "knn_topk10", || {
            for i in 0..10000 {
                let vector = vec![i as f32; BENCH_DIMENSIONS];
                vptree.knn(&vector, 10);
            }
        });
        microbench::bench(&bench_options, "knn_topk100", || {
            for i in 0..10000 {
                let vector = vec![i as f32; BENCH_DIMENSIONS];
                vptree.knn(&vector, 100);
            }
        });
        microbench::bench(&bench_options, "knn_topk1000", || {
            for i in 0..10000 {
                let vector = vec![i as f32; BENCH_DIMENSIONS];
                vptree.knn(&vector, 1000);
            }
        });
    }
}
