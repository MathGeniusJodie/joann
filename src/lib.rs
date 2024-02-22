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

#[inline(never)]
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

#[derive(Debug, Clone, Copy)]
pub struct BranchChild {
    vector_id: NodeID,
    id: NodeID,
}
#[derive(Debug, Clone, Copy)]
pub struct LeafChild {
    vector_id: NodeID,
}
#[derive(Debug)]
pub enum Child {
    BranchChild(BranchChild),
    LeafChild(LeafChild),
}
impl Child {
    fn vector_id(&self) -> NodeID {
        match self {
            Child::BranchChild(BranchChild { vector_id, .. }) => *vector_id,
            Child::LeafChild(LeafChild { vector_id }) => *vector_id,
        }
    }
    fn id(&self) -> Option<NodeID> {
        match self {
            Child::BranchChild(BranchChild { id, .. }) => Some(*id),
            Child::LeafChild(_) => None,
        }
    }
}
#[derive(Debug)]
pub struct Single<T> {
    children: [T; 1],
}
#[derive(Debug)]
pub struct Double<T, F: Float + Debug + Default> {
    children: [T; 2],
    distance: F,
}
#[derive(Debug)]
pub enum Branch<F: Float + Debug + Default> {
    Single(Single<BranchChild>),
    Double(Double<BranchChild, F>),
}
#[derive(Debug)]
pub enum Leaf<F: Float + Debug + Default> {
    Single(Single<LeafChild>),
    Double(Double<LeafChild, F>),
}
#[derive(Debug)]
pub enum Node<F: Float + Debug + Default> {
    Branch(Branch<F>),
    Leaf(Leaf<F>),
}
impl<F: Float + Debug + Default> Node<F> {
    fn len(&self) -> usize {
        match self {
            Node::Branch(Branch::Single(_)) | Node::Leaf(Leaf::Single(_)) => 1,
            Node::Branch(Branch::Double(_)) | Node::Leaf(Leaf::Double(_)) => 2,
        }
    }
    fn first(&self) -> Child {
        match self {
            Node::Branch(Branch::Single(Single { children })) => Child::BranchChild(children[0]),
            Node::Branch(Branch::Double(Double { children, .. })) => {
                Child::BranchChild(children[0])
            }
            Node::Leaf(Leaf::Single(Single { children })) => Child::LeafChild(children[0]),
            Node::Leaf(Leaf::Double(Double { children, .. })) => Child::LeafChild(children[0]),
        }
    }
    fn get(&self, i: usize) -> Option<Child> {
        match self {
            Node::Branch(Branch::Single(Single { children })) => {
                let child = children.get(i)?;
                Some(Child::BranchChild(*child))
            }
            Node::Branch(Branch::Double(Double { children, .. })) => {
                let child = children.get(i)?;
                Some(Child::BranchChild(*child))
            }
            Node::Leaf(Leaf::Single(Single { children })) => {
                let child = children.get(i)?;
                Some(Child::LeafChild(*child))
            }
            Node::Leaf(Leaf::Double(Double { children, .. })) => {
                let child = children.get(i)?;
                Some(Child::LeafChild(*child))
            }
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
            self.get_vector(self.nodes[current_node].first().vector_id()),
            self.space,
        );
        let mut parent_chain = vec![current_node];
        loop {
            match self.nodes[current_node] {
                Node::Leaf(_) => return Some(parent_chain),
                Node::Branch(_) => {}
            }
            (current_distance, current_node) = match self.nodes[current_node].get(1) {
                Some(child) => {
                    let distance = get_distance(q, self.get_vector(child.vector_id()), self.space);
                    if distance < current_distance {
                        (distance, child.id().unwrap())
                    } else {
                        (
                            current_distance,
                            self.nodes[current_node].first().id().unwrap(),
                        )
                    }
                }
                None => {
                    let child = self.nodes[current_node].first();
                    (current_distance, child.id().unwrap())
                }
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
                self.nodes.push(Node::Leaf(Leaf::Single(Single {
                    children: [LeafChild { vector_id }],
                })));
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
            self.get_vector(self.nodes[id].first().vector_id()),
            self.get_vector(new_vector_id),
            self.space,
        );
        let (new_distance, new_center) = match self.nodes[id] {
            Node::Leaf(Leaf::Single(Single { children })) => {
                self.nodes[id] = Node::Leaf(Leaf::Double(Double {
                    children: [
                        children[0],
                        LeafChild {
                            vector_id: new_vector_id,
                        },
                    ],
                    distance: new_child_distance,
                }));
                return;
            }
            Node::Branch(Branch::Single(Single { children })) => {
                self.nodes[id] = Node::Branch(Branch::Double(Double {
                    children: [
                        children[0],
                        BranchChild {
                            vector_id: new_vector_id,
                            id: new_id.unwrap(),
                        },
                    ],
                    distance: new_child_distance,
                }));
                return;
            }
            Node::Leaf(Leaf::Double(Double { children, distance })) => {
                let (new_distance, new_center) = if new_child_distance < distance {
                    let old_child = children[1];
                    self.nodes[id] = Node::Leaf(Leaf::Double(Double {
                        children: [
                            children[0],
                            LeafChild {
                                vector_id: new_vector_id,
                            },
                        ],
                        distance: new_child_distance,
                    }));
                    (distance, old_child)
                } else {
                    (
                        new_child_distance,
                        LeafChild {
                            vector_id: new_vector_id,
                        },
                    )
                };
                (new_distance, Child::LeafChild(new_center))
            }
            Node::Branch(Branch::Double(Double { children, distance })) => {
                let (new_distance, new_center) = if new_child_distance < distance {
                    let old_child = children[1];
                    self.nodes[id] = Node::Branch(Branch::Double(Double {
                        children: [
                            children[0],
                            BranchChild {
                                vector_id: new_vector_id,
                                id: new_id.unwrap(),
                            },
                        ],
                        distance: new_child_distance,
                    }));
                    (distance, old_child)
                } else {
                    (
                        new_child_distance,
                        BranchChild {
                            vector_id: new_vector_id,
                            id: new_id.unwrap(),
                        },
                    )
                };
                (new_distance, Child::BranchChild(new_center))
            }
        };
        let new_center_id = self.nodes.len();
        self.nodes.push(match new_center {
            Child::BranchChild(BranchChild { id, vector_id }) => {
                Node::Branch(Branch::Single(Single {
                    children: [BranchChild { id, vector_id }],
                }))
            }
            Child::LeafChild(LeafChild { vector_id }) => Node::Leaf(Leaf::Single(Single {
                children: [LeafChild { vector_id }],
            })),
        });
        if chain.is_empty() {
            let new_parent_id = self.nodes.len();
            self.nodes.push(Node::Branch(Branch::Double(Double {
                children: [
                    BranchChild {
                        vector_id: self.nodes[id].first().vector_id(),
                        id,
                    },
                    BranchChild {
                        vector_id: new_center.vector_id(),
                        id: new_center_id,
                    },
                ],
                distance: new_distance,
            })));
            self.top_node = Some(new_parent_id);
        } else {
            self.push_child(new_center.vector_id(), Some(new_center_id), chain);
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
            self.get_vector(self.nodes[current_id].first().vector_id()),
            self.space,
        );
        let mut stack: Vec<(usize, F)> = Vec::with_capacity(k);
        while result.len() < k {
            for i in 0..self.nodes[current_id].len() {
                let child = self.nodes[current_id].get(i).unwrap();
                let distance = if i > 0 {
                    get_distance(q, self.get_vector(child.vector_id()), self.space)
                } else {
                    current_distance
                };
                match self.nodes[current_id] {
                    Node::Leaf(_) => {
                        let tuple = (self.swid_layer[child.vector_id()], distance);
                        if filter(tuple) {
                            result.push(tuple);
                        }
                    }
                    Node::Branch(_) => {
                        stack.push((child.id().unwrap(), distance));
                    }
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
