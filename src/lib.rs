use memmap2::MmapMut;
use num_traits::Float;
use std::{
    collections::HashMap,
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
pub enum Store<T: Default + Clone> {
    Mmap((File, MmapMut)),
    Vec(Vec<T>),
}
impl<T: Default + Clone> Store<T> {
    pub fn slice_mut(&mut self) -> &mut [T] {
        match self {
            Store::Mmap((_, mmap)) => unsafe {
                std::slice::from_raw_parts_mut(
                    mmap.as_mut_ptr() as *mut T,
                    mmap.len() / std::mem::size_of::<T>(),
                )
            },
            Store::Vec(vec) => vec.as_mut_slice(),
        }
    }
    pub fn slice(&self) -> &[T] {
        match self {
            Store::Mmap((_, mmap)) => unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr() as *const T,
                    mmap.len() / std::mem::size_of::<T>(),
                )
            },
            Store::Vec(vec) => vec.as_slice(),
        }
    }
    fn resize(&mut self, n: isize) {
        let new_len = (self.slice().len() as isize + n) as usize;
        match self {
            Store::Mmap((ref file, ref mut mmap)) => {
                mmap.flush().unwrap();
                let bytes = new_len * std::mem::size_of::<T>();
                file.set_len(bytes as u64).unwrap();
                *mmap = unsafe { MmapMut::map_mut(file).unwrap() };
            }
            Store::Vec(ref mut vec) => {
                vec.resize(new_len, T::default());
            }
        };
    }
}
#[derive(Debug)]
pub struct VPTree<F: Float + Debug + Default> {
    pub nodes: Vec<Node<F>>,
    pub dimensions: usize,
    pub vector_store: Store<F>,
    pub swid_store: Store<Swid>,
    pub id_from_swid: HashMap<Swid, NodeID>,
    space: Distance,
    top_node: Option<NodeID>,
}

#[derive(Debug)]
pub enum Node<F> {
    Branch1 {
        left_next: NodeID,
        middle: Vec<F>,
    },
    Branch2 {
        left_next: NodeID,
        right_next: NodeID,
        middle: Vec<F>,
    },
    Leaf1 {
        left_vector: NodeID,
        middle: Vec<F>,
    },
    Leaf2 {
        left_vector: NodeID,
        right_vector: NodeID,
        middle: Vec<F>,
    },
}

impl<F: Float + Debug + Default> Node<F> {
    fn middle(&self) -> &[F] {
        match self {
            Node::Branch2 { middle, .. } => middle,
            Node::Leaf2 { middle, .. } => middle,
            Node::Branch1 { middle, .. } => middle,
            Node::Leaf1 { middle, .. } => middle,
        }
    }
}

impl<'a, F: Float + Debug + Default> VPTree<F> {
    pub fn new(space: Distance, dimensions: usize) -> VPTree<F> {
        VPTree {
            nodes: Vec::new(),
            dimensions,
            vector_store: Store::Vec(Vec::new()),
            swid_store: Store::Vec(Vec::new()),
            id_from_swid: HashMap::new(),
            space,
            top_node: None,
        }
    }
    pub fn new_with_store(
        space: Distance,
        dimensions: usize,
        vector_store: &Path,
        swid_store: &Path,
    ) -> VPTree<F> {
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
        let vector_mmap = unsafe { MmapMut::map_mut(&vector_file).unwrap() };
        let swid_mmap = unsafe { MmapMut::map_mut(&swid_file).unwrap() };
        let mut tree: VPTree<F> = VPTree {
            nodes: Vec::new(),
            dimensions,
            vector_store: Store::Mmap((vector_file, vector_mmap)),
            swid_store: Store::Mmap((swid_file, swid_mmap)),
            id_from_swid: HashMap::new(),
            space,
            top_node: None,
        };
        for i in 0..tree.swid_store.slice().len() {
            tree.index(i).unwrap()
        }
        tree
    }
    fn get_vector(&self, vector_id: NodeID) -> &[F] {
        self.vector_store
            .slice()
            .chunks(self.dimensions)
            .nth(vector_id)
            .unwrap()
    }
    fn get_closest_leaf(&self, q: &[F]) -> Option<Vec<NodeID>> {
        self.top_node?;
        let mut current_node = self.top_node.unwrap();
        //let mut current_distance = get_distance(q, self.nodes[current_node].middle(), self.space);
        let mut parent_chain = vec![current_node];
        loop {
            (_, current_node) = match self.nodes[current_node] {
                Node::Leaf1 { .. } | Node::Leaf2 { .. } => return Some(parent_chain),
                Node::Branch2 {
                    left_next,
                    right_next,
                    ..
                } => std::cmp::min_by(
                    (
                        get_distance(q, self.nodes[right_next].middle(), self.space),
                        right_next,
                    ),
                    (
                        get_distance(q, self.nodes[left_next].middle(), self.space),
                        left_next,
                    ),
                    |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal),
                ),
                Node::Branch1 { left_next, .. } => (
                    get_distance(q, self.nodes[left_next].middle(), self.space),
                    left_next,
                ),
            };
            parent_chain.push(current_node);
        }
    }
    fn resize(&mut self, n: isize) {
        self.vector_store.resize(n * self.dimensions as isize);
        self.swid_store.resize(n);
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) -> Result<(), ()> {
        let swid_id = self.swid_store.slice().len();
        let vector_id = swid_id;
        self.resize(1);
        self.swid_store.slice_mut()[swid_id] = swid;
        self.vector_store.slice_mut()
            [(vector_id * self.dimensions)..((vector_id + 1) * self.dimensions)]
            .copy_from_slice(q);
        self.index(vector_id)
    }
    pub fn index(&mut self, vector_id: NodeID) -> Result<(), ()> {
        let q = self.get_vector(vector_id).to_vec();
        let swid = self.swid_store.slice()[vector_id];
        let closest_leaf = self.get_closest_leaf(&q);
        if !self.id_from_swid.contains_key(&swid) {
            self.id_from_swid.insert(swid, vector_id);
        } else {
            return Err(());
        }
        match closest_leaf {
            Some(leaf_chain) => {
                self.push_child(Some(vector_id), None, leaf_chain);
            }
            None => {
                self.top_node = Some(0);
                self.nodes.push(Node::Leaf1 {
                    left_vector: vector_id,
                    middle: q,
                })
            }
        }
        Ok(())
    }
    fn push_child(
        &mut self,
        new_vector_id: Option<NodeID>,
        new_id: Option<NodeID>,
        mut chain: Vec<NodeID>,
    ) {
        let tip = chain.pop().unwrap();
        let id = tip;
        let new_center_id = match self.nodes[id] {
            Node::Leaf1 { left_vector, .. } => {
                self.nodes[id] = Node::Leaf2 {
                    left_vector: left_vector,
                    right_vector: new_vector_id.unwrap(),
                    middle: self
                        .get_vector(new_vector_id.unwrap())
                        .iter()
                        .zip(self.get_vector(left_vector))
                        .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                        .collect(),
                };
                return;
            }
            Node::Branch1 { left_next, .. } => {
                self.nodes[id] = Node::Branch2 {
                    left_next: left_next,
                    right_next: new_id.unwrap(),
                    middle: self.nodes[left_next]
                        .middle()
                        .iter()
                        .zip(self.nodes[new_id.unwrap()].middle())
                        .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                        .collect(),
                };
                return;
            }
            Node::Leaf2 {
                left_vector,
                right_vector,
                ref middle,
            } => {
                let new_distance =
                    get_distance(self.get_vector(new_vector_id.unwrap()), middle, self.space);
                let old_distance = get_distance(self.get_vector(left_vector), &middle, self.space);
                let new_center_id = self.nodes.len();
                if new_distance < old_distance {
                    self.nodes[id] = Node::Leaf2 {
                        left_vector: left_vector,
                        right_vector: new_vector_id.unwrap(),
                        middle: self
                            .get_vector(new_vector_id.unwrap())
                            .iter()
                            .zip(self.get_vector(left_vector))
                            .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                            .collect(),
                    };
                    self.nodes.push(Node::Leaf1 {
                        left_vector: right_vector,
                        middle: self.get_vector(right_vector).to_vec(),
                    });
                } else {
                    self.nodes.push(Node::Leaf1 {
                        left_vector: new_vector_id.unwrap(),
                        middle: self.get_vector(new_vector_id.unwrap()).to_vec(),
                    });
                }
                new_center_id
            }
            Node::Branch2 {
                left_next,
                right_next,
                ref middle,
            } => {
                let new_distance =
                    get_distance(self.nodes[new_id.unwrap()].middle(), middle, self.space);
                let old_distance =
                    get_distance(self.nodes[left_next].middle(), &middle, self.space);
                let new_center_id = self.nodes.len();
                if new_distance < old_distance {
                    self.nodes[id] = Node::Branch2 {
                        left_next: left_next,
                        right_next: new_id.unwrap(),
                        middle: self.nodes[left_next]
                            .middle()
                            .iter()
                            .zip(self.nodes[new_id.unwrap()].middle())
                            .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                            .collect(),
                    };
                    self.nodes.push(Node::Branch1 {
                        left_next: right_next,
                        middle: self.nodes[right_next].middle().to_vec(),
                    });
                } else {
                    self.nodes.push(Node::Branch1 {
                        left_next: new_id.unwrap(),
                        middle: self.nodes[new_id.unwrap()].middle().to_vec(),
                    });
                }
                new_center_id
            }
        };
        if chain.is_empty() {
            let new_parent_id = self.nodes.len();
            self.nodes.push(Node::Branch2 {
                left_next: id,
                right_next: new_center_id,
                middle: self.nodes[id]
                    .middle()
                    .iter()
                    .zip(self.nodes[new_center_id].middle())
                    .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                    .collect(),
            });
            self.top_node = Some(new_parent_id);
        } else {
            self.push_child(None, Some(new_center_id), chain);
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
        let mut stack: Vec<(usize, F)> = Vec::with_capacity(k);
        while result.len() < k {
            match self.nodes[current_id] {
                Node::Leaf1 { left_vector, .. } => {
                    let distance = get_distance(q, self.get_vector(left_vector), self.space);
                    let tuple = (self.swid_store.slice()[left_vector], distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                }
                Node::Branch1 { left_next, .. } => {
                    let distance = get_distance(q, self.nodes[left_next].middle(), self.space);
                    stack.push((left_next, distance));
                }
                Node::Leaf2 {
                    left_vector,
                    right_vector,
                    ..
                } => {
                    let distance = get_distance(q, self.get_vector(left_vector), self.space);
                    let tuple = (self.swid_store.slice()[left_vector], distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                    let distance = get_distance(q, self.get_vector(right_vector), self.space);
                    let tuple = (self.swid_store.slice()[right_vector], distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                }
                Node::Branch2 {
                    left_next,
                    right_next,
                    ..
                } => {
                    let distance = get_distance(q, self.nodes[left_next].middle(), self.space);
                    stack.push((left_next, distance));
                    let distance = get_distance(q, self.nodes[right_next].middle(), self.space);
                    stack.push((right_next, distance));
                }
            }
            stack.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            (current_id, _) = match stack.pop() {
                Some(x) => x,
                None => break,
            };
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        result
    }
    pub fn get_vector_by_swid(&self, swid: Swid) -> Option<&[F]> {
        match self.id_from_swid.get(&swid) {
            Some(&id) => Some(self.get_vector(id)),
            None => None,
        }
    }
    pub fn remove(&mut self, swid_to_remove: Swid) -> Result<(), ()> {
        match self.id_from_swid.get(&swid_to_remove) {
            Some(&swid_id) => {
                let last = self.swid_store.slice().len() - 1;
                let mut last_vector =
                    self.vector_store.slice()[last * self.dimensions..].to_owned();
                self.swid_store.slice_mut().swap(swid_id, last);
                self.vector_store.slice_mut()
                    [swid_id * self.dimensions..(swid_id + 1) * self.dimensions]
                    .swap_with_slice(last_vector.as_mut_slice());
                self.resize(-1);
            }
            None => return Err(()),
        }
        self.nodes.clear();
        self.id_from_swid.clear();
        self.top_node = None;
        for i in 0..self.swid_store.slice().len() {
            self.index(i).unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::abs;
    use rand::Rng;

    #[test]
    fn test_vp_tree() {
        let mut vptree = VPTree::<f64>::new(Distance::Euclidean, 2);
        vptree.insert(&[3.0, 3.0], 4).unwrap();
        vptree.insert(&[4.0, 4.0], 352).unwrap();
        vptree.insert(&[5.0, 5.0], 43).unwrap();
        vptree.insert(&[6.0, 6.0], 41).unwrap();
        vptree.insert(&[7.0, 7.0], 35).unwrap();
        vptree.insert(&[8.0, 8.0], 52).unwrap();
        vptree.insert(&[9.0, 9.0], 42).unwrap();
        vptree.insert(&[0.0, 0.0], 32).unwrap();
        vptree.insert(&[1.0, 1.0], 222).unwrap();
        vptree.insert(&[2.0, 2.0], 567).unwrap();
        //dbg!(&vptree);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (32, 0.0),
                (222, std::f64::consts::SQRT_2),
                (567, 2.0 * std::f64::consts::SQRT_2)
            ]
        );
        vptree.remove(32).unwrap();
        vptree.remove(4).unwrap();
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
                vptree.insert(&vector, i).unwrap();
            }
            vptree = VPTree::<f32>::new(Distance::Euclidean, BENCH_DIMENSIONS);
        });
        for i in 0..10000 {
            let vector = vec![i as f32; BENCH_DIMENSIONS];
            vptree.insert(&vector, i).unwrap();
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

    const LINEAR_SEARCH_SIZE: usize = 2000;
    const LINEAR_SEARCH_TOPK: usize = 1000;
    #[test]
    fn test_vs_linear_search() {
        //make LINEAR_SEARCH_SIZE random 300 dimensional vectors
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::new();
        for _ in 0..LINEAR_SEARCH_SIZE {
            let vector: Vec<f32> = (0..300).map(|_| rng.gen_range(-100.0..100.0)).collect();
            //divide each vector by 100
            let vector: Vec<f32> = vector.iter().map(|x| x / 100.0).collect();
            vectors.push(vector);
        }

        //build a vptree
        println!("building vptree");
        let mut vptree = VPTree::<f32>::new(Distance::Euclidean, BENCH_DIMENSIONS);
        for (i, vector) in vectors.iter().enumerate() {
            vptree.insert(vector, i as u128).unwrap();
        }

        //get random vector for sampling
        let random_vector = &vectors[0];

        //topk LINEAR_SEARCH_TOPK
        println!("getting topk LINEAR_SEARCH_TOPK");
        let topk = vptree.knn(random_vector, LINEAR_SEARCH_TOPK);

        //linear search topk LINEAR_SEARCH_TOPK
        println!("getting linear search topk LINEAR_SEARCH_TOPK");
        let mut linear_search_topk = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            let distance = get_distance(random_vector, vector, Distance::Euclidean);
            linear_search_topk.push((i as u128, distance));
        }
        linear_search_topk.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut differences = Vec::new();

        //compare and see if the topk is the same, if they aren't, print the index they differ
        println!("comparing topk and linear search topk");
        for (a, b) in topk.iter().zip(linear_search_topk.iter()) {
            if a.0 != b.0 {
                differences.push(abs(a.1 - b.1));
            }
        }

        //average distance of topk
        let failures = differences.len();
        println!("{} failures per {}", failures, LINEAR_SEARCH_TOPK);
        assert!(failures < LINEAR_SEARCH_TOPK / 2);
    }
}
