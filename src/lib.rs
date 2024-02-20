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
const M: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    L2,
    IP,
}

#[inline(never)]
fn get_distance<F: Float + Debug>(a: &[F], b: &[F], space: Distance) -> F {
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
pub struct VPTree<'a, F: Float + Debug> {
    pub nodes: Vec<Node<F>>,
    pub dimensions: usize,
    pub vector_layer: &'a mut [F],
    pub swid_layer: &'a mut [Swid],
    pub vector_store: Store<F>,
    pub swid_store: Store<Swid>,
    space: Distance,
    top_node: Option<NodeID>,
}
#[derive(Copy, Clone, Debug, Default)]
pub struct Child<F: Float + Debug> {
    distance: F,
    vector_id: NodeID,
    id: Option<NodeID>,
}
#[derive(Debug)]
pub struct Node<F: Float + Debug> {
    pub children: Vec<Child<F>>,
    pub parent: Option<NodeID>,
}
impl<F: Float + Debug> Node<F> {
    pub fn is_leaf(&self) -> bool {
        self.children.first().unwrap().id.is_none()
    }
}

impl<'a, F: Float + Debug> VPTree<'a, F> {
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
    fn get_closest_leaf(&self, q: &[F]) -> Option<NodeID> {
        self.top_node?;
        let mut current_node = self.top_node.unwrap();
        loop {
            if self.nodes[current_node].is_leaf() {
                return Some(current_node);
            }
            current_node = self.nodes[current_node]
                .children
                .iter()
                .map(|child| {
                    let distance = get_distance(q, self.get_vector(child.vector_id), self.space);
                    (distance, child.id.unwrap())
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .1;
        }
    }
    fn push_child(&mut self, leaf_id: NodeID, vector_id: NodeID, id: Option<NodeID>) {
        if self.nodes[leaf_id].children.first().is_none() {
            self.nodes[leaf_id].children.push(Child {
                distance: F::zero(),
                vector_id,
                id,
            });
            return;
        }
        let center_id = self.nodes[leaf_id].children.first().unwrap().vector_id;
        let distance = get_distance(
            self.get_vector(vector_id),
            self.get_vector(center_id),
            self.space,
        );
        self.nodes[leaf_id].children.push(Child {
            distance,
            vector_id,
            id,
        });
    }
    fn push_child_sorted(&mut self, leaf_id: NodeID, vector_id: NodeID, id: Option<NodeID>) {
        self.push_child(leaf_id, vector_id, id);
        self.nodes[leaf_id].children.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    fn resize(&mut self, n: isize) {
        let new_len = if n > 0 {
            self.vector_layer.len() + n.unsigned_abs() * self.dimensions
        } else {
            self.vector_layer.len() - n.unsigned_abs() * self.dimensions
        };
        match self.vector_store {
            Store::Mmap((ref file, ref mut mmap)) => {
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
        match &mut self.vector_store {
            Store::Mmap((_file, mmap)) => {
                mmap.flush().unwrap();
            }
            Store::Vec(_) => (),
        };
        match &mut self.swid_store {
            Store::Mmap((_file, mmap)) => {
                mmap.flush().unwrap();
            }
            Store::Vec(_) => (),
        };
        let closest_leaf = self.get_closest_leaf(q);
        match closest_leaf {
            Some(leaf_id) => {
                self.push_child_sorted(leaf_id, vector_id, None);
                self.split(leaf_id);
            }
            None => {
                self.top_node = Some(0);
                self.nodes.push(Node {
                    children: Vec::with_capacity(M + 1),
                    parent: None,
                });
                self.push_child(0, vector_id, None);
            }
        }
    }
    fn split(&mut self, id: NodeID) {
        let node = &mut self.nodes[id];
        if node.children.len() <= M {
            return;
        }
        let new_node_id = self.nodes.len();
        let mut new_center = self.nodes[id].children.pop().unwrap();
        new_center.distance = F::zero();
        let mut new_node = Node {
            children: vec![new_center],
            parent: self.nodes[id].parent,
        };
        //put children the closest to the center of the two nodes
        let mut i = 0;
        while i < self.nodes[id].children.len() {
            let child = &self.nodes[id].children[i];
            let distance = get_distance(
                self.get_vector(child.vector_id),
                self.get_vector(new_center.vector_id),
                self.space,
            );
            if distance < child.distance
                || (distance == child.distance && new_node.children.len() < M / 2)
            {
                let mut new_child = self.nodes[id].children.remove(i);
                new_child.distance = distance;
                new_node.children.push(new_child);
            } else {
                i += 1;
            }
        }
        self.nodes[id].children.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.nodes.push(new_node);
        if self.nodes[id].parent.is_some() {
            let parent_id = self.nodes[id].parent.unwrap();
            self.push_child_sorted(parent_id, new_center.vector_id, Some(new_node_id));
            self.split(parent_id);
        } else {
            let new_parent_id = self.nodes.len();
            self.nodes.push(Node {
                children: Vec::with_capacity(M + 1),
                parent: None,
            });
            self.push_child(
                new_parent_id,
                self.nodes[id].children.first().unwrap().vector_id,
                Some(id),
            );
            self.push_child(new_parent_id, new_center.vector_id, Some(new_node_id));
            self.nodes[id].parent = Some(new_parent_id);
            self.nodes[new_node_id].parent = Some(new_parent_id);
            self.top_node = Some(new_parent_id);
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
            self.get_vector(self.nodes[current_id].children[0].vector_id),
            self.space,
        );
        let mut stack: Vec<(usize, F)> = Vec::with_capacity(k);
        while result.len() < k {
            for i in 0..self.nodes[current_id].children.len() {
                let child = &self.nodes[current_id].children[i];
                let distance = if i > 0 {
                    get_distance(q, self.get_vector(child.vector_id), self.space)
                } else {
                    current_distance
                };
                if self.nodes[current_id].is_leaf() {
                    let tuple = (self.swid_layer[child.vector_id], distance);
                    if filter(tuple) {
                        result.push(tuple);
                    }
                } else {
                    let i = match stack.binary_search_by(|a| {
                        distance
                            .partial_cmp(&a.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(i) => i,
                        Err(i) => i,
                    };
                    if stack.len() < k || i > stack.len() - k {
                        stack.insert(i, (child.id.unwrap(), distance));
                    }
                }
            }
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
                match &mut self.vector_store {
                    Store::Mmap((_file, mmap)) => {
                        mmap.flush().unwrap();
                    }
                    Store::Vec(_) => (),
                };
                match &mut self.swid_store {
                    Store::Mmap((_file, mmap)) => {
                        mmap.flush().unwrap();
                    }
                    Store::Vec(_) => (),
                };
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
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (222, std::f64::consts::SQRT_2),
                (567, 2.0 * std::f64::consts::SQRT_2),
                (352, 4.0 * std::f64::consts::SQRT_2)
            ]
        );
        //dbg!(&vptree);
    }

    #[test]
    fn test_10000() {
        use microbench::*;
        let mut vptree = VPTree::<f32>::new(Distance::Euclidean, 300);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "insert", || {
            for i in 0..10000 {
                let vector = vec![i as f32; 300];
                vptree.insert(&vector, i);
            }
            vptree = VPTree::<f32>::new(Distance::Euclidean, 300);
        });
        for i in 0..10000 {
            let vector = vec![i as f32; 300];
            vptree.insert(&vector, i);
        }
        microbench::bench(&bench_options, "knn_topk1", || {
            for i in 0..10000 {
                let vector = vec![i as f32; 300];
                vptree.knn(&vector, 1);
            }
        });
        microbench::bench(&bench_options, "knn_topk10", || {
            for i in 0..10000 {
                let vector = vec![i as f32; 300];
                vptree.knn(&vector, 10);
            }
        });
        microbench::bench(&bench_options, "knn_topk100", || {
            for i in 0..10000 {
                let vector = vec![i as f32; 300];
                vptree.knn(&vector, 100);
            }
        });
        microbench::bench(&bench_options, "knn_topk1000", || {
            for i in 0..10000 {
                let vector = vec![i as f32; 300];
                vptree.knn(&vector, 1000);
            }
        });
    }
}
