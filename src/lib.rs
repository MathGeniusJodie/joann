use memmap2::MmapMut;
use num_traits::Float;
use std::{
    collections::HashMap,
    fmt::Debug,
    fs::{File, OpenOptions},
    path::Path,
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
    pub nodeid_from_swid: HashMap<Swid, NodeID>,
    space: Distance,
    top_node: Option<NodeID>,
}

#[derive(Debug)]
pub enum Node<F> {
    Branch1 {
        left_next: NodeID,
        parent: Option<NodeID>,
        middle: Vec<F>,
    },
    Branch2 {
        left_next: NodeID,
        right_next: NodeID,
        parent: Option<NodeID>,
        middle: Vec<F>,
    },
    Leaf1 {
        left_vector: NodeID,
        parent: Option<NodeID>,
        middle: Vec<F>,
    },
    Leaf2 {
        left_vector: NodeID,
        right_vector: NodeID,
        parent: Option<NodeID>,
        middle: Vec<F>,
    },
    Leaf0 {
        parent: Option<NodeID>,
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
            Node::Leaf0 { middle, .. } => middle,
        }
    }
    fn set_parent(&mut self, parent: Option<NodeID>) {
        match self {
            Node::Branch2 {
                parent: ref mut p, ..
            } => *p = parent,
            Node::Leaf2 {
                parent: ref mut p, ..
            } => *p = parent,
            Node::Branch1 {
                parent: ref mut p, ..
            } => *p = parent,
            Node::Leaf1 {
                parent: ref mut p, ..
            } => *p = parent,
            Node::Leaf0 {
                parent: ref mut p, ..
            } => *p = parent,
        }
    }
    fn parent(&self) -> Option<NodeID> {
        match self {
            Node::Branch2 { parent, .. } => *parent,
            Node::Leaf2 { parent, .. } => *parent,
            Node::Branch1 { parent, .. } => *parent,
            Node::Leaf1 { parent, .. } => *parent,
            Node::Leaf0 { parent, .. } => *parent,
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
            nodeid_from_swid: HashMap::new(),
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
            nodeid_from_swid: HashMap::new(),
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
    fn get_closest_leaf(&self, q: &[F]) -> Option<NodeID> {
        self.top_node?;
        let mut current_node = self.top_node.unwrap();
        loop {
            (_, current_node) = match self.nodes[current_node] {
                Node::Leaf1 { .. } | Node::Leaf2 { .. } | Node::Leaf0 { .. } => {
                    return Some(current_node)
                }
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
                Node::Branch1 { left_next, .. } => (F::zero(), left_next),
            };
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
        match closest_leaf {
            Some(leaf_chain) => {
                self.push_child(Some(vector_id), None, leaf_chain);
            }
            None => {
                self.top_node = Some(0);
                self.nodes.push(Node::Leaf1 {
                    left_vector: vector_id,
                    parent: None,
                    middle: q,
                });
                self.nodeid_from_swid.insert(swid, 0);
            }
        }
        Ok(())
    }
    fn push_child(
        &mut self,
        new_vector_id: Option<NodeID>,
        new_id: Option<NodeID>,
        tip: NodeID
    ) {
        let id = tip;
        let new_center_id = match self.nodes[id] {
            Node::Leaf0 { parent, .. } => {
                self.nodes[id] = Node::Leaf1 {
                    left_vector: new_vector_id.unwrap(),
                    parent,
                    middle: self.get_vector(new_vector_id.unwrap()).to_vec(),
                };
                let swid = self.swid_store.slice()[new_vector_id.unwrap()];
                self.nodeid_from_swid.insert(swid, id);
                if self.nodes[id].parent().is_some() {
                    self.recalculate_middle(self.nodes[id].parent().unwrap());
                }
                return;
            }
            Node::Leaf1 {
                left_vector,
                parent,
                ..
            } => {
                self.nodes[id] = Node::Leaf2 {
                    left_vector: left_vector,
                    right_vector: new_vector_id.unwrap(),
                    parent,
                    middle: self
                        .get_vector(new_vector_id.unwrap())
                        .iter()
                        .zip(self.get_vector(left_vector))
                        .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                        .collect(),
                };
                if self.nodes[id].parent().is_some() {
                    self.recalculate_middle(self.nodes[id].parent().unwrap());
                }
                let swid = self.swid_store.slice()[new_vector_id.unwrap()];
                self.nodeid_from_swid.insert(swid, id);
                return;
            }
            Node::Branch1 {
                left_next, parent, ..
            } => {
                self.nodes[id] = Node::Branch2 {
                    left_next: left_next,
                    right_next: new_id.unwrap(),
                    parent,
                    middle: self.nodes[left_next]
                        .middle()
                        .iter()
                        .zip(self.nodes[new_id.unwrap()].middle())
                        .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                        .collect(),
                };
                if self.nodes[id].parent().is_some() {
                    self.recalculate_middle(self.nodes[id].parent().unwrap());
                }
                return;
            }
            Node::Leaf2 {
                left_vector,
                right_vector,
                parent,
                ref middle,
                ..
            } => {
                let new_distance =
                    get_distance(self.get_vector(new_vector_id.unwrap()), middle, self.space);
                let old_distance = get_distance(self.get_vector(right_vector), &middle, self.space);
                let new_center_id = self.nodes.len();
                if new_distance < old_distance {
                    self.nodes[id] = Node::Leaf2 {
                        left_vector: left_vector,
                        right_vector: new_vector_id.unwrap(),
                        parent,
                        middle: self
                            .get_vector(new_vector_id.unwrap())
                            .iter()
                            .zip(self.get_vector(left_vector))
                            .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                            .collect(),
                    };
                    self.nodes.push(Node::Leaf1 {
                        left_vector: right_vector,
                        parent,
                        middle: self.get_vector(right_vector).to_vec(),
                    });
                    let swid = self.swid_store.slice()[new_vector_id.unwrap()];
                    self.nodeid_from_swid.insert(swid, id);
                    let swid = self.swid_store.slice()[right_vector];
                    self.nodeid_from_swid.insert(swid, new_center_id);
                } else {
                    self.nodes.push(Node::Leaf1 {
                        left_vector: new_vector_id.unwrap(),
                        parent,
                        middle: self.get_vector(new_vector_id.unwrap()).to_vec(),
                    });
                    let swid = self.swid_store.slice()[new_vector_id.unwrap()];
                    self.nodeid_from_swid.insert(swid, new_center_id);
                }
                new_center_id
            }
            Node::Branch2 {
                left_next,
                right_next,
                parent,
                ref middle,
            } => {
                let new_distance =
                    get_distance(self.nodes[new_id.unwrap()].middle(), middle, self.space);
                let old_distance =
                    get_distance(self.nodes[right_next].middle(), &middle, self.space);
                let new_center_id = self.nodes.len();
                if new_distance < old_distance {
                    self.nodes[id] = Node::Branch2 {
                        left_next,
                        right_next: new_id.unwrap(),
                        parent,
                        middle: self.nodes[left_next]
                            .middle()
                            .iter()
                            .zip(self.nodes[new_id.unwrap()].middle())
                            .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                            .collect(),
                    };
                    self.nodes.push(Node::Branch1 {
                        left_next: right_next,
                        parent,
                        middle: self.nodes[right_next].middle().to_vec(),
                    });
                    self.nodes[right_next].set_parent(Some(new_center_id));
                } else {
                    self.nodes.push(Node::Branch1 {
                        left_next: new_id.unwrap(),
                        parent,
                        middle: self.nodes[new_id.unwrap()].middle().to_vec(),
                    });
                    self.nodes[new_id.unwrap()].set_parent(Some(new_center_id));
                }
                new_center_id
            }
        };
        if self.nodes[id].parent().is_none() {
            let new_parent_id = self.nodes.len();
            self.nodes.push(Node::Branch2 {
                left_next: id,
                right_next: new_center_id,
                parent: None,
                middle: self.nodes[id]
                    .middle()
                    .iter()
                    .zip(self.nodes[new_center_id].middle())
                    .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                    .collect(),
            });
            self.nodes[id].set_parent(Some(new_parent_id));
            self.nodes[new_center_id].set_parent(Some(new_parent_id));
            self.top_node = Some(new_parent_id);
        } else {
            self.push_child(None, Some(new_center_id), self.nodes[id].parent().unwrap());
        }
    }
    fn recalculate_middle(&mut self, parent: NodeID) {
        let new_middle = match self.nodes[parent] {
            Node::Branch1 { left_next, .. } => self.nodes[left_next].middle().to_vec(),
            Node::Branch2 {
                left_next,
                right_next,
                ..
            } => self.nodes[left_next]
                .middle()
                .iter()
                .zip(self.nodes[right_next].middle())
                .map(|(&a, &b)| (a + b) * F::from(0.5).unwrap())
                .collect(),
            _ => unreachable!(),
        };
        match self.nodes[parent] {
            Node::Branch1 { ref mut middle, .. } => {
                *middle = new_middle;
            }
            Node::Branch2 { ref mut middle, .. } => {
                *middle = new_middle;
            }
            _ => {}
        }
        if self.nodes[parent].parent().is_some() {
            self.recalculate_middle(self.nodes[parent].parent().unwrap());
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        let mut count = 0;
        let mut result = self.search(q, |_| {
            count += 1;
            (false, count > k + 1)
        });
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        result
    }
    pub fn search(
        &self,
        q: &[F],
        mut filter: impl FnMut((Swid, F)) -> (bool, bool),
    ) -> Vec<(Swid, F)> {
        let mut result: Vec<(u128, F)> = Vec::new();
        let mut current_id = self.top_node.unwrap();
        let mut stack: Vec<(usize, F)> = Vec::new();
        loop {
            match self.nodes[current_id] {
                Node::Leaf0 { .. } => {}
                Node::Leaf1 { left_vector, .. } => {
                    let distance = get_distance(q, self.get_vector(left_vector), self.space);
                    let tuple = (self.swid_store.slice()[left_vector], distance);
                    let (continue_flag, break_flag) = filter(tuple);
                    if break_flag {
                        break;
                    }
                    if !continue_flag {
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
                    let (continue_flag, break_flag) = filter(tuple);
                    if break_flag {
                        break;
                    }
                    if !continue_flag {
                        result.push(tuple);
                    }
                    let distance = get_distance(q, self.get_vector(right_vector), self.space);
                    let tuple = (self.swid_store.slice()[right_vector], distance);
                    let (continue_flag, break_flag) = filter(tuple);
                    if break_flag {
                        break;
                    }
                    if !continue_flag {
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
        result
    }
    fn get_vector_id_by_swid(&self, swid: Swid) -> Option<NodeID> {
        match self.nodeid_from_swid.get(&swid) {
            Some(&id) => match self.nodes[id] {
                Node::Leaf1 { left_vector, .. } => Some(left_vector),
                Node::Leaf2 {
                    left_vector,
                    right_vector,
                    ..
                } => {
                    let left_swid = self.swid_store.slice()[left_vector];
                    let right_swid = self.swid_store.slice()[right_vector];
                    if left_swid == swid {
                        Some(left_vector)
                    } else if right_swid == swid {
                        Some(right_vector)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            None => None,
        }
    }
    pub fn remove(&mut self, swid_to_remove: Swid) -> Result<(), ()> {
        let swid_id = match self.get_vector_id_by_swid(swid_to_remove) {
            Some(id) => id,
            None => return Err(()),
        };
        let node_id_to_remove = *self.nodeid_from_swid.get(&swid_to_remove).unwrap();
        // remove reference to swid_to_remove in the nodes
        match self.nodes[node_id_to_remove] {
            Node::Leaf2 {
                left_vector,
                right_vector,
                parent,
                ..
            } => {
                let left_swid = self.swid_store.slice()[left_vector];
                let right_swid = self.swid_store.slice()[right_vector];
                if left_swid == swid_to_remove {
                    self.nodes[node_id_to_remove] = Node::Leaf1 {
                        left_vector: right_vector,
                        parent,
                        middle: self.get_vector(right_vector).to_vec(),
                    };
                } else if right_swid == swid_to_remove {
                    self.nodes[node_id_to_remove] = Node::Leaf1 {
                        left_vector,
                        parent,
                        middle: self.get_vector(left_vector).to_vec(),
                    };
                }
                if parent.is_some() {
                    self.recalculate_middle(parent.unwrap());
                }
            }
            Node::Leaf1 {
                parent, ref middle, ..
            } => {
                self.nodes[node_id_to_remove] = Node::Leaf0 {
                    parent,
                    middle: middle.to_vec(),
                };
                if parent.is_some() {
                    self.recalculate_middle(parent.unwrap());
                }
            }
            _ => {}
        }
        let last_swid_id = self.swid_store.slice().len() - 1;
        let last_swid = self.swid_store.slice()[last_swid_id];
        //swap the last swid with the swid to remove
        self.swid_store.slice_mut().swap(swid_id, last_swid_id);
        //swap the last vector with the vector to remove
        let mut last_vector =
            self.vector_store.slice()[last_swid_id * self.dimensions..].to_owned();
        self.vector_store.slice_mut()[swid_id * self.dimensions..(swid_id + 1) * self.dimensions]
            .swap_with_slice(last_vector.as_mut_slice());
        self.resize(-1);
        // swap references to the last swid with the swid to remove
        let last_node_id = *self.nodeid_from_swid.get(&last_swid).unwrap();
        self.nodeid_from_swid.insert(swid_to_remove, last_node_id);
        self.nodeid_from_swid.remove(&last_swid);
        // replace last_swid_id with swid_id in the nodes
        match self.nodes[last_node_id] {
            Node::Leaf1 {
                ref mut left_vector,
                ..
            } => {
                if *left_vector == last_swid_id {
                    *left_vector = swid_id;
                }
            }
            Node::Leaf2 {
                ref mut left_vector,
                ref mut right_vector,
                ..
            } => {
                if *left_vector == last_swid_id {
                    *left_vector = swid_id;
                } else if *right_vector == last_swid_id {
                    *right_vector = swid_id;
                }
            }
            _ => {}
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
        dbg!(&vptree);
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
    /*
    fn repeatedly_test_vs_linear_search() {
        for _ in 0..1000 {
            test_vs_linear_search();
        }
    }*/
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
        let mut vptree = VPTree::<f32>::new(Distance::Euclidean, BENCH_DIMENSIONS);
        for (i, vector) in vectors.iter().enumerate() {
            vptree.insert(vector, i as u128).unwrap();
        }

        //get random vector for sampling
        let random_vector = &vectors[0];

        //topk LINEAR_SEARCH_TOPK
        let topk = vptree.knn(random_vector, LINEAR_SEARCH_TOPK);

        //linear search topk LINEAR_SEARCH_TOPK
        let mut linear_search_topk = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            let distance = get_distance(random_vector, vector, Distance::Euclidean);
            linear_search_topk.push((i as u128, distance));
        }
        linear_search_topk.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut differences = Vec::new();

        //compare and see if the topk is the same, if they aren't, print the index they differ
        for (a, b) in topk.iter().zip(linear_search_topk.iter()) {
            if a.0 != b.0 {
                differences.push(abs(a.1 - b.1));
            }
        }

        //average distance of topk
        let recall = 1.0 - differences.len() as f64 / LINEAR_SEARCH_TOPK as f64;
        println!("recall: {} ", recall);
    }
}
