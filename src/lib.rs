use bit_vec::BitVec;
use num_traits::Float;
use std::fmt::Debug;

type Swid = u128;
type NodeID = usize;
const MAX_LAYER: usize = 16;

#[derive(Copy, Clone, Debug, Default)]
pub struct Neighbor<F: Float + Debug + Default> {
    id: NodeID,
    distance: F,
}
impl<F: Float + Debug + Default> PartialEq for Neighbor<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl<F: Float + Debug + Default> Eq for Neighbor<F> {}
impl<F: Float + Debug + Default> PartialOrd for Neighbor<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<F: Float + Debug + Default> Ord for Neighbor<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.distance == other.distance {
            return self.id.cmp(&other.id);
        }
        match self.distance.partial_cmp(&other.distance) {
            Some(ord) => ord,
            None => std::cmp::Ordering::Equal,
        }
    }
}

#[derive(Debug)]
pub struct HNSW<F: Float + Debug + Default> {
    pub layers: [Vec<Node<F>>; MAX_LAYER],
    pub dimensions: usize,
    pub swid_layer: Vec<Swid>,
    pub vector_layer: Vec<F>,
    ef_construction: usize,
    space: Distance,
    m: usize,
}

#[derive(Debug)]
pub struct Node<F: Float + Debug + Default> {
    neighbors: Vec<Neighbor<F>>,
    lower_id: NodeID,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    L2,
    IP,
}
fn pop_min<T: Ord>(v: &mut Vec<T>) -> T {
    let min_index = v
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cmp(b))
        .unwrap()
        .0;
    v.swap_remove(min_index)
}
fn pop_max<T: Ord>(v: &mut Vec<T>) -> T {
    let max_index = v
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .unwrap()
        .0;
    v.swap_remove(max_index)
}
#[inline(never)]
fn get_distance<F: Float + Debug + Default>(a: &[F], b: &[F], space: Distance) -> F {
    match space {
        Distance::Euclidean => {
            let mut sum: F = F::zero();
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum = diff.mul_add(diff, sum);
            }
            sum.sqrt()
        }
        Distance::Cosine => {
            let mut dot = F::zero();
            let mut xx = F::zero();
            let mut yy = F::zero();

            for i in 0..a.len() {
                let xi = a[i];
                let yi = b[i];
                dot = dot + xi * yi;
                xx = xx + xi * xi;
                yy = yy + yi * yi;
            }

            //handle 0 vectors
            if xx * yy <= F::zero() {
                return F::zero();
            }

            F::one() - dot / (xx * yy).sqrt()
        }
        Distance::L2 => {
            let mut sum: F = F::zero();
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum = diff.mul_add(diff, sum);
            }
            sum
        }
        Distance::IP => {
            let mut dot: F = F::zero();
            for i in 0..a.len() {
                dot = a[i].mul_add(b[i], dot);
            }
            dot
        }
    }
}

impl<F: Float + Debug + Default> HNSW<F> {
    pub fn new(ef_construction: usize, space: Distance, dimensions: usize, m: usize) -> HNSW<F> {
        let layers: [Vec<Node<F>>; MAX_LAYER] = Default::default();
        HNSW {
            layers,
            dimensions,
            swid_layer: Vec::new(),
            vector_layer: Vec::new(),
            ef_construction,
            space,
            m,
        }
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) {
        let l =
            ((-rand::random::<f64>().ln() * (1.0f64 / 16.0f64.ln())) as usize).min(MAX_LAYER - 1);
        let mut ep = 0;
        for lc in (l + 1..MAX_LAYER).rev() {
            ep = match self.search_layer(q, ep, 1, lc).first() {
                Some(n) => self.layers[lc][n.id].lower_id,
                None => 0,
            };
        }

        for lc in (0..=l).rev() {
            let mut n = self.search_layer(q, ep, self.ef_construction, lc);
            n.truncate(self.m);
            let qid = self.layers[lc].len();
            for neighbor in &n {
                self.layers[lc][neighbor.id].neighbors.push(Neighbor {
                    id: qid,
                    distance: neighbor.distance,
                });
                self.layers[lc][neighbor.id].neighbors.sort();
                self.layers[lc][neighbor.id].neighbors.truncate(self.m);
            }
            let lower_id = if lc == 0 {
                self.swid_layer.len()
            } else {
                self.layers[lc - 1].len()
            };
            ep = match n.first() {
                Some(n) => self.layers[lc][n.id].lower_id,
                None => 0,
            };
            self.layers[lc].push(Node {
                neighbors: n,
                lower_id,
            });
        }
        self.swid_layer.push(swid);
        self.vector_layer.extend_from_slice(q);
    }
    pub fn remove(&mut self, swid_to_remove: Swid) {
        let mut new_hnsw: HNSW<F> =
            HNSW::new(self.ef_construction, self.space, self.dimensions, self.m);
        self.swid_layer
            .iter()
            .zip(self.vector_layer.chunks(self.dimensions))
            .for_each(|(swid, vector)| {
                if *swid != swid_to_remove {
                    new_hnsw.insert(vector, *swid);
                }
            });
        self.layers = new_hnsw.layers;
        self.swid_layer = new_hnsw.swid_layer;
        self.vector_layer = new_hnsw.vector_layer;
    }
    fn search_layer(&self, q: &[F], ep: usize, ef: usize, layer: usize) -> Vec<Neighbor<F>> {
        if ef > self.layers[layer].len() {
            let len = self.layers[layer].len();
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                result.push(Neighbor {
                    id: i,
                    distance: get_distance(self.get_vector(layer, i), q, self.space),
                });
            }
            result.sort();
            return result;
        }
        let ep_dist = get_distance(self.get_vector(layer, ep), q, self.space);
        let mut visited = BitVec::from_elem(self.layers[layer].len(), false);
        let mut candidates = Vec::with_capacity(self.m);
        let mut result = Vec::with_capacity(ef);
        visited.set(ep, true);
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        result.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        let mut max_dist = ep_dist;
        while !candidates.is_empty() {
            let c = pop_min(&mut candidates);
            if c.distance > max_dist {
                break;
            }
            for e in &self.layers[layer][c.id].neighbors {
                if visited.get(e.id).unwrap() {
                    continue;
                }
                visited.set(e.id, true);
                let d_e = get_distance(self.get_vector(layer, e.id), q, self.space);
                if d_e < max_dist || result.len() < ef {
                    result.push(Neighbor {
                        id: e.id,
                        distance: d_e,
                    });
                    max_dist = max_dist.max(d_e);
                    candidates.push(Neighbor {
                        id: e.id,
                        distance: d_e,
                    });
                    if result.len() > ef {
                        max_dist = pop_max(&mut result).distance;
                    }
                }
            }
        }
        result.sort();
        result
    }
    fn get_swid(&self, layer: usize, id: NodeID) -> Swid {
        let lower = self.layers[layer][id].lower_id;
        if layer == 0 {
            self.swid_layer[lower]
        } else {
            self.get_swid(layer - 1, lower)
        }
    }
    fn get_vector(&self, layer: usize, id: NodeID) -> &[F] {
        let lower = self.layers[layer][id].lower_id;
        if layer == 0 {
            self.vector_layer
                .chunks(self.dimensions)
                .nth(lower)
                .unwrap()
        } else {
            self.get_vector(layer - 1, lower)
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        let ef_search = self.ef_construction.max(k);
        let mut ep = 0;
        for lc in (1..MAX_LAYER).rev() {
            ep = match self.search_layer(q, ep, 1, lc).first() {
                Some(n) => self.layers[lc][n.id].lower_id,
                None => 0,
            };
        }
        self.search_layer(q, ep, ef_search, 0)
            .iter()
            .take(k)
            .map(|n| (self.get_swid(0, n.id), n.distance))
            .collect()
    }
}

pub struct VPTree<F: Float + Debug + Default> {
    pub layers: [Vec<VPNode<F>>; MAX_LAYER],
    pub dimensions: usize,
    pub swid_layer: Vec<Swid>,
    pub vector_layer: Vec<F>,
    ef_construction: usize,
    space: Distance,
    m: usize,
}

#[derive(Debug)]
pub struct VPNode<F: Float + Debug + Default> {
    pub children: Vec<Neighbor<F>>,
    pub parent: Option<NodeID>,
    pub center: NodeID,
}

impl<F: Float + Debug + Default> VPTree<F> {
    pub fn new(ef_construction: usize, space: Distance, dimensions: usize, m: usize) -> VPTree<F> {
        let layers: [Vec<VPNode<F>>; MAX_LAYER] = Default::default();
        VPTree {
            layers,
            dimensions,
            swid_layer: Vec::new(),
            vector_layer: Vec::new(),
            ef_construction,
            space,
            m,
        }
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) {
        if q.len() != self.dimensions {
            panic!("Dimensions do not match");
        }
        let id = self.swid_layer.len();
        self.swid_layer.push(swid);
        self.vector_layer.extend_from_slice(q);
        match self.find_closest_node(q) {
            Some(closest) => {
                let distance = get_distance(
                    self.get_vector(0, self.layers[0][closest].center),
                    q,
                    self.space,
                );
                self.layers[0][closest]
                    .children
                    .push(Neighbor { id, distance });
                self.recursive_split(0, closest);
            }
            None => {
                self.layers[0].push(VPNode {
                    children: vec![Neighbor {
                        id,
                        distance: F::zero(),
                    }],
                    parent: None,
                    center: id,
                });
            }
        }
    }
    fn recursive_split(&mut self, layer: usize, id: NodeID) {
        if self.layers[layer][id].children.len() > self.m {
            let new_id = self.layers[layer].len();
            let center = self.layers[layer][id].children.iter().max().unwrap();
            let center_id = center.id;
            let center_distance = center.distance;
            let new_node = VPNode {
                children: Vec::with_capacity(self.m + 1),
                parent: self.layers[layer][id].parent,
                center: center_id,
            };
            self.layers[layer].push(new_node);
            let mut i = 0;
            if center_distance != F::zero() {
                while i < self.layers[layer][id].children.len() {
                    let child = self.layers[layer][id].children[i];
                    let distance = get_distance(
                        self.get_vector(layer, center_id),
                        self.get_vector(layer, child.id),
                        self.space,
                    );
                    if distance < child.distance {
                        self.layers[layer][new_id].children.push(Neighbor {
                            id: child.id,
                            distance,
                        });
                        if layer > 0 {
                            self.layers[layer - 1][child.id].parent = Some(new_id);
                        }
                        self.layers[layer][id].children.swap_remove(i);
                    } else {
                        i += 1;
                    }
                }
            } else {
                self.layers[layer][new_id].children = self.layers[layer][id].children.split_off(self.m/2);
            }
            if self.layers[layer][id].parent.is_some() {
                // todo: double check this
                let parent_id = self.layers[layer][id].parent.unwrap();
                let parent_center = self.layers[layer + 1][parent_id].center;
                let distance = get_distance(
                    self.get_vector(layer, center_id),
                    self.get_vector(layer + 1, parent_center),
                    self.space,
                );
                self.layers[layer + 1][parent_id].children.push(Neighbor {
                    id: new_id,
                    distance,
                });
                self.recursive_split(layer + 1, self.layers[layer][id].parent.unwrap());
            } else {
                let new_parent_id = self.layers[layer + 1].len();
                self.layers[layer + 1].push(VPNode {
                    children: vec![
                        Neighbor {
                            id,
                            distance: F::zero(),
                        },
                        Neighbor {
                            id: new_id,
                            distance: get_distance(
                                self.get_vector(layer, id),
                                self.get_vector(layer, new_id),
                                self.space,
                            ),
                        },
                    ],
                    parent: None,
                    center: id,
                });
                self.layers[layer][id].parent = Some(new_parent_id);
                self.layers[layer][new_id].parent = Some(new_parent_id);
            }
        }
    }
    fn find_closest_node(&self, q: &[F]) -> Option<NodeID> {
        if self.layers[0].is_empty() {
            return None;
        }
        let mut layer = MAX_LAYER - 1;
        loop {
            if self.layers[layer].is_empty() {
                layer -= 1;
            } else {
                break;
            }
        }
        let mut id = self.layers[layer]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                get_distance(self.get_vector(layer, a.center), q, self.space)
                    .partial_cmp(&get_distance(
                        self.get_vector(layer, b.center),
                        q,
                        self.space,
                    ))
                    .unwrap()
            })
            .unwrap()
            .0;
        loop {
            if layer == 0 {
                return Some(id);
            }
            id = self.layers[layer][id]
                .children
                .iter()
                .min_by(|a, b| {
                    get_distance(self.get_vector(layer, a.id), q, self.space)
                        .partial_cmp(&get_distance(self.get_vector(layer, b.id), q, self.space))
                        .unwrap()
                })
                .unwrap()
                .id;
            layer -= 1;
        }
    }
    fn get_vector(&self, layer: usize, id: NodeID) -> &[F] {
        if layer == 0 {
            self.vector_layer.chunks(self.dimensions).nth(id).unwrap()
        } else {
            self.get_vector(layer - 1, self.layers[layer - 1][id].center)
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        let mut res: Vec<(Swid, F)> = self
            .knn_ids(q, k)
            .iter()
            .map(|id| {
                (
                    self.swid_layer[*id],
                    get_distance(self.get_vector(0, *id), q, self.space),
                )
            })
            .collect();
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        res.truncate(k);
        res
    }
    fn knn_ids(&self, q: &[F], k: usize) -> Vec<NodeID> {
        let closest = match self.find_closest_node(q) {
            Some(closest) => closest,
            None => return Vec::new(),
        };
        let ef = self.ef_construction.max(k);
        // go up layers until ef is reached
        let mut layer = 0;
        let mut id = closest;
        let mut n = self.m;
        loop {
            if n > ef || self.layers[layer][id].parent.is_none() {
                break;
            }
            n *= self.m;
            id = self.layers[layer][id].parent.unwrap();
            layer += 1;
        }
        self.get_all_children(layer, id)
    }

    fn get_all_children(&self, layer: usize, id: NodeID) -> Vec<NodeID> {
        let mut result = Vec::new();
        let mut stack = vec![(layer, id)];

        while let Some((layer, id)) = stack.pop() {
            let children = &self.layers[layer][id].children;
            if layer == 0 {
                result.extend(children.iter().map(|child| child.id));
                continue;
            } else {
                stack.extend(children.iter().map(|child| (layer - 1, child.id)));
            }
        }
        result
    }
    pub fn remove(&mut self, swid_to_remove: Swid) {
        let id = self
            .swid_layer
            .iter()
            .position(|swid| *swid == swid_to_remove)
            .unwrap();
        self.layers[0].iter_mut().for_each(|node| {
            node.children.retain(|n| n.id != id);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw() {
        let mut hnsw = HNSW::<f64>::new(16, Distance::Euclidean, 2, 16);
        hnsw.insert(&[3.0, 3.0], 3);
        hnsw.insert(&[4.0, 4.0], 4);
        hnsw.insert(&[5.0, 5.0], 5);
        hnsw.insert(&[6.0, 6.0], 6);
        hnsw.insert(&[7.0, 7.0], 7);
        hnsw.insert(&[8.0, 8.0], 8);
        hnsw.insert(&[9.0, 9.0], 9);
        hnsw.insert(&[0.0, 0.0], 420);
        hnsw.insert(&[1.0, 1.0], 69);
        hnsw.insert(&[2.0, 2.0], 42);
        //dbg!(&hnsw);
        assert_eq!(
            hnsw.knn(&[0.0, 0.0], 3),
            vec![
                (420, 0.0),
                (69, std::f64::consts::SQRT_2),
                (42, 2.0 * std::f64::consts::SQRT_2)
            ]
        );
    }
    #[test]
    fn test_vp_tree() {
        let mut vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 16);
        vptree.insert(&[3.0, 3.0], 3);
        vptree.insert(&[4.0, 4.0], 4);
        vptree.insert(&[5.0, 5.0], 5);
        vptree.insert(&[6.0, 6.0], 6);
        vptree.insert(&[7.0, 7.0], 7);
        vptree.insert(&[8.0, 8.0], 8);
        vptree.insert(&[9.0, 9.0], 9);
        vptree.insert(&[0.0, 0.0], 420);
        vptree.insert(&[1.0, 1.0], 69);
        vptree.insert(&[2.0, 2.0], 42);
        //dbg!(&vptree);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (420, 0.0),
                (69, std::f64::consts::SQRT_2),
                (42, 2.0 * std::f64::consts::SQRT_2)
            ]
        );
    }

    #[test]
    fn test_insert_10000() {
        use microbench::*;
        let mut hnsw = HNSW::<f64>::new(16, Distance::Euclidean, 2, 16);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "hnsw_test_insert_10000", || {
            for i in 0..10000 {
                hnsw.insert(&[i as f64, i as f64], i);
            }
            hnsw = HNSW::<f64>::new(16, Distance::Euclidean, 2, 16);
        });
        for i in 0..10000 {
            hnsw.insert(&[i as f64, i as f64], i);
        }
        microbench::bench(&bench_options, "hnsw_test_knn_10000", || {
            for i in 0..10000 {
                hnsw.knn(&[i as f64, i as f64], 3);
            }
        });
        let mut vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 4);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "vp_tree_test_insert_10000", || {
            for i in 0..10000 {
                vptree.insert(&[i as f64, i as f64], i);
            }
            vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 4);
        });
        for i in 0..10000 {
            vptree.insert(&[i as f64, i as f64], i);
        }
        microbench::bench(&bench_options, "vp_tree_test_knn_10000", || {
            for i in 0..10000 {
                vptree.knn(&[i as f64, i as f64], 3);
            }
        });
    }
}
