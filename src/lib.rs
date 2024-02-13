use num_traits::Float;
use std::fmt::Debug;

type Swid = u128;
type NodeID = usize;

#[derive(Copy, Clone, Debug, Default)]
pub struct Neighbor<F: Float + Debug + Default> {
    id: NodeID,
    distance: F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    L2,
    IP,
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

pub struct VPTree<F: Float + Debug + Default> {
    pub layers: Vec<VPNode<F>>,
    pub dimensions: usize,
    pub swid_layer: Vec<Swid>,
    pub vector_layer: Vec<F>,
    ef_construction: usize,
    space: Distance,
    m: usize,
    top_node: Option<NodeID>,
}

#[derive(Debug)]
pub struct VPNode<F: Float + Debug + Default> {
    pub children: Vec<Neighbor<F>>,
    pub parent: Option<NodeID>,
    pub center: NodeID,
    pub is_leaf: bool,
}

impl<F: Float + Debug + Default> VPTree<F> {
    pub fn new(ef_construction: usize, space: Distance, dimensions: usize, m: usize) -> VPTree<F> {
        VPTree {
            layers: Vec::new(),
            dimensions,
            swid_layer: Vec::new(),
            vector_layer: Vec::new(),
            ef_construction,
            space,
            m,
            top_node: None,
        }
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) {
        let bottom_id = self.swid_layer.len();
        self.swid_layer.push(swid);
        self.vector_layer.extend_from_slice(q);
        match self.find_closest_leaf(q) {
            Some(closest) => {
                let distance = get_distance(
                    self.vector_layer
                        .chunks(self.dimensions)
                        .nth(self.layers[closest].center)
                        .unwrap(),
                    q,
                    self.space,
                );
                self.layers[closest].children.push(Neighbor {
                    id: bottom_id,
                    distance,
                });
                self.recursive_split(closest);
            }
            None => {
                self.layers.push(VPNode {
                    children: vec![Neighbor {
                        id: bottom_id,
                        distance: F::zero(),
                    }],
                    parent: None,
                    center: bottom_id,
                    is_leaf: true,
                });
                self.top_node = Some(0);
            }
        }
    }
    fn recursive_split(&mut self, id: NodeID) {
        if self.layers[id].children.len() > self.m {
            let new_id = self.layers.len();
            let center = self.layers[id]
                .children
                .iter()
                .max_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let center_id = center.id;
            let center_vector = (match self.layers[id].is_leaf {
                true => self
                    .vector_layer
                    .chunks(self.dimensions)
                    .nth(center_id)
                    .unwrap(),
                false => self.get_vector(center_id),
            })
            .to_vec(); // todo: clone to get around lifetime issues
            let new_node = VPNode {
                children: Vec::with_capacity(self.m + 1),
                parent: self.layers[id].parent,
                center: center_id,
                is_leaf: self.layers[id].is_leaf,
            };
            self.layers.push(new_node);

            let mut i = 0;

            while i < self.layers[id].children.len() {
                let child = self.layers[id].children[i];
                let distance = match self.layers[id].is_leaf {
                    true => get_distance(
                        &center_vector,
                        self.vector_layer
                            .chunks(self.dimensions)
                            .nth(child.id)
                            .unwrap(),
                        self.space,
                    ),
                    false => get_distance(&center_vector, self.get_vector(child.id), self.space),
                };

                if distance < child.distance
                    || (distance == child.distance && self.layers[id].children.len() > self.m / 2)
                {
                    self.layers[new_id].children.push(Neighbor {
                        id: child.id,
                        distance,
                    });
                    if !self.layers[new_id].is_leaf {
                        self.layers[child.id].parent = Some(new_id);
                    }
                    self.layers[id].children.swap_remove(i);
                } else {
                    i += 1;
                }
            }

            if self.layers[id].parent.is_some() {
                let parent_id = self.layers[id].parent.unwrap();
                let parent_center = self.layers[parent_id].center;
                let distance =
                    get_distance(&center_vector, self.get_vector(parent_center), self.space);
                self.layers[parent_id].children.push(Neighbor {
                    id: new_id,
                    distance,
                });
                self.recursive_split(self.layers[id].parent.unwrap());
            } else {
                let new_parent_id = self.layers.len();
                self.layers.push(VPNode {
                    children: vec![
                        Neighbor {
                            id,
                            distance: F::zero(),
                        },
                        Neighbor {
                            id: new_id,
                            distance: get_distance(
                                self.get_vector(id),
                                self.get_vector(new_id),
                                self.space,
                            ),
                        },
                    ],
                    parent: None,
                    center: id,
                    is_leaf: false,
                });
                self.top_node = Some(new_parent_id);
                self.layers[id].parent = Some(new_parent_id);
                self.layers[new_id].parent = Some(new_parent_id);
            }
        }
    }
    fn find_closest_leaf(&self, q: &[F]) -> Option<NodeID> {
        if self.layers.is_empty() {
            return None;
        }
        let mut id = self.top_node.unwrap();
        loop {
            if self.layers[id].is_leaf {
                return Some(id);
            }
            let center = self.layers[id].center;
            id = self.layers[id]
                .children
                .iter()
                .map(|n| Neighbor {
                    id: n.id,
                    distance: get_distance(self.get_vector(n.id), q, self.space),
                })
                .min_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(Neighbor {
                    id: center,
                    distance: F::zero(),
                })
                .id;
        }
    }
    fn get_vector(&self, mut id: NodeID) -> &[F] {
        let mut leaf = self.layers[id].is_leaf;
        loop {
            id = self.layers[id].center;
            if leaf {
                return self.vector_layer.chunks(self.dimensions).nth(id).unwrap();
            }
            leaf = self.layers[id].is_leaf;
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        let mut res: Vec<(Swid, F)> = self
            .knn_ids(q, k)
            .iter()
            .map(|id| {
                (
                    self.swid_layer[*id],
                    get_distance(
                        self.vector_layer.chunks(self.dimensions).nth(*id).unwrap(),
                        q,
                        self.space,
                    ),
                )
            })
            .collect();
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        res.truncate(k);
        res
    }
    fn knn_ids(&self, q: &[F], k: usize) -> Vec<NodeID> {
        let closest = match self.find_closest_leaf(q) {
            Some(closest) => closest,
            None => return Vec::new(),
        };
        let ef = self.ef_construction.max(k);
        let mut id = closest;
        let mut n = self.m;
        loop {
            if n > ef || self.layers[id].parent.is_none() {
                break;
            }
            n *= self.m;
            id = self.layers[id].parent.unwrap();
        }
        self.get_all_children(id, n)
    }

    fn get_all_children(&self, id: NodeID, n: usize) -> Vec<NodeID> {
        let mut result = Vec::with_capacity(n);
        let mut stack = vec![id];

        while let Some(id) = stack.pop() {
            let children = &self.layers[id].children;
            if self.layers[id].is_leaf {
                result.extend(children.iter().map(|child| child.id));
            } else {
                stack.extend(children.iter().map(|child| child.id));
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
        self.layers.iter_mut().for_each(|node| {
            node.children.retain(|n| n.id != id);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_10000() {
        use microbench::*;
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
