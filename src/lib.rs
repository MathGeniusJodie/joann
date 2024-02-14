use num_traits::Float;
use std::fmt::Debug;
type NodeID = usize;

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

#[derive(Debug)]
pub struct VPTree<F: Float + Debug + Default> {
    pub nodes: Vec<Node<F>>,
    pub dimensions: usize,
    pub vector_layer: Vec<F>,
    ef_construction: usize,
    space: Distance,
    m: usize,
    top_node: Option<NodeID>,
}
#[derive(Copy, Clone, Debug, Default)]
pub struct Child<F: Float + Debug + Default> {
    distance: F,
    vector_id: NodeID,
    id: Option<NodeID>,
}
#[derive(Debug)]
pub struct Node<F: Float + Debug + Default> {
    pub children: Vec<Child<F>>,
    pub parent: Option<NodeID>,
}
impl<F: Float + Debug + Default> Node<F> {
    pub fn is_leaf(&self) -> bool {
        self.children.first().unwrap().id.is_none()
    }
}

impl<F: Float + Debug + Default> VPTree<F> {
    pub fn new(ef_construction: usize, space: Distance, dimensions: usize, m: usize) -> VPTree<F> {
        VPTree {
            nodes: Vec::new(),
            dimensions,
            vector_layer: Vec::new(),
            ef_construction,
            space,
            m,
            top_node: None,
        }
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
    pub fn insert(&mut self, q: &[F]) {
        let vector_id = self.vector_layer.len() / self.dimensions;
        self.vector_layer.extend_from_slice(q);
        let closest_leaf = self.get_closest_leaf(q);
        match closest_leaf {
            Some(leaf_id) => {
                self.push_child_sorted(leaf_id, vector_id, None);
                self.split(leaf_id);
            }
            None => {
                self.top_node = Some(0);
                self.nodes.push(Node {
                    children: Vec::with_capacity(self.m + 1),
                    parent: None,
                });
                self.push_child(0, vector_id, None);
            }
        }
    }
    fn split(&mut self, id: NodeID) {
        let node = &mut self.nodes[id];
        if node.children.len() <= self.m {
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
                || (distance == child.distance && new_node.children.len() < self.m / 2)
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
                children: Vec::with_capacity(self.m + 1),
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

    pub fn knn(&self, q: &[F], k: usize) -> Vec<(&[F], F)> {
        let closest_leaf = self.get_closest_leaf(q);
        if closest_leaf.is_none() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.ef_construction);
        let mut n = self.m;
        let mut id = closest_leaf.unwrap();
        loop {
            if n >= self.ef_construction {
                break;
            }
            if self.nodes[id].parent.is_none() {
                break;
            }
            id = self.nodes[id].parent.unwrap();
            n *= self.m;
        }
        let mut stack = vec![id];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if node.is_leaf() {
                for child in &node.children {
                    let distance = get_distance(q, self.get_vector(child.vector_id), self.space);
                    result.push((self.get_vector(child.vector_id), distance));
                }
            } else {
                for child in &node.children {
                    stack.push(child.id.unwrap());
                }
            }
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        result
    }
    pub fn remove(&mut self, vector_to_remove: &[F]) {
        let mut new_vp_tree = Self {
            nodes: Vec::new(),
            dimensions: self.dimensions,
            vector_layer: Vec::new(),
            ef_construction: self.ef_construction,
            space: self.space,
            m: self.m,
            top_node: None,
        };
        self.vector_layer.chunks(self.dimensions).for_each(|v| {
            if v != vector_to_remove {
                new_vp_tree.insert(v);
            }
        });
        *self = new_vp_tree;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp_tree() {
        let mut vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 4);
        vptree.insert(&[3.0, 3.0]);
        vptree.insert(&[4.0, 4.0]);
        vptree.insert(&[5.0, 5.0]);
        vptree.insert(&[6.0, 6.0]);
        vptree.insert(&[7.0, 7.0]);
        vptree.insert(&[8.0, 8.0]);
        vptree.insert(&[9.0, 9.0]);
        vptree.insert(&[0.0, 0.0]);
        vptree.insert(&[1.0, 1.0]);
        vptree.insert(&[2.0, 2.0]);
        //dbg!(&vptree);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (vec![0.0, 0.0].as_slice(), 0.0),
                (vec![1.0, 1.0].as_slice(), std::f64::consts::SQRT_2),
                (vec![2.0, 2.0].as_slice(), 2.0 * std::f64::consts::SQRT_2)
            ]
        );
        vptree.remove(&[0.0, 0.0]);
        vptree.remove(&[3.0, 3.0]);
        assert_eq!(
            vptree.knn(&[0.0, 0.0], 3),
            vec![
                (vec![1.0, 1.0].as_slice(), std::f64::consts::SQRT_2),
                (vec![2.0, 2.0].as_slice(), 2.0 * std::f64::consts::SQRT_2),
                (vec![4.0, 4.0].as_slice(), 4.0 * std::f64::consts::SQRT_2)
            ]
        );
        //dbg!(&vptree);
    }

    #[test]
    fn test_10000() {
        use microbench::*;
        let mut vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 4);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "vp_tree_test_insert_10000", || {
            for i in 0..10000 {
                vptree.insert(&[i as f64, i as f64]);
            }
            vptree = VPTree::<f64>::new(16, Distance::Euclidean, 2, 4);
        });
        for i in 0..10000 {
            vptree.insert(&[i as f64, i as f64]);
        }
        microbench::bench(&bench_options, "vp_tree_test_knn_10000", || {
            for i in 0..10000 {
                vptree.knn(&[i as f64, i as f64], 3);
            }
        });
    }
}
