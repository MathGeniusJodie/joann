use num_traits::Float;
use std::{collections::BTreeSet, fmt::Debug};

mod forever_vec;

type Swid = u128;
type NodeID = usize;
const MAX_LAYER: usize = 16;

#[derive(Copy, Clone, Debug, Default)]
struct Neighbor<F: Float + Debug + Default> {
    id: NodeID,
    distance: F,
}
impl<F: Float + Debug + Default> PartialEq for Neighbor<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
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
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Debug)]
pub struct HNSW<F: Float + Debug + Default, const M: usize> {
    layers: [Vec<Node<F, M>>; MAX_LAYER],
    dimensions: usize,
    //base_layer: Vec<BaseNode<DIM, F>>,
    swid_layer: Vec<Swid>,
    vector_layer: Vec<F>,
    ef_construction: usize,
    space: Distance,
}

#[derive(Debug)]
struct Node<F: Float + Debug + Default, const M: usize> {
    neighbors: [Neighbor<F>; M],
    n_neighbors: usize,
    lower_id: NodeID,
}
impl<F: Float + Debug + Default, const M: usize> Node<F, M> {
    fn insert_neighbor(&mut self, neighbor: Neighbor<F>) {
        if let Err(i) = self.neighbors[..self.n_neighbors as usize].binary_search(&neighbor) {
            if i < M {
                self.n_neighbors = (self.n_neighbors + 1).min(M);
                self.neighbors[i..self.n_neighbors as usize].rotate_right(1);
                self.neighbors[i] = neighbor;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Euclidean,
    Cosine,
    L2,
    IP,
}

fn get_distance<F: Float + Debug + Default>(
    a: &[F],
    b: &[F],
    space: Distance,
) -> F {
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
            let mut dot: F = F::zero();
            let mut norm_a: F = F::zero();
            let mut norm_b: F = F::zero();
            for i in 0..a.len() {
                dot = a[i].mul_add(b[i], dot);
                norm_a = a[i].mul_add(a[i], norm_a);
                norm_b = b[i].mul_add(b[i], norm_b);
            }
            dot / (norm_a.sqrt() * norm_b.sqrt())
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

/// Returns a number in the range [0, 1)
#[inline]
fn rand_f() -> f64 {
    rand::random::<f64>()
}

impl<F: Float + Debug + Default, const M: usize> HNSW<F, M> {
    pub fn new(ef_construction: usize, space: Distance, dimensions:usize) -> HNSW<F, M> {
        let layers: [Vec<Node<F, M>>; MAX_LAYER] = Default::default();
        HNSW {
            layers,
            dimensions,
            //base_layer: Vec::new(),
            swid_layer: Vec::new(),
            vector_layer: Vec::new(),
            ef_construction,
            space,
        }
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) {
        let l = ((-rand_f().ln() * (1.0f64 / 16.0f64.ln())) as usize).min(MAX_LAYER - 1);
        let mut ep = 0;
        for lc in (l..MAX_LAYER).rev() {
            ep = match self.search_layer(q, ep, 1, lc).first() {
                Some(n) => self.layers[lc][n.id].lower_id,
                None => 0,
            };
        }

        for lc in (0..=l).rev() {
            let mut n = self.search_layer(q, ep, self.ef_construction, lc);
            let nl = n.len();
            let qid = self.layers[lc].len();
            for neighbor in &n {
                self.layers[lc][neighbor.id].insert_neighbor(Neighbor {
                    id: qid,
                    distance: neighbor.distance,
                });
            }
            n.resize(M, Neighbor::default());
            self.layers[lc].push(Node {
                neighbors: n.try_into().unwrap(),
                n_neighbors: (nl).min(M),
                lower_id: if lc == 0 { self.swid_layer.len() } else { qid },
            });
        }
        self.swid_layer.push(swid);
        self.vector_layer.extend_from_slice(&q);
    }
    /*
    pub fn remove(&mut self, swid: Swid) {
        let mut new_hnsw = HNSW::new(self.ef_construction, self.space);
        for node in &self.base_layer {
            if node.swid != swid {
                new_hnsw.insert(node.vector, node.swid);
            }
        }
        self.layers = new_hnsw.layers;
        self.base_layer = new_hnsw.base_layer;
    }*/
    fn search_layer(&self, q: &[F], ep: usize, ef: usize, layer: usize) -> Vec<Neighbor<F>> {
        if self.layers[layer].is_empty() {
            return Vec::new();
        }
        let ep_dist = get_distance(self.get_vector(layer, ep), &q, self.space);
        let mut visited = BTreeSet::new();
        let mut candidates = BTreeSet::new();
        let mut result = BTreeSet::new();
        visited.insert(ep);
        candidates.insert(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        result.insert(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        while !candidates.is_empty() {
            let c = candidates.pop_first().unwrap();
            let f = result.last().unwrap();
            if c.distance > f.distance {
                break;
            }
            for i in 0..self.layers[layer][c.id].n_neighbors {
                let e = self.layers[layer][c.id].neighbors[i];
                if visited.contains(&e.id) {
                    continue;
                }
                visited.insert(e.id);
                let f = result.last().unwrap();
                let d_e = get_distance(self.get_vector(layer, e.id), &q, self.space);
                if d_e < f.distance || result.len() < ef {
                    candidates.insert(Neighbor {
                        id: e.id,
                        distance: d_e,
                    });
                    result.insert(Neighbor {
                        id: e.id,
                        distance: d_e,
                    });
                }
            }
        }
        result.into_iter().collect()
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
            &self.vector_layer[lower*self.dimensions..(lower+1)*self.dimensions]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw() {
        let mut hnsw = HNSW::<f64, 16>::new(16, Distance::Euclidean, 2);
        hnsw.insert(&[0.0, 0.0], 0);
        hnsw.insert(&[1.0, 1.0], 1);
        hnsw.insert(&[2.0, 2.0], 2);
        hnsw.insert(&[3.0, 3.0], 3);
        hnsw.insert(&[4.0, 4.0], 4);
        hnsw.insert(&[5.0, 5.0], 5);
        hnsw.insert(&[6.0, 6.0], 6);
        hnsw.insert(&[7.0, 7.0], 7);
        hnsw.insert(&[8.0, 8.0], 8);
        hnsw.insert(&[9.0, 9.0], 9);
        assert_eq!(
            hnsw.knn(&[0.0, 0.0], 3),
            vec![
                (0, 0.0),
                (1, std::f64::consts::SQRT_2),
                (2, 2.0 * std::f64::consts::SQRT_2)
            ]
        );
    }

    #[test]
    fn test_insert_10000() {
        use microbench::*;
        let mut hnsw = HNSW::<f64, 16>::new(16, Distance::Euclidean,2);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "test_insert_10000", || {
            for i in 0..10000 {
                hnsw.insert(&[i as f64, i as f64], i);
            }
        });
    }
}
