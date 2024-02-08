use bit_vec::BitVec;
use num_traits::Float;
use std::collections::BTreeSet;
use std::fmt::Debug;

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
            let mut dot: F = F::zero();
            let mut norm_a: F = F::zero();
            let mut norm_b: F = F::zero();
            for i in 0..a.len() {
                dot = a[i].mul_add(b[i], dot);
                norm_a = a[i].mul_add(a[i], norm_a);
                norm_b = b[i].mul_add(b[i], norm_b);
            }
            F::one() - dot / (norm_a.sqrt() * norm_b.sqrt())
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
            let n = self.search_layer(q, ep, self.ef_construction, lc);
            let qid = self.layers[lc].len();
            for neighbor in &n {
                let new_neighbor = Neighbor {
                    id: qid,
                    distance: neighbor.distance,
                };
                let i = match self.layers[lc][neighbor.id]
                    .neighbors
                    .binary_search(&new_neighbor)
                {
                    Ok(i) => i,
                    Err(i) => i,
                };
                self.layers[lc][neighbor.id]
                    .neighbors
                    .insert(i, new_neighbor);
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
                neighbors: n.into_iter().take(self.m).collect(),
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
        if self.layers[layer].is_empty() {
            return Vec::new();
        }
        let ep_dist = get_distance(self.get_vector(layer, ep), q, self.space);
        let mut visited = BitVec::from_elem(self.layers[layer].len(), false);
        let mut candidates = BTreeSet::new();
        let mut result = Vec::with_capacity(ef);
        visited.set(ep, true);
        candidates.insert(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        result.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });
        let mut max_dist = ep_dist;
        while !candidates.is_empty() {
            let c = candidates.pop_first().unwrap();
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
                    if d_e < max_dist {
                        // slightly faster
                        candidates.insert(Neighbor {
                            id: e.id,
                            distance: d_e,
                        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw() {
        let mut hnsw = HNSW::<f64>::new(16, Distance::Euclidean, 2, 16);
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
        //dbg!(&hnsw);
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
        let mut hnsw = HNSW::<f64>::new(16, Distance::Euclidean, 2, 16);
        let bench_options = Options::default();
        microbench::bench(&bench_options, "test_insert_10000", || {
            for i in 0..10000 {
                hnsw.insert(&[i as f64, i as f64], i);
            }
        });
        microbench::bench(&bench_options, "test_knn_10000", || {
            for i in 0..10000 {
                hnsw.knn(&[i as f64, i as f64], 3);
            }
        });
        print!("{}\n", hnsw.layers[0].len());
        print!("{}\n", hnsw.layers[1].len());
        print!("{}\n", hnsw.layers[2].len());
        print!("{}\n", hnsw.layers[3].len());
        print!("{}\n", hnsw.layers[4].len());
    }
}
