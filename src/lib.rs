use std::{collections::BTreeSet, sync::atomic};
use num::traits::Float;

type Swid = u128;
type NodeID = usize;
const MAX_LAYER: usize = 16;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
struct Neighbor<F:Copy+Default+Ord+PartialOrd+Float+From<f64>>{
    id: NodeID,
    distance: F
}

struct BaseNode <const DIM:usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>,const M:usize>{
    vector:[F;DIM],
    swid: Swid,
}
struct Node <const DIM:usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>,const M:usize>{
    neighbors: [Neighbor<F>;M],
    n_neighbors: u8,
    lower_id: NodeID,
}
pub struct HNSW <const DIM: usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>,const M:usize> {
    layers: [Vec<Node<DIM,F,M>>;MAX_LAYER],
    base_layer: Vec<BaseNode<DIM,F,M>>,
    ef_construction: usize
}

impl <const DIM:usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>,const M:usize> Node<DIM,F,M> {
    fn insert(&mut self, neighbor: Neighbor<F>) {
        //insert new neighbor sorted by distance
        let mut index = 0;
        for i in 0..self.n_neighbors {
            if self.neighbors[i as usize].distance > neighbor.distance {
                index = i;
                break;
            }
        }
        for j in (index..self.n_neighbors).rev() {
            if j as usize>=M-1 {
                continue;
            }
            self.neighbors[j as usize+1] = self.neighbors[j as usize];
        }
        self.neighbors[index as usize] = neighbor;
        self.n_neighbors += 1;
    }
}

fn get_distance<const DIM:usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>>(a: &[F;DIM], b: &[F;DIM]) -> F {
    // euclidean distance
    let mut sum: F = 0.0.into();
    for i in 0..DIM {
        sum = sum + (a[i]-b[i])*(a[i]-b[i]);
    }
    sum.sqrt()
}

// multi-thread rng
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
static mut RNG: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
fn rand_f() -> f64 {
    let rng = unsafe {RNG.fetch_add(1, atomic::Ordering::Relaxed)};
    let mut s = DefaultHasher::new();
    rng.hash(&mut s);
    s.finish() as f64 / u64::MAX as f64
}

impl <const DIM: usize,F:Copy+Default+Ord+PartialOrd+Float+From<f64>,const M:usize> HNSW<DIM,F,M> {
    pub fn new(ef_construction: usize) -> HNSW<DIM,F,M> {
        let layers:[Vec<Node<DIM,F,M>>;MAX_LAYER] = Default::default();
        HNSW {
            layers,
            base_layer: Vec::new(),
            ef_construction,
        }
    }
    pub fn insert(&mut self, q: [F;DIM], swid: Swid) {
        let l = ((-(rand_f()).ln()*(1.0f64/(16.0f64).ln())).floor() as usize).min(MAX_LAYER-1);
        for lc in (0..l).rev() {
            let mut neighbors:[Neighbor<F>;M] = [Neighbor{id:0,distance:F::default()};M];
            let n = self.search_layer(q,0,self.ef_construction,lc);
            for (i,neighbor) in n.iter().enumerate() {
                if i<M {
                    neighbors[i] = *neighbor;
                } else {
                    break;
                }
            }
            let qid = self.layers[lc].len();
            self.layers[lc].push(Node{
                neighbors,
                n_neighbors:neighbors.len() as u8,
                lower_id: if lc==0 {self.base_layer.len()} else {qid} // todo: check if this is correct (off by one?)
            });
            for neighbor in neighbors {
                self.layers[lc][neighbor.id].insert(Neighbor{
                    id:qid,
                    distance: neighbor.distance
                });
            }
        }
        self.base_layer.push(BaseNode{
            vector: q,
            swid,
        });
    }
    pub fn remove(&mut self, swid: Swid) {
        let mut new_hnsw = HNSW::new(self.ef_construction);
        for node in &self.base_layer {
            if node.swid != swid {
                new_hnsw.insert(node.vector, node.swid);
            }
        }
        self.layers = new_hnsw.layers;
        self.base_layer = new_hnsw.base_layer;
    }
    fn search_layer(&self, q: [F;DIM],ep:usize, ef: usize, layer: usize) -> Vec<Neighbor<F>> {
        let ep_dist = get_distance(&self.get_base(layer,ep).vector,&q);
        let mut visited = BTreeSet::new();
        let mut candidates = BTreeSet::new();
        let mut result = BTreeSet::new();
        visited.insert(ep);
        candidates.insert(Neighbor{
            id: ep,
            distance: ep_dist
        });
        result.insert(Neighbor{
            id: ep,
            distance: ep_dist
        });
        while !candidates.is_empty() {
            let c = candidates.pop_first().unwrap();
            let f = result.last().unwrap();
            if c.distance > f.distance {
                break;
            }
            for e in self.layers[layer][c.id].neighbors.iter(){
                if visited.contains(&e.id) {
                    continue;
                }
                visited.insert(e.id);
                let f = result.last().unwrap();
                let d_e = get_distance(&self.get_base(layer,e.id).vector,&q);
                if d_e< f.distance || result.len() < ef{
                    candidates.insert(Neighbor{
                        id: e.id,
                        distance: d_e
                    });
                    result.insert(Neighbor{
                        id: e.id,
                        distance: d_e
                    });
                }
            }
        }
        result.into_iter().collect()
    }
    fn get_base(&self, layer: usize, id: NodeID) -> &BaseNode<DIM,F,M> {
        if layer == 0 {
            &self.base_layer[id]
        } else {
            self.get_base(layer-1,self.layers[layer][id].lower_id)
        }
    }
    pub fn knn(&self, q: [F;DIM], k: usize) -> Vec<Swid> {
        let ef_search = self.ef_construction.max(k);
        let mut w = BTreeSet::new();
        let mut top_layer = MAX_LAYER-1;
        while self.layers[top_layer].is_empty() {
            top_layer -= 1;
        }
        let mut ep = 0;
        for lc in (1..=top_layer).rev() {
            let neighbors = self.search_layer(q,ep,1,lc);
            let neighbor = neighbors.first().unwrap();
            ep = self.layers[lc][neighbor.id].lower_id;
        }
        for neighbor in self.search_layer(q,ep,ef_search,0) {
            w.insert(neighbor);
        }

        let mut result = Vec::new();
        for neighbor in w {
            result.push(self.get_base(top_layer,neighbor.id).swid);
        }
        result
    }
}
