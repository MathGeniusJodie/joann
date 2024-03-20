use memmap2::{Mmap, MmapMut};
use num_traits::Float;
use std::{
    collections::HashMap,
    fmt::Debug,
    fs::{File, OpenOptions},
    iter::Sum,
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
fn get_distance<F: Float + Debug + Default + Sum>(
    a: &[F],
    b: &[F],
    aa: F,
    bb: F,
    space: Distance,
) -> F {
    #[cfg(target_feature = "avx")]
    const STRIDE: usize = 8;
    #[cfg(not(target_feature = "avx"))]
    const STRIDE: usize = 4;
    let chunks_a = a.chunks_exact(STRIDE);
    let chunks_b = b.chunks_exact(STRIDE);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    let mut dot = [F::zero(); STRIDE];
    for (a, b) in chunks_a.zip(chunks_b) {
        for i in 0..STRIDE {
            dot[i] = a[i].mul_add(b[i], dot[i]);
        }
    }
    let mut dot = dot.into_iter().sum();
    for i in 0..rem_a.len() {
        dot = rem_a[i].mul_add(rem_b[i], dot);
    }
    match space {
        Distance::Cosine => {
            //handle 0 vectors
            if aa * bb <= F::zero() {
                return F::zero();
            }
            F::one() - dot / (aa * bb).sqrt()
        }
        Distance::Euclidean => (aa + bb - F::from(2).unwrap() * dot).sqrt(),
        Distance::L2 => aa + bb - F::from(2).unwrap() * dot,
        Distance::IP => dot,
    }
}
fn get_length_2<F: Float + Debug + Default + Sum>(a: &[F]) -> F {
    #[cfg(target_feature = "avx")]
    const STRIDE: usize = 8;
    #[cfg(not(target_feature = "avx"))]
    const STRIDE: usize = 4;
    let chunks_a = a.chunks_exact(STRIDE);
    let rem_a = chunks_a.remainder();
    let mut aa = [F::zero(); STRIDE];
    for a in chunks_a {
        for i in 0..STRIDE {
            aa[i] = a[i].mul_add(a[i], aa[i]);
        }
    }
    let mut aa = aa.iter().fold(F::zero(), |acc, &x| acc + x);
    for i in 0..rem_a.len() {
        aa = rem_a[i].mul_add(rem_a[i], aa);
    }
    aa
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
                // unmap so we can resize the file in windows
                #[cfg(target_os = "windows")]
                {   
                    *mmap = MmapMut::map_anon(0).unwrap();
                }
                file.set_len(bytes as u64).unwrap();
                *mmap = unsafe { MmapMut::map_mut(file).unwrap() };
            }
            Store::Vec(ref mut vec) => {
                vec.resize(new_len, T::default());
            }
        };
    }
}

const MAX_LAYER: usize = 16;

#[derive(Copy, Clone, Debug, Default)]
struct Neighbor<F: Float + Debug + Default + Sum> {
    id: NodeID,
    distance: F,
}
impl<F: Float + Debug + Default + Sum> PartialEq for Neighbor<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl<F: Float + Debug + Default + Sum> Eq for Neighbor<F> {}
impl<F: Float + Debug + Default + Sum> PartialOrd for Neighbor<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<F: Float + Debug + Default + Sum> Ord for Neighbor<F> {
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
pub struct Index<F: Float + Debug + Default + Sum> {
    pub layers: [Vec<Node<F>>; MAX_LAYER],
    pub dimensions: usize,
    pub swid_layer: Store<Swid>,
    pub vector_layer: Store<F>,
    pub length_2_layer: Store<F>,
    pub ef_construction: usize,
    pub space: Distance,
    pub m: usize,
    pub swid_to_id: HashMap<Swid, NodeID>,
}

#[derive(Debug)]
pub struct Node<F: Float + Debug + Default + Sum> {
    neighbors: Vec<Neighbor<F>>,
    lower_id: NodeID,
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

impl<F: Float + Debug + Default + Sum> Index<F> {
    pub fn new(ef_construction: usize, space: Distance, dimensions: usize, m: usize) -> Index<F> {
        let layers: [Vec<Node<F>>; MAX_LAYER] = Default::default();
        Index {
            layers,
            dimensions,
            swid_layer: Store::Vec(Vec::new()),
            vector_layer: Store::Vec(Vec::new()),
            length_2_layer: Store::Vec(Vec::new()),
            ef_construction,
            space,
            m,
            swid_to_id: HashMap::new(),
        }
    }
    pub fn new_with_store(
        ef_construction: usize,
        space: Distance,
        dimensions: usize,
        m: usize,
        vector_store: &Path,
        swid_store: &Path,
    ) -> Index<F> {
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

        let mut index = Index {
            layers: Default::default(),
            dimensions,
            swid_layer: Store::Mmap((swid_file, swid_mmap)),
            vector_layer: Store::Mmap((vector_file, vector_mmap)),
            length_2_layer: Store::Vec(Vec::new()),
            ef_construction,
            space,
            m,
            swid_to_id: HashMap::new(),
        };
        for i in 0..index.swid_layer.slice().len() {
            let q = index.vector_layer.slice()[i * index.dimensions..(i + 1) * index.dimensions]
                .to_owned();
            index.index(&q, i);
        }
        index
    }
    pub fn insert(&mut self, q: &[F], swid: Swid) -> Result<(), ()> {
        if self.swid_to_id.contains_key(&swid) {
            return Err(());
        }
        let id = self.swid_layer.slice().len();
        self.swid_layer.resize(1);
        self.swid_layer.slice_mut()[id] = swid;
        self.vector_layer.resize(self.dimensions as isize);
        self.vector_layer.slice_mut()[(id * self.dimensions)..((id + 1) * self.dimensions)]
            .copy_from_slice(q);
        self.index(q, id);
        Ok(())
    }
    fn index(&mut self, q: &[F], id: NodeID) {
        self.swid_to_id.insert(self.swid_layer.slice()[id], id);
        self.length_2_layer.resize(1);
        self.length_2_layer.slice_mut()[id] = get_length_2(q);
        let l = ((-rand::random::<f64>().ln() * (1.0f64 / (self.m as f64).ln())) as usize)
            .min(MAX_LAYER - 1);
        let mut ep = 0;
        for lc in (l + 1..MAX_LAYER).rev() {
            ep = match self.search_layer(q, ep, 1, lc).first() {
                Some(n) => self.layers[lc][n.id].lower_id,
                None => 0,
            };
        }

        for lc in (0..=l).rev() {
            let mut n = self.search_layer(q, ep, self.ef_construction, lc);
            n.truncate(if lc == 0 { self.m * 2 } else { self.m });
            let qid = self.layers[lc].len();
            for neighbor in &n {
                self.layers[lc][neighbor.id].neighbors.push(Neighbor {
                    id: qid,
                    distance: neighbor.distance,
                });
                self.layers[lc][neighbor.id].neighbors.sort();
                self.layers[lc][neighbor.id].neighbors.truncate(if lc == 0 {
                    self.m * 2
                } else {
                    self.m
                });
            }
            let lower_id = if lc == 0 {
                id
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
    }
    pub fn remove(&mut self, swid_to_remove: Swid) {
        let last = self.swid_layer.slice().len() - 1;
        let mut last_vector = self.vector_layer.slice()[last * self.dimensions..].to_owned();
        let id_to_remove = *self.swid_to_id.get(&swid_to_remove).unwrap();
        self.swid_layer.slice_mut().swap(id_to_remove, last);
        self.length_2_layer.slice_mut().swap(id_to_remove, last);
        self.vector_layer.slice_mut()
            [id_to_remove * self.dimensions..(id_to_remove + 1) * self.dimensions]
            .swap_with_slice(&mut last_vector);
        self.vector_layer.resize(self.dimensions as isize * -1);
        self.swid_layer.resize(-1);
        self.length_2_layer.resize(-1);
        self.layers.iter_mut().for_each(|layer| layer.clear());
        for i in 0..last {
            let q = self.vector_layer.slice()[i * self.dimensions..(i + 1) * self.dimensions]
                .to_owned();
            self.index(&q, i);
        }
        self.swid_to_id.remove(&swid_to_remove);
    }
    fn search_layer(&self, q: &[F], ep: usize, ef: usize, layer: usize) -> Vec<Neighbor<F>> {
        let qq = get_length_2(q);
        if ef >= self.layers[layer].len() {
            let len = self.layers[layer].len();
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                result.push(Neighbor {
                    id: i,
                    distance: get_distance(
                        self.get_vector(layer, i),
                        q,
                        self.get_length_2(layer, i),
                        qq,
                        self.space,
                    ),
                });
            }
            result.sort();
            return result;
        }
        let ep_dist = get_distance(
            self.get_vector(layer, ep),
            q,
            self.get_length_2(layer, ep),
            qq,
            self.space,
        );
        let mut visited = smallbitvec::SmallBitVec::from_elem(self.layers[layer].len(), false);
        let mut candidates = Vec::new();
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

                let d_e = get_distance(
                    self.get_vector(layer, e.id),
                    q,
                    self.get_length_2(layer, e.id),
                    qq,
                    self.space,
                );
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
                        pop_max(&mut result);
                        max_dist = result.iter().max().unwrap().distance;
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
            self.swid_layer.slice()[lower]
        } else {
            self.get_swid(layer - 1, lower)
        }
    }
    fn get_vector(&self, layer: usize, id: NodeID) -> &[F] {
        let lower = self.layers[layer][id].lower_id;
        if layer == 0 {
            self.vector_layer
                .slice()
                .chunks(self.dimensions)
                .nth(lower)
                .unwrap()
        } else {
            self.get_vector(layer - 1, lower)
        }
    }
    fn get_length_2(&self, layer: usize, id: NodeID) -> F {
        let lower = self.layers[layer][id].lower_id;
        if layer == 0 {
            self.length_2_layer.slice()[lower]
        } else {
            self.get_length_2(layer - 1, lower)
        }
    }
    pub fn knn(&self, q: &[F], k: usize) -> Vec<(Swid, F)> {
        let ef_search = k;
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
    use num_traits::abs;
    use rand::Rng;

    #[test]
    fn test_tree() {
        let mut tree = Index::<f32>::new(32, Distance::Euclidean, 2, 16);
        tree.insert(&[3.0, 3.0], 4).unwrap();
        tree.insert(&[4.0, 4.0], 352).unwrap();
        tree.insert(&[5.0, 5.0], 43).unwrap();
        tree.insert(&[6.0, 6.0], 41).unwrap();
        tree.insert(&[7.0, 7.0], 35).unwrap();
        tree.insert(&[8.0, 8.0], 52).unwrap();
        tree.insert(&[9.0, 9.0], 42).unwrap();
        tree.insert(&[0.0, 0.0], 32).unwrap();
        tree.insert(&[1.0, 1.0], 222).unwrap();
        tree.insert(&[2.0, 2.0], 567).unwrap();
        //dbg!(&tree);
        assert_eq!(
            tree.knn(&[0.0, 0.0], 3),
            vec![
                (32, 0.0),
                (222, std::f32::consts::SQRT_2),
                (567, 2.0 * std::f32::consts::SQRT_2)
            ]
        );
        tree.remove(32);
        tree.remove(4);
        //dbg!(&tree);
        assert_eq!(
            tree.knn(&[0.0, 0.0], 3),
            vec![
                (222, std::f32::consts::SQRT_2),
                (567, 2.0 * std::f32::consts::SQRT_2),
                (352, 4.0 * std::f32::consts::SQRT_2)
            ]
        );
    }
    const BENCH_DIMENSIONS: usize = 300;
    const LINEAR_SEARCH_SIZE: usize = 10000;
    const LINEAR_SEARCH_TOPK: usize = 50;
    #[test]
    fn test_speed() {
        use microbench::*;
        let mut tree = Index::<f32>::new(200, Distance::Euclidean, BENCH_DIMENSIONS, 32);

        let mut rng = rand::thread_rng();
        let mut vectors = Vec::with_capacity(LINEAR_SEARCH_SIZE);
        vectors.resize_with(LINEAR_SEARCH_SIZE, || {
            let mut vector: Vec<f32> = Vec::with_capacity(BENCH_DIMENSIONS);
            vector.resize_with(BENCH_DIMENSIONS, || rng.gen::<f32>());
            vector
        });

        let bench_options = Options::default();
        microbench::bench(&bench_options, "insert", || {
            vectors.iter().enumerate().for_each(|(i, vector)| {
                tree.insert(&vector, i as Swid).unwrap();
            });
            tree = Index::<f32>::new(200, Distance::Euclidean, BENCH_DIMENSIONS, 32);
        });
        vectors.iter().enumerate().for_each(|(i, vector)| {
            tree.insert(&vector, i as Swid).unwrap();
        });
        microbench::bench(&bench_options, "knn_topk1", || {
            for i in 0..LINEAR_SEARCH_SIZE {
                let vector = vectors[i].as_slice();
                tree.knn(&vector, 1);
            }
        });
        microbench::bench(&bench_options, "knn_topk10", || {
            for i in 0..LINEAR_SEARCH_SIZE {
                let vector = vectors[i].as_slice();
                tree.knn(&vector, 10);
            }
        });
        microbench::bench(&bench_options, "knn_topk100", || {
            for i in 0..LINEAR_SEARCH_SIZE {
                let vector = vectors[i].as_slice();
                tree.knn(&vector, 100);
            }
        });
    }
    #[test]
    /*
    fn repeatedly_test_vs_linear_search() {
        for _ in 0..1000 {
            test_vs_linear_search();
        }
    }*/
    fn test_vs_linear_search() {
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::with_capacity(LINEAR_SEARCH_SIZE);
        vectors.resize_with(LINEAR_SEARCH_SIZE, || {
            let mut vector: Vec<f32> = Vec::with_capacity(BENCH_DIMENSIONS);
            vector.resize_with(BENCH_DIMENSIONS, || rng.gen::<f32>());
            vector
        });

        //build a tree
        let mut tree = Index::<f32>::new(200, Distance::Euclidean, BENCH_DIMENSIONS, 32);
        for (i, vector) in vectors.iter().enumerate() {
            tree.insert(vector, i as u128).unwrap();
        }

        //get random vector for sampling
        let random_vector = &vectors[0];

        //topk LINEAR_SEARCH_TOPK
        let mut topk = tree.knn(random_vector, LINEAR_SEARCH_TOPK*2);
        topk.truncate(LINEAR_SEARCH_TOPK);

        //linear search topk LINEAR_SEARCH_TOPK
        let mut linear_search_topk = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            let distance = get_distance(
                random_vector,
                vector,
                get_length_2(&random_vector),
                get_length_2(&vector),
                Distance::Euclidean,
            );
            linear_search_topk.push((i as u128, distance));
        }
        linear_search_topk.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut differences = Vec::new();

        //compare and see if the topk is the same, if they aren't, print the index they differ
        for (a, b) in topk.iter().zip(linear_search_topk.iter()) {
            if !topk.contains(b) {
                differences.push(abs(a.1 - b.1));
            }
        }

        //average distance of topk
        let recall = 1.0 - differences.len() as f64 / LINEAR_SEARCH_TOPK as f64;
        println!("recall: {} ", recall);
    }
}
