// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Radix-tree prefix cache over token ids -> KV slots, with reference counting
//! and LRU eviction of unreferenced leaves (the SGLang / FreeToken design).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub type NodeId = usize;
pub const ROOT: NodeId = 0;

#[derive(Debug, Default)]
struct Node {
    key: Vec<u32>,
    value: Vec<u32>,
    children: HashMap<u32, NodeId>,
    parent: Option<NodeId>,
    ref_count: u32,
    last_access: u64,
    alive: bool,
    /// Recurrent-state snapshot taken after this node's last token (hybrid models). A snapshot
    /// cannot be split: it stays on the suffix node, whose end boundary is unchanged.
    snapshot: Option<u32>,
}

#[derive(Debug)]
pub struct RadixCache {
    nodes: Vec<Node>,
    free_ids: Vec<NodeId>,
    evictable: usize,
    protected: usize,
    tick: u64,
    /// Snapshot slots released by evicted nodes, drained by `take_freed_snapshots`.
    freed_snapshots: Vec<u32>,
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl RadixCache {
    pub fn new() -> Self {
        let root = Node {
            ref_count: 1,
            alive: true,
            ..Default::default()
        };
        Self {
            nodes: vec![root],
            free_ids: vec![],
            evictable: 0,
            protected: 0,
            tick: 0,
            freed_snapshots: vec![],
        }
    }

    /// Tokens whose slots can be reclaimed by `evict`.
    pub fn evictable_size(&self) -> usize {
        self.evictable
    }

    /// Tokens held by in-flight requests.
    pub fn protected_size(&self) -> usize {
        self.protected
    }

    pub fn cached_tokens(&self) -> usize {
        self.evictable + self.protected
    }

    fn new_node(&mut self, node: Node) -> NodeId {
        if let Some(id) = self.free_ids.pop() {
            self.nodes[id] = node;
            id
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Split `id` so its first `pos` tokens become a new parent node. Returns the new node.
    fn split(&mut self, id: NodeId, pos: usize) -> NodeId {
        let parent = self.nodes[id].parent.expect("cannot split root");
        let (key_head, value_head) = {
            let n = &mut self.nodes[id];
            let kt = n.key.split_off(pos);
            let vt = n.value.split_off(pos);
            let kh = std::mem::replace(&mut n.key, kt);
            let vh = std::mem::replace(&mut n.value, vt);
            (kh, vh)
        };
        let first = key_head[0];
        let suffix_first = self.nodes[id].key[0];
        let new_id = self.new_node(Node {
            key: key_head,
            value: value_head,
            children: HashMap::from([(suffix_first, id)]),
            parent: Some(parent),
            ref_count: self.nodes[id].ref_count,
            last_access: self.nodes[id].last_access,
            alive: true,
            snapshot: None,
        });
        self.nodes[id].parent = Some(new_id);
        self.nodes[parent].children.insert(first, new_id);
        new_id
    }

    /// Longest cached prefix of `key`. Returns the node ending exactly at the match and the
    /// slots of the matched tokens.
    pub fn match_prefix(&mut self, key: &[u32]) -> (NodeId, Vec<u32>) {
        let now = self.touch();
        let mut node = ROOT;
        let mut pos = 0;
        let mut slots = Vec::new();
        while pos < key.len() {
            let Some(&child) = self.nodes[node].children.get(&key[pos]) else {
                break;
            };
            let m = common_prefix(&self.nodes[child].key, &key[pos..]);
            let child = if m < self.nodes[child].key.len() {
                self.split(child, m)
            } else {
                child
            };
            self.nodes[child].last_access = now;
            slots.extend_from_slice(&self.nodes[child].value);
            node = child;
            pos += m;
        }
        (node, slots)
    }

    /// Insert `key -> value`. Returns how many leading tokens were already cached; the caller
    /// owns (and should free) the duplicate slots `value[already_cached_by_caller..returned]`.
    pub fn insert(&mut self, key: &[u32], value: &[u32]) -> usize {
        assert_eq!(key.len(), value.len());
        let now = self.touch();
        let mut node = ROOT;
        let mut pos = 0;
        while pos < key.len() {
            match self.nodes[node].children.get(&key[pos]).copied() {
                Some(child) => {
                    let m = common_prefix(&self.nodes[child].key, &key[pos..]);
                    let child = if m < self.nodes[child].key.len() {
                        self.split(child, m)
                    } else {
                        child
                    };
                    self.nodes[child].last_access = now;
                    node = child;
                    pos += m;
                }
                None => {
                    let len = key.len() - pos;
                    let id = self.new_node(Node {
                        key: key[pos..].to_vec(),
                        value: value[pos..].to_vec(),
                        children: HashMap::new(),
                        parent: Some(node),
                        ref_count: 0,
                        last_access: now,
                        alive: true,
                        snapshot: None,
                    });
                    self.nodes[node].children.insert(key[pos], id);
                    self.evictable += len;
                    return pos;
                }
            }
        }
        pos
    }

    pub fn lock(&mut self, mut node: NodeId) {
        while node != ROOT {
            let n = &mut self.nodes[node];
            if n.ref_count == 0 {
                self.evictable -= n.key.len();
                self.protected += n.key.len();
            }
            n.ref_count += 1;
            node = n.parent.unwrap();
        }
    }

    pub fn unlock(&mut self, mut node: NodeId) {
        while node != ROOT {
            let n = &mut self.nodes[node];
            assert!(n.ref_count > 0, "radix unlock underflow");
            n.ref_count -= 1;
            if n.ref_count == 0 {
                self.evictable += n.key.len();
                self.protected -= n.key.len();
            }
            node = n.parent.unwrap();
        }
    }

    /// Evict least-recently-used unreferenced leaves until at least `n` slots are freed
    /// (or nothing evictable is left). Returns the freed slots.
    pub fn evict(&mut self, n: usize) -> Vec<u32> {
        let mut heap: BinaryHeap<Reverse<(u64, NodeId)>> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(id, x)| *id != ROOT && x.alive && x.children.is_empty() && x.ref_count == 0)
            .map(|(id, x)| Reverse((x.last_access, id)))
            .collect();
        let mut freed = Vec::new();
        while freed.len() < n {
            let Some(Reverse((_, id))) = heap.pop() else {
                break;
            };
            let node = std::mem::take(&mut self.nodes[id]);
            self.evictable -= node.key.len();
            self.freed_snapshots.extend(node.snapshot);
            freed.extend(node.value);
            let parent = node.parent.unwrap();
            self.nodes[parent].children.remove(&node.key[0]);
            self.free_ids.push(id);
            let p = &self.nodes[parent];
            if parent != ROOT && p.children.is_empty() && p.ref_count == 0 {
                heap.push(Reverse((p.last_access, parent)));
            }
        }
        freed
    }

    /// Attach a state snapshot at the end of `node`; returns a snapshot it replaced.
    pub fn set_snapshot(&mut self, node: NodeId, slot: u32) -> Option<u32> {
        assert_ne!(node, ROOT, "the root has no end boundary");
        self.nodes[node].snapshot.replace(slot)
    }

    pub fn snapshot(&self, node: NodeId) -> Option<u32> {
        self.nodes[node].snapshot
    }

    /// Nearest node on the path from `node` to the root that carries a snapshot:
    /// `(node, prefix length it ends at, snapshot slot)`.
    pub fn snapshot_ancestor(&self, mut node: NodeId) -> Option<(NodeId, usize, u32)> {
        while node != ROOT {
            if let Some(s) = self.nodes[node].snapshot {
                return Some((node, self.depth(node), s));
            }
            node = self.nodes[node].parent.unwrap();
        }
        None
    }

    /// Number of tokens from the root to the end of `node`.
    pub fn depth(&self, mut node: NodeId) -> usize {
        let mut d = 0;
        while node != ROOT {
            d += self.nodes[node].key.len();
            node = self.nodes[node].parent.unwrap();
        }
        d
    }

    /// Detach the least recently used snapshot of an unreferenced node (its KV stays cached).
    pub fn evict_snapshot(&mut self) -> Option<u32> {
        let id = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(id, n)| *id != ROOT && n.alive && n.snapshot.is_some() && n.ref_count == 0)
            .min_by_key(|(_, n)| n.last_access)
            .map(|(id, _)| id)?;
        self.nodes[id].snapshot.take()
    }

    pub fn snapshot_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.alive && n.snapshot.is_some())
            .count()
    }

    pub fn take_freed_snapshots(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.freed_snapshots)
    }

    /// Drop every unreferenced node, returning their slots.
    pub fn evict_all(&mut self) -> Vec<u32> {
        self.evict(usize::MAX)
    }

    /// Consistency check used by tests and debug builds.
    pub fn check(&self) -> Result<(), String> {
        let mut ev = 0;
        let mut pr = 0;
        for (id, n) in self.nodes.iter().enumerate() {
            if id == ROOT || !n.alive {
                continue;
            }
            if n.key.len() != n.value.len() || n.key.is_empty() {
                return Err(format!("node {id} has bad key/value"));
            }
            let parent = n.parent.ok_or("orphan")?;
            if self.nodes[parent].children.get(&n.key[0]) != Some(&id) {
                return Err(format!("node {id} not linked from parent"));
            }
            if n.ref_count > self.nodes[parent].ref_count && parent != ROOT {
                return Err(format!("node {id} ref above parent"));
            }
            if n.ref_count == 0 {
                ev += n.key.len();
            } else {
                pr += n.key.len();
            }
        }
        if ev != self.evictable || pr != self.protected {
            return Err(format!(
                "size mismatch ev {ev}/{} pr {pr}/{}",
                self.evictable, self.protected
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_match_split() {
        let mut c = RadixCache::new();
        assert_eq!(c.insert(&[1, 2, 3, 4], &[10, 20, 30, 40]), 0);
        let (node, slots) = c.match_prefix(&[1, 2, 9]);
        assert_eq!(slots, vec![10, 20]);
        assert_ne!(node, ROOT);
        c.check().unwrap();
        // inserting a sequence sharing [1,2] reports 2 cached tokens
        assert_eq!(c.insert(&[1, 2, 5], &[11, 21, 51]), 2);
        assert_eq!(c.match_prefix(&[1, 2, 5, 6]).1, vec![10, 20, 51]);
        assert_eq!(c.match_prefix(&[1, 2, 3, 4, 5]).1, vec![10, 20, 30, 40]);
        assert_eq!(c.evictable_size(), 5);
        c.check().unwrap();
    }

    #[test]
    fn lock_prevents_eviction_and_lru_order() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &[1, 2, 3]);
        c.insert(&[7, 8], &[7, 8]);
        let (n, _) = c.match_prefix(&[1, 2, 3]);
        c.lock(n);
        assert_eq!(c.protected_size(), 3);
        assert_eq!(c.evictable_size(), 2);
        let freed = c.evict(10);
        assert_eq!(freed, vec![7, 8]);
        assert_eq!(c.evict(10), Vec::<u32>::new());
        c.unlock(n);
        c.check().unwrap();
        let mut freed = c.evict(1);
        freed.sort();
        assert_eq!(freed, vec![1, 2, 3]);
        assert_eq!(c.cached_tokens(), 0);
        c.check().unwrap();
    }

    #[test]
    fn snapshots_follow_boundaries() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        let (n, _) = c.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(c.set_snapshot(n, 7), None);
        // a shorter match splits the node: the snapshot stays at the original end (depth 4)
        let (m, slots) = c.match_prefix(&[1, 2, 9]);
        assert_eq!(slots.len(), 2);
        assert_eq!(c.snapshot_ancestor(m), None);
        assert_eq!(c.snapshot_ancestor(n), Some((n, 4, 7)));
        // extending past the snapshot: nearest snapshot ancestor is at depth 4
        c.insert(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5, 6]);
        let (deep, _) = c.match_prefix(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(c.snapshot_ancestor(deep), Some((n, 4, 7)));
        // locked snapshots are not evicted; unlocked ones are
        c.lock(n);
        assert_eq!(c.evict_snapshot(), None);
        c.unlock(n);
        assert_eq!(c.evict_snapshot(), Some(7));
        assert_eq!(c.snapshot_count(), 0);
        // evicting a node releases its snapshot
        c.set_snapshot(deep, 8);
        c.evict(100);
        assert_eq!(c.take_freed_snapshots(), vec![8]);
        c.check().unwrap();
    }

    #[test]
    fn evicts_leaf_then_parent() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &[1, 2, 3]);
        c.insert(&[1, 2, 4], &[1, 2, 4]);
        c.match_prefix(&[1, 2, 4]); // most recent
        let freed = c.evict(1);
        assert_eq!(freed, vec![3]);
        let freed = c.evict(2);
        assert_eq!(freed, vec![4, 1, 2]);
        c.check().unwrap();
        // node ids are recycled
        c.insert(&[5], &[5]);
        c.check().unwrap();
    }
}
