// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Token-granular KV slot allocator (page size 1).

#[derive(Debug)]
pub struct SlotAllocator {
    free: Vec<u32>,
    capacity: usize,
}

impl SlotAllocator {
    pub fn new(capacity: usize) -> Self {
        // reversed so low slot ids are handed out first and runs stay mostly contiguous
        Self {
            free: (0..capacity as u32).rev().collect(),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn alloc(&mut self, n: usize) -> Option<Vec<u32>> {
        if n > self.free.len() {
            return None;
        }
        let at = self.free.len() - n;
        let mut out = self.free.split_off(at);
        out.reverse();
        Some(out)
    }

    pub fn free(&mut self, slots: &[u32]) {
        debug_assert!(slots.iter().all(|&s| (s as usize) < self.capacity));
        self.free.extend(slots.iter().rev().copied());
        debug_assert!(self.free.len() <= self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let mut a = SlotAllocator::new(8);
        let x = a.alloc(3).unwrap();
        assert_eq!(x, vec![0, 1, 2]);
        let y = a.alloc(5).unwrap();
        assert_eq!(y, vec![3, 4, 5, 6, 7]);
        assert!(a.alloc(1).is_none());
        a.free(&x);
        assert_eq!(a.available(), 3);
        assert_eq!(a.alloc(3).unwrap(), vec![0, 1, 2]);
    }
}
