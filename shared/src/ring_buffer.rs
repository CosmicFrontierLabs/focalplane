//! Fixed-capacity ring buffer for rolling data storage.
//!
//! Provides a generic ring buffer that automatically evicts oldest elements
//! when capacity is reached. Useful for rolling averages, history tracking,
//! and bounded logging.

use std::collections::vec_deque::Iter;
use std::collections::VecDeque;

/// A fixed-capacity ring buffer that evicts oldest elements when full.
///
/// When the buffer reaches capacity, pushing a new element automatically
/// removes the oldest element, maintaining a sliding window of the most
/// recent items.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Creates a new ring buffer with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of elements the buffer can hold
    ///
    /// # Panics
    /// Panics if capacity is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be greater than 0");
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Pushes an element to the back of the buffer.
    ///
    /// If the buffer is at capacity, the oldest element (front) is removed
    /// before adding the new element.
    pub fn push(&mut self, item: T) {
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the maximum capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an iterator over the elements in order (oldest to newest).
    ///
    /// The returned iterator implements `DoubleEndedIterator`, allowing
    /// reverse iteration with `.rev()`.
    pub fn iter(&self) -> Iter<'_, T> {
        self.items.iter()
    }

    /// Clears all elements from the buffer.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Returns a reference to the underlying VecDeque.
    ///
    /// Useful for direct access when needed for compatibility with existing code.
    pub fn as_deque(&self) -> &VecDeque<T> {
        &self.items
    }

    /// Returns a reference to the most recent element, if any.
    pub fn back(&self) -> Option<&T> {
        self.items.back()
    }

    /// Returns a reference to the oldest element, if any.
    pub fn front(&self) -> Option<&T> {
        self.items.front()
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Collects elements into a Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.items.iter().cloned().collect()
    }
}

impl<T: PartialEq> PartialEq for RingBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity && self.items == other.items
    }
}

impl<T> Default for RingBuffer<T> {
    /// Creates a ring buffer with a default capacity of 100.
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_and_len() {
        let mut buffer = RingBuffer::new(5);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.push(1);
        buffer.push(2);
        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);

        // Should evict 1
        buffer.push(4);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.to_vec(), vec![2, 3, 4]);

        // Should evict 2
        buffer.push(5);
        assert_eq!(buffer.to_vec(), vec![3, 4, 5]);
    }

    #[test]
    fn test_iter() {
        let mut buffer = RingBuffer::new(3);
        buffer.push("a");
        buffer.push("b");
        buffer.push("c");

        let items: Vec<_> = buffer.iter().collect();
        assert_eq!(items, vec![&"a", &"b", &"c"]);
    }

    #[test]
    fn test_clear() {
        let mut buffer = RingBuffer::new(5);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_front_and_back() {
        let mut buffer = RingBuffer::new(3);
        assert!(buffer.front().is_none());
        assert!(buffer.back().is_none());

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.front(), Some(&1));
        assert_eq!(buffer.back(), Some(&3));

        buffer.push(4); // evicts 1
        assert_eq!(buffer.front(), Some(&2));
        assert_eq!(buffer.back(), Some(&4));
    }

    #[test]
    fn test_capacity() {
        let buffer: RingBuffer<i32> = RingBuffer::new(42);
        assert_eq!(buffer.capacity(), 42);
    }

    #[test]
    #[should_panic(expected = "capacity must be greater than 0")]
    fn test_zero_capacity_panics() {
        let _buffer: RingBuffer<i32> = RingBuffer::new(0);
    }

    #[test]
    fn test_default() {
        let buffer: RingBuffer<i32> = RingBuffer::default();
        assert_eq!(buffer.capacity(), 100);
    }

    #[test]
    fn test_partial_eq() {
        let mut buf1 = RingBuffer::new(3);
        let mut buf2 = RingBuffer::new(3);

        buf1.push(1);
        buf1.push(2);
        buf2.push(1);
        buf2.push(2);

        assert_eq!(buf1, buf2);

        buf2.push(3);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_as_deque() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);

        let deque = buffer.as_deque();
        assert_eq!(deque.len(), 2);
        assert_eq!(deque[0], 1);
        assert_eq!(deque[1], 2);
    }
}
