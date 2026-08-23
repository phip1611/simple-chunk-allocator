/*
MIT License

Copyright (c) 2026 Philipp Schuster

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
//! Module for [`ChunkCacheEntry`].

/// Hint where the next allocation search should start.
///
/// The entry is never trusted: [`crate::ChunkAllocator`] re-verifies
/// availability and alignment of every chunk it looks at. Its only purpose is
/// to keep the common case from scanning the whole bitmap.
#[derive(Debug)]
pub(crate) struct ChunkCacheEntry {
    /// Index of the first chunk of the cached region.
    index: usize,
    /// Known alignment of the cached region. Power of two.
    ///
    /// This is a lower bound: an entry recorded as 256-aligned may well sit at
    /// a page boundary.
    alignment: usize,
    /// Length of the continuous memory region in chunks. Always > 0.
    chunk_count: usize,
}

impl ChunkCacheEntry {
    #[inline(always)]
    pub const fn new(
        index: usize,
        alignment: usize,
        chunk_count: usize,
    ) -> Self {
        debug_assert!(chunk_count > 0, "chunk count must be > 0");
        debug_assert!(
            alignment.is_power_of_two(),
            "alignment must be power of 2"
        );
        Self {
            index,
            alignment,
            chunk_count,
        }
    }

    #[inline(always)]
    pub const fn index(&self) -> usize {
        self.index
    }

    #[expect(dead_code, reason = "the entry is about to become a plain index")]
    #[inline(always)]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    #[expect(dead_code, reason = "the entry is about to become a plain index")]
    #[inline(always)]
    pub const fn chunk_count(&self) -> usize {
        self.chunk_count
    }
}
