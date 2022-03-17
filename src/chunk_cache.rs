/*
MIT License

Copyright (c) 2022 Philipp Schuster

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

/*/// Chunk Cache for possible alignments.
/// Helper struct for [`crate::ChunkAllocator`].
#[derive(Debug)]
pub(crate) struct ChunkCache {
    /// Used for all other alignments.
    align_1: Option<ChunkCacheEntry>,
    align_256: Option<ChunkCacheEntry>,
    align_512: Option<ChunkCacheEntry>,
    align_1024: Option<ChunkCacheEntry>,
    align_2048: Option<ChunkCacheEntry>,
    align_4096: Option<ChunkCacheEntry>,
}

impl ChunkCache {
    /// Constructor.
    ///
    /// The initialization assumes that the backing memory starts at a page aligned address.
    /// If this is not the case, this will cost performance but will not result in error.
    /// This cache is only a hint. The allocator will verify the availability of
    /// the entries and the alignment in any case.
    #[inline]
    pub const fn new() -> Self {
        Self {
            align_1: Some(ChunkCacheEntry::new(0, 1, 1)),
            align_256: Some(ChunkCacheEntry::new(0, 256, 1)),
            align_512: Some(ChunkCacheEntry::new(0, 512, 1)),
            align_1024: Some(ChunkCacheEntry::new(0, 1024, 1)),
            align_2048: Some(ChunkCacheEntry::new(0, 2048, 1)),
            align_4096: Some(ChunkCacheEntry::new(0, 4096, 1)),
        }
    }

    /// Updates an existing cache entry, if one exists for the provided alignment.
    #[inline(always)]
    pub const fn update(&mut self, index: usize, alignment: usize, chunk_count: usize) {
        debug_assert!(chunk_count > 0, "chunk count must be > 0");
        let entry = self.lookup_entry_by_alignment(alignment);
        if entry.is_none() {
            entry.replace(ChunkCacheEntry::new(index, alignment, chunk_count));
        } else {
            let entry_ref = entry.as_ref().unwrap();
            // prevent fragmentation; prefer small memory regions
            if entry_ref.chunk_count > chunk_count {
                entry.replace(ChunkCacheEntry::new(index, alignment, chunk_count));
            }
        }
    }

    #[inline(always)]
    pub const fn lookup(
        &mut self,
        alignment: usize,
        chunk_count: usize,
    ) -> Option<ChunkCacheEntry> {
        debug_assert!(chunk_count > 0, "chunk count must be > 0");
        let entry = self.lookup_entry_by_alignment(alignment);
        if entry.is_none() {
            None
        } else {
            let entry_ref = entry.as_ref().unwrap();
            if entry_ref.chunk_count >= chunk_count && entry_ref.alignment >= alignment {
                entry.take()
            } else {
                None
            }
        }
    }

    const fn lookup_entry_by_alignment(
        &mut self,
        alignment: usize,
    ) -> &mut Option<ChunkCacheEntry> {
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
        match alignment {
            256 => &mut self.align_256,
            512 => &mut self.align_512,
            1024 => &mut self.align_1024,
            2048 => &mut self.align_2048,
            4096 => &mut self.align_4096,
            _ => &mut self.align_1,
        }
    }
}*/

/// Single cache chunk entry.
#[derive(Debug)]
pub(crate) struct ChunkCacheEntry {
    /// Chunk index inside
    index: usize,
    /// Alignment. Power of 2. If this is 256 the entry can still be page-aligned (4096).
    /// This is only a hint that gets set and used during runtime.
    alignment: usize,
    /// Length of the continuous memory region in chunks. x > 0.
    chunk_count: usize,
}

impl ChunkCacheEntry {
    #[inline(always)]
    pub const fn new(index: usize, alignment: usize, chunk_count: usize) -> Self {
        debug_assert!(chunk_count > 0, "chunk count must be > 0");
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
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

    #[inline(always)]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    #[inline(always)]
    pub const fn chunk_count(&self) -> usize {
        self.chunk_count
    }
}

/*#[cfg(test)]
mod tests {
    use crate::chunk_cache::ChunkCache;

    #[test]
    fn test_chunk_cache() {
        let mut cache = ChunkCache::new();
        cache.update(0, 8, 128);
        cache.update(0, 128, 128);
        cache.update(0, 4096, 128);
        assert!(cache.lookup(4096, 129).is_none());
        assert!(cache.lookup(4096, 128).is_some());
        assert!(cache.lookup(4096, 128).is_none(), "already taken");
    }
}*/
