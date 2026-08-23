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
//! Shared scaffolding for the integration tests.

use core::alloc::Layout;
use core::ptr::NonNull;

/// Page-aligned backing memory for an allocator under test.
///
/// Tests that expect a specific chunk count, or that exercise allocations
/// aligned beyond the chunk size, need to know where their region starts. A
/// page boundary is the strongest alignment the allocator can make use of, and
/// over-allocating a `Vec` and taking an aligned window out of it is enough to
/// get one.
#[derive(Debug)]
pub struct Region {
    buffer: Vec<u8>,
    offset: usize,
    len: usize,
}

impl Region {
    /// Creates `len` page-aligned bytes, every byte set to `fill`.
    ///
    /// A fill other than zero proves the allocator does not rely on a
    /// pre-zeroed region.
    pub fn new(len: usize, fill: u8) -> Self {
        let buffer = vec![fill; len + 4096];
        let offset = buffer.as_ptr().align_offset(4096);
        Self {
            buffer,
            offset,
            len,
        }
    }

    /// Creates memory that holds exactly `chunk_count` chunks plus their
    /// bitmap.
    pub fn for_chunks<const CHUNK_SIZE: usize>(chunk_count: usize) -> Self {
        Self::new(chunk_count * CHUNK_SIZE + chunk_count.div_ceil(8), 0)
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer[self.offset..self.offset + self.len]
    }
}

/// A live allocation together with the layout it was created for and a byte
/// pattern that marks it.
///
/// Giving each allocation its own pattern turns two allocations that overlap
/// in the heap into a failed assertion: the second [`fill`] overwrites the
/// first pattern, and the next [`assert_pattern`] of the older allocation sees
/// the foreign bytes.
///
/// [`fill`]: Self::fill
/// [`assert_pattern`]: Self::assert_pattern
#[derive(Debug)]
pub struct Allocation {
    pub ptr: NonNull<u8>,
    pub layout: Layout,
    pub pattern: u8,
}

impl Allocation {
    /// Writes the pattern over the whole allocation.
    pub fn fill(&self) {
        // SAFETY: `ptr` names a live allocation of `layout.size()` bytes.
        unsafe {
            core::ptr::write_bytes(
                self.ptr.as_ptr(),
                self.pattern,
                self.layout.size(),
            )
        };
    }

    /// Asserts that the first `len` bytes still carry the pattern.
    ///
    /// `len` is a parameter because a caller may know the pattern for fewer
    /// bytes than the allocation currently holds. After a growing realloc, for
    /// example, only the bytes copied from the old allocation carry it, while
    /// the rest is uninitialized and must not be read.
    pub fn assert_pattern(&self, len: usize) {
        assert!(
            len <= self.layout.size(),
            "cannot check more bytes than the allocation holds"
        );
        // SAFETY: `len` is within the live allocation, and the bytes were
        // initialized by `fill`.
        let bytes =
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), len) };
        assert!(
            bytes.iter().all(|byte| *byte == self.pattern),
            "allocation lost its pattern {:#x}",
            self.pattern
        );
    }
}
