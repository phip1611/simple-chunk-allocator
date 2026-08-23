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
//! Module for [`ChunkAllocator`].

use crate::chunk_cache::ChunkCacheEntry;
use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

/// Zero sized types may trigger this; according to the Rust doc of the
/// `Allocator` trait this is intended. I work around this by changing the size
/// to 1. This makes core simpler.
macro_rules! normalize_layout {
    ($layout:ident) => {
        if $layout.size() == 0 {
            core::alloc::Layout::from_size_align(1, $layout.align()).unwrap()
        } else {
            $layout
        }
    };
}

/// Errors returned when creating or allocating from a [`ChunkAllocator`].
#[derive(Debug, Copy, Clone)]
pub enum ChunkAllocatorError {
    /// The heap is empty, misaligned, or has an incompatible length.
    BadHeapMemory,
    /// The bitmap does not contain exactly one bit per heap chunk.
    BadBitmapMemory,
    /// No free, suitably aligned run of chunks can satisfy the request.
    OutOfMemory,
}

/// Default chunk size: 256 bytes.
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Allocates from caller-provided storage in fixed-size chunks.
///
/// Each allocation consumes whole chunks. A larger chunk size reduces bitmap
/// size and search work; a smaller size reduces internal fragmentation.
#[derive(Debug)]
pub struct ChunkAllocator<const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE> {
    /// Backing memory for heap.
    heap: NonNull<u8>,
    /// Length of `heap` in bytes.
    heap_len: usize,
    /// Backing memory for bookkeeping.
    bitmap: NonNull<u8>,
    /// Length of `bitmap` in bytes.
    bitmap_len: usize,
    /// Helper to do some initial initialization on the first runtime
    /// invocation.
    is_first_alloc: Cell<bool>,
    /// Contains the next free continuous memory region with a minimum length
    /// of one chunk. It might happen that this entry is invalid because
    /// the heap is full or the next chunk after the previous allocation is
    /// already in use.
    ///
    /// This optimization mechanism prevents the need to iterate over all
    /// chunks everytime which can take up to tens of thousands of CPU
    /// cycles in the worst case (fragmented heap).
    maybe_next_free_chunk: ChunkCacheEntry,
    /// Counts the number of blocks in use.
    chunks_in_use: usize,
}

// SAFETY: the allocator is the exclusive owner of both backing allocations, and
// its raw pointers stay valid wherever it is moved to.
unsafe impl<const CHUNK_SIZE: usize> Send for ChunkAllocator<CHUNK_SIZE> {}

impl<const CHUNK_SIZE: usize> ChunkAllocator<CHUNK_SIZE> {
    /// Rejects an invalid chunk size when the allocator is instantiated.
    ///
    /// A power of two is what makes every chunk `CHUNK_SIZE`-aligned once the
    /// heap base is. Zero is covered as well, because `0` is not a power of
    /// two.
    const VALIDATE_CHUNK_SIZE: () = assert!(
        CHUNK_SIZE.is_power_of_two(),
        "CHUNK_SIZE must be a power of two"
    );

    /// Returns the used chunk size.
    #[inline]
    pub const fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }

    /// Returns the minimum guaranteed alignment by the allocator per chunk.
    /// A chunk will never be at an address like 0x13, i.e., unaligned.
    pub const fn min_alignment(&self) -> usize {
        CHUNK_SIZE
    }

    /// Creates an allocator over caller-provided backing memory.
    ///
    /// The heap must be non-empty, `CHUNK_SIZE`-aligned, and hold a multiple
    /// of eight chunks; the bitmap must hold exactly one bit per chunk. All of
    /// this is checked here and panics on violation, which turns into a
    /// compile error when the allocator is built in a const context. The
    /// alignment is the exception: const evaluation cannot see an address, so
    /// it is verified on the first allocation instead [0].
    ///
    /// [0]: https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
    ///
    /// # Safety
    /// `heap` and `bitmap` must be valid, non-null and non-overlapping for as
    /// long as the allocator lives, and it must be their only user for that
    /// time. Nothing in the type system enforces this any more.
    #[inline]
    pub const unsafe fn new(heap: *mut [u8], bitmap: *mut [u8]) -> Self {
        let () = Self::VALIDATE_CHUNK_SIZE;

        // SAFETY: validity and exclusivity are guaranteed by the caller.
        let (heap, bitmap) = unsafe { (&mut *heap, &mut *bitmap) };

        assert!(
            !heap.is_empty() && heap.len().is_multiple_of(CHUNK_SIZE),
            "heap must be not empty and a multiple of the chunk size"
        );

        let chunk_count = heap.len() / CHUNK_SIZE;
        assert!(
            chunk_count.is_multiple_of(8),
            "chunk count must be a multiple of 8"
        );
        assert!(
            chunk_count == bitmap.len() * 8,
            "the bitmap must cover the amount of chunks exactly"
        );

        let heap_len = heap.len();
        let bitmap_len = bitmap.len();
        Self {
            heap: NonNull::new(heap.as_mut_ptr()).unwrap(),
            heap_len,
            bitmap: NonNull::new(bitmap.as_mut_ptr()).unwrap(),
            bitmap_len,
            is_first_alloc: Cell::new(true),
            // The real alignment is unknown until the first allocation, so the
            // hint starts at the weakest possible value.
            maybe_next_free_chunk: ChunkCacheEntry::new(0, 1, chunk_count),
            chunks_in_use: 0,
        }
    }

    /// Capacity in bytes of the allocator.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.heap_len
    }

    /// Returns number of chunks.
    #[inline]
    pub const fn chunk_count(&self) -> usize {
        // size is a multiple of CHUNK_SIZE;
        // ensured in new()
        self.capacity() / CHUNK_SIZE
    }

    /// Returns the current memory usage in percentage rounded to two decimal
    /// places.
    #[inline]
    pub fn usage(&self) -> f32 {
        if self.chunks_in_use == 0 {
            0.0
        } else {
            let ratio = self.chunks_in_use as f32 / self.chunk_count() as f32;
            libm::roundf(ratio * 10000.0) / 100.0
        }
    }

    /// Returns whether a chunk is free according to the bitmap.
    ///
    /// # Parameters
    /// - `chunk_index` describes the start chunk; i.e. the search space inside
    ///   the backing storage
    #[inline(always)]
    fn chunk_is_free(&self, chunk_index: usize) -> bool {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index={} is bigger than max chunk index={}",
            chunk_index,
            self.chunk_count() - 1
        );
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // SAFETY: `byte_i` is within the validated bitmap capacity.
        let relevant_bit =
            unsafe { (*self.bitmap.as_ptr().add(byte_i) >> bit) & 1 };
        relevant_bit == 0
    }

    /// Marks a chunk as used, i.e. write a 1 into the bitmap at the right
    /// position.
    #[inline(always)]
    fn mark_chunk_as_used(&mut self, chunk_index: usize) {
        debug_assert!(chunk_index < self.chunk_count());
        if !self.chunk_is_free(chunk_index) {
            panic!(
                "tried to mark chunk {} as used but it is already used",
                chunk_index
            );
        }
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // xor => keep all bits, except bitflip at relevant position
        // SAFETY: `byte_i` is within the validated bitmap capacity.
        unsafe { *self.bitmap.as_ptr().add(byte_i) ^= 1 << bit };
    }

    /// Marks a chunk as free, i.e. write a 0 into the bitmap at the right
    /// position.
    #[inline(always)]
    fn mark_chunk_as_free(&mut self, chunk_index: usize) {
        debug_assert!(chunk_index < self.chunk_count());
        if self.chunk_is_free(chunk_index) {
            panic!(
                "tried to mark chunk {} as free but it is already free",
                chunk_index
            );
        }
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // xor => keep all bits, except bitflip at relevant position
        // SAFETY: `byte_i` is within the validated bitmap capacity.
        unsafe {
            let byte = self.bitmap.as_ptr().add(byte_i);
            *byte ^= 1 << bit;
        }
    }

    /// Returns the indices into the bitmap array of a given chunk index.
    #[inline(always)]
    fn chunk_index_to_bitmap_indices(
        &self,
        chunk_index: usize,
    ) -> (usize, usize) {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index out of range!"
        );
        (chunk_index / 8, chunk_index % 8)
    }

    /// Finds the next available continuous memory region, i.e. coherent
    /// available/free chunks. Returns the beginning index. Does not mark
    /// them as used. This is the responsibility of the caller.
    ///
    /// # Parameters
    /// - `chunk_num_request` number of chunks that must be all free without
    ///   gaps in-between; greater than 0
    /// - `alignment` required alignment of the chunk in memory. Must be a power
    ///   of 2. This usually comes from [`core::alloc::Layout`] which already
    ///   guarantees that it is a power of two.
    #[inline(always)]
    fn find_free_continuous_memory_region(
        &mut self,
        chunk_num_request: usize,
        alignment: usize,
    ) -> Result<usize, ChunkAllocatorError> {
        if chunk_num_request > self.chunk_count() {
            // out of memory
            return Err(ChunkAllocatorError::OutOfMemory);
        }

        // We hope that the index and its succeeding chunks stored in the cache
        // fits the requested memory region.
        let start_index = self.maybe_next_free_chunk.index();

        let chunk_count = self.chunk_count();

        (start_index..(start_index + chunk_count))
            .filter_map(|index| {
                // Cope with wrapping indices (i.e. index 0 follows 31).
                // This will lead to scenarios where it iterates like:
                // 4,5,6,7,0,1,2,3 (assuming there are 8
                // chunks).
                let chunk_index = index % chunk_count;

                // It only makes sense to start the lookup at chunks that are
                // available.
                if !self.chunk_is_free(chunk_index) {
                    return None;
                }

                // If the heap has 8 chunks and we need 4 but start the search
                // at index 6, then we don't have enough
                // continuous chunks to fulfill the request. Thus, we skip
                // those.
                if chunk_index + chunk_num_request > self.chunk_count() {
                    return None;
                }

                // Does the heap address has the right alignment to fulfill the
                // request? SAFETY: `chunk_index` is within the
                // allocator's heap.
                let ptr = unsafe { self.chunk_index_to_ptr(chunk_index) };
                if ptr.align_offset(alignment) != 0 {
                    return None;
                }

                // Now look for the continuous region: are all succeeding chunks
                // free? This is safe because earlier I skipped
                // chunk_indices that are too close to
                // the end. Return the first result.
                let is_free_region = {
                    // inclusive
                    let from = chunk_index + 1;
                    // -1: indices start at 0
                    // exclusive
                    let to = from + chunk_num_request - 1;

                    (from..to).all(|index| self.chunk_is_free(index))
                };

                if !is_free_region {
                    return None;
                }

                Some(chunk_index)
            })
            .next()
            // OK or out of memory
            .ok_or(ChunkAllocatorError::OutOfMemory)
    }

    /// Returns the pointer to the beginning of the chunk.
    #[inline(always)]
    unsafe fn chunk_index_to_ptr(&mut self, chunk_index: usize) -> *mut u8 {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index out of range!"
        );
        // SAFETY: `chunk_index` is bounded by `chunk_count`.
        unsafe { self.heap.as_ptr().add(chunk_index * CHUNK_SIZE) }
    }

    /// Returns the chunk index of the given pointer (which points to the
    /// beginning of a chunk).
    #[inline(always)]
    unsafe fn ptr_to_chunk_index(&self, ptr: *const u8) -> usize {
        let heap_begin_inclusive = self.heap.as_ptr().cast_const();
        // SAFETY: `heap_len` is the length of the backing allocation.
        let heap_end_exclusive =
            unsafe { self.heap.as_ptr().add(self.heap_len) };
        debug_assert!(
            heap_begin_inclusive <= ptr && ptr < heap_end_exclusive,
            "pointer {ptr:?} is outside the allocator's backing storage \
             {heap_begin_inclusive:?}..{heap_end_exclusive:?}"
        );
        (ptr as usize - heap_begin_inclusive as usize) / CHUNK_SIZE
    }

    /// Calculates the number of required chunks to fulfill an allocation
    /// request.
    #[inline(always)]
    const fn calc_required_chunks(&self, size: usize) -> usize {
        assert!(size > 0);
        size.div_ceil(CHUNK_SIZE)
    }

    /// Performs initialization steps on the first allocation.
    /// - checks heap memory (alignment etc) because this can't be done during
    ///   const new initialization
    /// - zeroes the bitmap
    fn init(&mut self) -> Result<(), ChunkAllocatorError> {
        self.is_first_alloc.replace(false);
        // Zero bitmap
        // SAFETY: `bitmap` points to `bitmap_len` bytes of exclusively owned
        // storage.
        unsafe {
            core::ptr::write_bytes(self.bitmap.as_ptr(), 0, self.bitmap_len)
        };

        if self.heap.as_ptr().align_offset(4096) != 0 && CHUNK_SIZE < 4096 {
            log::debug!(
                "Page-aligned backing memory is recommended for the heap."
            );
        }

        // this can't be done in const new constructor
        // see: https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
        if self.heap.as_ptr().align_offset(self.min_alignment()) != 0 {
            log::error!(
                "The heap must be CHUNK_SIZE-aligned; page alignment is recommended."
            );
            Err(ChunkAllocatorError::BadHeapMemory)
        } else {
            // Now update; we checked the minimum alignment
            self.maybe_next_free_chunk = ChunkCacheEntry::new(
                self.maybe_next_free_chunk.index(),
                self.min_alignment(),
                self.maybe_next_free_chunk.chunk_count(),
            );
            Ok(())
        }
    }

    /// Allocates memory according to the specific layout.
    #[track_caller]
    #[inline]
    #[must_use = "The pointer must be freed eventually to prevent memory leaks."]
    pub fn allocate(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, ChunkAllocatorError> {
        log::trace!("called allocate");
        if self.is_first_alloc.get() {
            self.init()?;
        }

        let layout = normalize_layout!(layout);

        let required_chunks = self.calc_required_chunks(layout.size());

        let index = self.find_free_continuous_memory_region(
            required_chunks,
            layout.align(),
        );

        if index.is_err() {
            log::warn!(
                "Out of memory for {layout:?}; usage: {}%/{} byte",
                self.usage(),
                ((self.usage() * self.capacity() as f32) as u64)
            );
        }

        // unwrap or return error
        let index = index?;

        for i in index..index + required_chunks {
            self.mark_chunk_as_used(i);
        }
        self.chunks_in_use += required_chunks;

        // Only update "maybe_next_free_chunk" if it doesn't already point to a
        // free location; For example, it could be that it was not used
        // in this allocation.
        //
        // MAKE SURE THIS GETS CALLED AFTER USED CHUNKS ARE MARKED AS SUCH
        // EARLIER.
        if !self.chunk_is_free(self.maybe_next_free_chunk.index()) {
            // at next allocation: continue search at this index
            let next_index = (index + 1) % self.chunk_count();
            // - alignment of chunk_size is always guaranteed.
            // - We do not know yet if the next entry is actually available. We
            //   just give the algorithm an hint where to start for the next
            //   search.
            self.maybe_next_free_chunk =
                ChunkCacheEntry::new(next_index, CHUNK_SIZE, 1);
        }

        // SAFETY: `index` was returned from the in-bounds chunk search.
        let heap_ptr = unsafe { self.chunk_index_to_ptr(index) };
        log::trace!(
            "alloc: layout={layout:?}, ptr={heap_ptr:?}, #chunks={}",
            required_chunks
        );
        let heap_ptr = NonNull::new(heap_ptr).unwrap();
        Ok(NonNull::slice_from_raw_parts(
            heap_ptr,
            required_chunks * self.chunk_size(),
        ))
    }

    /// Deallocates an allocation from this allocator.
    ///
    /// # Safety
    /// `ptr` must be a live allocation returned by this allocator for `layout`.
    /// It must be deallocated exactly once and not used afterwards.
    #[track_caller]
    #[inline]
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        log::trace!("called deallocate");
        let layout = normalize_layout!(layout);

        let freed_chunks = self.calc_required_chunks(layout.size());

        log::trace!("dealloc: layout={:?}, #chunks={})", layout, freed_chunks);

        // SAFETY: callers must pass a pointer returned by this allocator.
        let index = unsafe { self.ptr_to_chunk_index(ptr.as_ptr()) };
        for i in index..index + freed_chunks {
            self.mark_chunk_as_free(i);
        }
        self.chunks_in_use -= freed_chunks;

        // This helps the next allocation to be faster because we know that this
        // block was just freed. This only works if the next allocation
        // fits into the continuous region of memory.
        //
        // Currently, this prefers the smallest possible continuous region with
        // the lowest possible alignment which prevents fragmentation at
        // the cost of larger lookup times. It assumes/hopes the next
        // allocation only needs as few chunks as possible (ideally a
        // fitting one).
        //
        // Small alignments (1, 2, 4, 8) are common but 4096 (page-alignment) is
        // rather rare. Therefore, it is okay to try to prevent small
        // allocations in addresses with big alignment.

        // 1) freed memory region smaller then cached?
        if freed_chunks < self.maybe_next_free_chunk.chunk_count()
            // 2) if same size: check alignment
            || (freed_chunks == self.maybe_next_free_chunk.chunk_count()
                // The layout alignment is greater than the minimum alignment.
                // and smaller than the one currently cached
                && layout.align() > CHUNK_SIZE
                && layout.align() < self.maybe_next_free_chunk.alignment())
        {
            self.maybe_next_free_chunk =
                ChunkCacheEntry::new(index, layout.align(), freed_chunks);
        }
    }

    /// Resizes an allocation from this allocator.
    ///
    /// # Safety
    /// `ptr` must be a live allocation returned by this allocator for
    /// `old_layout`. The caller must not use `ptr` after a successful move.
    #[track_caller]
    #[inline]
    pub unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<[u8]>, ChunkAllocatorError> {
        log::trace!("called realloc");

        // zero sized types may trigger this; according to the Rust doc of the
        // `Allocator` trait this is intended. I work around this by
        // changing the size to 1.
        let old_layout = normalize_layout!(old_layout);

        let required_chunks = self.calc_required_chunks(old_layout.size());
        let occupied_size = required_chunks * CHUNK_SIZE;

        // Reuse the allocation when it already has enough space.
        if new_size <= occupied_size {
            // `max(1)`: a shrink to zero keeps the allocation alive, and
            // `normalize_layout!` will report one chunk on the matching
            // deallocation.
            let required_new_chunks =
                self.calc_required_chunks(new_size.max(1));
            if required_new_chunks < required_chunks {
                // SAFETY: callers provide a live allocation from this
                // allocator.
                let index = unsafe { self.ptr_to_chunk_index(ptr.as_ptr()) };
                // The allocation keeps its leading `required_new_chunks`
                // chunks; the chunks behind them become free again.
                let begin = index + required_new_chunks;
                let end = index + required_chunks;
                for chunk_index in begin..end {
                    self.mark_chunk_as_free(chunk_index);
                }
                self.chunks_in_use -= end - begin;
            }
            log::trace!("realloc fast return possible!");
            Ok(NonNull::slice_from_raw_parts(ptr, new_size))
        } else {
            log::trace!("realloc fast return NOT possible!");

            // SAFETY: the caller must ensure that the `new_size` does not
            // overflow. `layout.align()` comes from a `Layout` and
            // is thus guaranteed to be valid.
            let new_layout =
                // SAFETY: `new_size` and `old_layout` satisfy `realloc`'s contract.
                unsafe {
                    Layout::from_size_align_unchecked(new_size, old_layout.align())
                };
            // SAFETY: the caller must ensure that `new_layout` is greater than
            // zero.
            let new_ptr = self.allocate(new_layout)?;

            // SAFETY: the allocations do not overlap and the caller owns `ptr`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new_ptr.as_mut_ptr(),
                    core::cmp::min(old_layout.size(), new_size),
                );
                self.deallocate(ptr, old_layout);
            }

            Ok(new_ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PageAligned;
    use std::alloc::{AllocError, Allocator, Global};
    use std::cmp::max;
    use std::ptr::NonNull;
    use std::vec::Vec;

    mod helpers {

        use super::*;

        /// Helper struct to let the std vector align stuff at a page boundary.
        /// Forwards requests to the global Rust allocator provided by the
        /// standard library.
        pub struct GlobalPageAlignedAlloc;

        // SAFETY: each method forwards to `Global` using the matching layout.
        unsafe impl Allocator for GlobalPageAlignedAlloc {
            fn allocate(
                &self,
                layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                let alignment = max(layout.align(), 4096);
                // unwrap should never fail, because layout.align() is already a
                // power of 2, otherwise the value not exist
                // here.
                let layout = layout.align_to(alignment).unwrap();
                Global.allocate(layout)
            }

            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                let alignment = max(layout.align(), 4096);
                // unwrap should never fail, because layout.align() is already a
                // power of 2, otherwise the value not exist
                // here.
                let layout = layout.align_to(alignment).unwrap();
                // SAFETY: delegated allocation used the same adjusted layout.
                unsafe { Global.deallocate(ptr, layout) }
            }
        }

        /// Creates backing memory for the allocator and the bitmap management
        /// structure. Uses the std global allocator for this. The
        /// memory is page-aligned.
        pub fn create_heap_and_bitmap_vectors() -> (
            Vec<u8, GlobalPageAlignedAlloc>,
            Vec<u8, GlobalPageAlignedAlloc>,
        ) {
            // 32 chunks with default chunk size = 256 bytes = 2 pages = 2*4096
            const CHUNK_COUNT: usize = 32;
            const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
            let mut heap =
                Vec::with_capacity_in(HEAP_SIZE, GlobalPageAlignedAlloc);
            (0..heap.capacity()).for_each(|_| heap.push(0));
            const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE / 8;
            let mut heap_bitmap =
                Vec::with_capacity_in(BITMAP_SIZE, GlobalPageAlignedAlloc);
            (0..heap_bitmap.capacity()).for_each(|_| heap_bitmap.push(0));

            assert_eq!(
                heap.as_ptr().align_offset(DEFAULT_CHUNK_SIZE),
                0,
                "heap must be at least allocated to CHUNK_SIZE"
            );
            assert_eq!(
                heap.as_ptr().align_offset(4096),
                0,
                "must be page aligned"
            );
            assert_eq!(
                heap_bitmap.as_ptr().align_offset(4096),
                0,
                "must be page aligned"
            );

            (heap, heap_bitmap)
        }

        pub fn create_heap_and_bitmap_vectors_for<const CHUNK_SIZE: usize>(
            chunk_count: usize,
        ) -> (
            Vec<u8, GlobalPageAlignedAlloc>,
            Vec<u8, GlobalPageAlignedAlloc>,
        ) {
            assert!(chunk_count.is_multiple_of(8));
            let heap_size = CHUNK_SIZE * chunk_count;
            let mut heap =
                Vec::with_capacity_in(heap_size, GlobalPageAlignedAlloc);
            heap.resize(heap_size, 0);
            let mut bitmap =
                Vec::with_capacity_in(chunk_count / 8, GlobalPageAlignedAlloc);
            bitmap.resize(chunk_count / 8, 0);
            assert_eq!(heap.as_ptr().align_offset(CHUNK_SIZE), 0);
            (heap, bitmap)
        }

        /// Creates an allocator over the given backing memory.
        ///
        /// Every test keeps its backing vectors in scope for as long as the
        /// allocator, so the shim can hide the unsafe block that every
        /// construction would otherwise repeat.
        pub fn allocator_over<const CHUNK_SIZE: usize>(
            heap: &mut [u8],
            bitmap: &mut [u8],
        ) -> ChunkAllocator<CHUNK_SIZE> {
            // SAFETY: both slices are distinct allocations, so they cannot
            // overlap. Callers keep them alive and untouched, which the
            // returned allocator no longer expresses in its type.
            unsafe { ChunkAllocator::new(heap, bitmap) }
        }

        /// A live allocation together with the layout it was created for and
        /// a byte pattern that marks it.
        ///
        /// Giving each allocation its own pattern turns two allocations that
        /// overlap in the heap into a failed assertion: the second [`fill`]
        /// overwrites the first pattern, and the next [`assert_pattern`] of
        /// the older allocation sees the foreign bytes.
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
                // SAFETY: `ptr` names a live allocation of `layout.size()`
                // bytes.
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
            /// `len` is a parameter because a caller may know the pattern for
            /// fewer bytes than the allocation currently holds. After a
            /// growing realloc, for example, only the bytes copied from the
            /// old allocation carry it, while the rest is uninitialized and
            /// must not be read.
            pub fn assert_pattern(&self, len: usize) {
                assert!(
                    len <= self.layout.size(),
                    "cannot check more bytes than the allocation holds"
                );
                // SAFETY: `len` is within the live allocation, and the bytes
                // were initialized by `fill`.
                let bytes = unsafe {
                    core::slice::from_raw_parts(self.ptr.as_ptr(), len)
                };
                assert!(
                    bytes.iter().all(|byte| *byte == self.pattern),
                    "allocation lost its pattern {:#x}",
                    self.pattern
                );
            }
        }
    }

    /// The constructor is the only place that can reject a bad heap/bitmap
    /// geometry. It panics instead of returning an error so that the mistake
    /// becomes a compile error in the const context it is meant for.
    #[test]
    fn test_new_rejects_bad_geometry() {
        const CS: usize = DEFAULT_CHUNK_SIZE;
        let cases = [
            (0, 0, "empty heap"),
            (8 * CS + 1, 1, "heap length is not a multiple of CHUNK_SIZE"),
            (4 * CS, 1, "chunk count is not a multiple of eight"),
            (8 * CS, 2, "bitmap covers more chunks than the heap has"),
        ];

        for (heap_size, bitmap_size, case) in cases {
            std::panic::catch_unwind(move || {
                let mut heap = vec![0_u8; heap_size];
                let mut bitmap = vec![0_u8; bitmap_size];
                let _alloc =
                    helpers::allocator_over::<CS>(&mut heap, &mut bitmap);
            })
            .expect_err(case);
        }
    }

    /// Constructing from raw pointers into statics is the shape a
    /// `#[global_allocator]` needs. That this also works in a const context is
    /// covered by the doctests and `examples/minimal.rs`, which build a
    /// `static` allocator.
    #[test]
    fn test_new_accepts_raw_static_memory() {
        const CHUNK_COUNT: usize = 16;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
        static mut HEAP: PageAligned<[u8; HEAP_SIZE]> =
            PageAligned::new([0; HEAP_SIZE]);
        static mut HEAP_BITMAP: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];

        // SAFETY: both statics are used by this allocator alone.
        let alloc: ChunkAllocator = unsafe {
            ChunkAllocator::new(
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(HEAP).cast(),
                    HEAP_SIZE,
                ),
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(HEAP_BITMAP).cast(),
                    BITMAP_SIZE,
                ),
            )
        };
        assert_eq!(alloc.chunk_count(), CHUNK_COUNT);
    }

    /// The bitmap must describe every chunk of the heap and nothing beyond it.
    #[test]
    #[cfg_attr(miri, ignore)] // passes but is very slow in Miri
    fn test_chunk_count_matches_bitmap() {
        // The bitmap has byte granularity, so the smallest heap the
        // constructor accepts holds eight chunks. 128 is an arbitrary upper
        // bound.
        for chunk_count in (8..128).step_by(8) {
            let (mut heap, mut bitmap) =
                helpers::create_heap_and_bitmap_vectors_for::<DEFAULT_CHUNK_SIZE>(
                    chunk_count,
                );
            let alloc: ChunkAllocator =
                helpers::allocator_over(&mut heap, &mut bitmap);
            assert_eq!(chunk_count, alloc.chunk_count());
        }
    }

    /// Tests the method `chunk_index_to_bitmap_indices()`.
    #[test]
    fn test_chunk_index_to_bitmap_indices() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        let alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        // chunk 3 gets described by bitmap byte 0 bit 3
        assert_eq!((0, 3), alloc.chunk_index_to_bitmap_indices(3));
        assert_eq!((0, 7), alloc.chunk_index_to_bitmap_indices(7));
        // chunk 8 gets described by bitmap byte 1 bit 0
        assert_eq!((1, 0), alloc.chunk_index_to_bitmap_indices(8));
        assert_eq!((1, 1), alloc.chunk_index_to_bitmap_indices(9));
        assert_eq!((1, 7), alloc.chunk_index_to_bitmap_indices(15));
    }

    /// Gives the allocator a bitmap where a few fields
    #[test]
    fn test_chunk_is_free() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        heap_bitmap[0] = 0x2f;
        let alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        assert!(!alloc.chunk_is_free(0));
        assert!(!alloc.chunk_is_free(1));
        assert!(!alloc.chunk_is_free(2));
        assert!(!alloc.chunk_is_free(3));
        assert!(alloc.chunk_is_free(4));
        assert!(!alloc.chunk_is_free(5));
    }

    /// Tests the `chunk_index_to_ptr` method.
    #[test]
    fn test_chunk_index_to_ptr() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        let heap_ptr = heap.as_ptr();
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        // SAFETY: all computed pointers remain within `heap`.
        unsafe {
            assert_eq!(heap_ptr, alloc.chunk_index_to_ptr(0));
            assert_eq!(
                heap_ptr.add(alloc.chunk_size()),
                alloc.chunk_index_to_ptr(1)
            );
            assert_eq!(
                heap_ptr.add(alloc.chunk_size() * 7),
                alloc.chunk_index_to_ptr(7)
            );
        }
    }

    /// Test to get single chunks of memory. Tests
    /// `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_basic() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        // I made this test for these two properties. Test might need to get
        // adjusted if they change
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        assert_eq!(
            0,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(0);
        alloc.maybe_next_free_chunk =
            ChunkCacheEntry::new(1, DEFAULT_CHUNK_SIZE, 1);

        assert_eq!(1, alloc.find_free_continuous_memory_region(1, 1).unwrap());
        alloc.maybe_next_free_chunk =
            ChunkCacheEntry::new(2, DEFAULT_CHUNK_SIZE, 1);
        assert_eq!(
            // 16: 256*16 = 4096 => second page in heap mem that consists of
            // two pages
            16,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(16);
        // makes sure the next search
        alloc.maybe_next_free_chunk =
            ChunkCacheEntry::new(17, DEFAULT_CHUNK_SIZE, 1);

        assert!(
            alloc.find_free_continuous_memory_region(1, 4096).is_err(),
            "out of memory; only 2 pages of memory"
        );

        // now free the first chunk again, which enables a further 4096 byte
        // aligned allocation
        alloc.mark_chunk_as_free(0);
        assert_eq!(
            0,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
    }

    /// Test to get a continuous region of memory. Tests
    /// `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_full_1() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        // I made this test for these two properties. Test might need to get
        // adjusted if they change
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        assert!(
            alloc.find_free_continuous_memory_region(33, 1).is_err(),
            "out of memory"
        );

        // free all 32 chunks; claim again
        let res = alloc.find_free_continuous_memory_region(32, 1);
        assert!(res.is_ok());
        assert_eq!(0, res.unwrap());
        for i in 0..32 {
            alloc.mark_chunk_as_used(i);
        }

        assert!(
            alloc.find_free_continuous_memory_region(32, 1).is_err(),
            "out of memory"
        );

        // free first 16 chunks; claim again
        for i in 16..32 {
            alloc.mark_chunk_as_free(i);
        }
        let res = alloc.find_free_continuous_memory_region(16, 4096);
        assert_eq!(16, res.unwrap());
        for i in 16..32 {
            alloc.mark_chunk_as_used(i);
        }
    }

    /// Test to get a continuous region of memory. Tests
    /// `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_full_2() {
        let (mut heap, mut heap_bitmap) =
            helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(&mut heap, &mut heap_bitmap);

        // I made this test for these two properties. Test might need to get
        // adjusted if they change
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        alloc.mark_chunk_as_used(0);
        alloc.mark_chunk_as_used(1);
        alloc.mark_chunk_as_used(2);
        alloc.mark_chunk_as_used(16);

        assert!(
            alloc.find_free_continuous_memory_region(1, 4096).is_err(),
            "out of memory: chunks 0 and 16 occupy the only page-aligned addresses"
        );
        assert_eq!(
            17,
            alloc.find_free_continuous_memory_region(15, 1).unwrap(),
        );
    }

    #[test]
    fn test_allocate_respects_boundaries_and_reuses_chunks() {
        let (mut heap, mut bitmap) =
            helpers::create_heap_and_bitmap_vectors_for::<256>(8);
        let mut allocator =
            helpers::allocator_over::<256>(&mut heap, &mut bitmap);
        let layout = Layout::from_size_align(256, 1).unwrap();
        let mut allocations = Vec::new();

        for _ in 0..8 {
            allocations.push(allocator.allocate(layout).unwrap());
        }
        assert!(matches!(
            allocator.allocate(layout),
            Err(ChunkAllocatorError::OutOfMemory)
        ));
        assert_eq!(allocator.usage(), 100.0);

        let ptr = allocations.pop().unwrap().cast();
        // SAFETY: `ptr` is the most recent live allocation for `layout`.
        unsafe { allocator.deallocate(ptr, layout) };
        assert!(allocator.allocate(layout).is_ok());
    }

    #[test]
    fn test_allocate_honors_requested_alignment() {
        let (mut heap, mut bitmap) =
            helpers::create_heap_and_bitmap_vectors_for::<256>(64);
        let mut allocator =
            helpers::allocator_over::<256>(&mut heap, &mut bitmap);
        let mut allocations = Vec::new();

        for alignment in
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        {
            let layout = Layout::from_size_align(1, alignment).unwrap();
            let allocation = allocator.allocate(layout).unwrap();
            assert_eq!(
                allocation.as_ptr().cast::<u8>().align_offset(alignment),
                0,
                "allocation for alignment {alignment} is misaligned"
            );
            allocations.push((allocation.cast(), layout));
        }

        for (ptr, layout) in allocations {
            // SAFETY: each pointer is live and paired with its original layout.
            unsafe { allocator.deallocate(ptr, layout) };
        }
        assert_eq!(allocator.usage(), 0.0);
    }

    /// Follows one allocation through grow and shrink and checks the chunk
    /// bookkeeping after every step. The heap has 16 chunks, so every expected
    /// usage below is `chunks_in_use / 16`.
    #[test]
    fn test_realloc_preserves_data_and_releases_chunks() {
        let (mut heap, mut bitmap) =
            helpers::create_heap_and_bitmap_vectors_for::<256>(16);
        let mut allocator =
            helpers::allocator_over::<256>(&mut heap, &mut bitmap);
        let old_layout = Layout::from_size_align(128, 64).unwrap();
        let allocation = allocator.allocate(old_layout).unwrap();
        let record = helpers::Allocation {
            ptr: allocation.cast(),
            layout: old_layout,
            pattern: 0xa5,
        };
        record.fill();
        // 128 byte occupy a single chunk.
        assert_eq!(allocator.usage(), 6.25);

        // 600 byte do not fit into the occupied chunk, so this moves the
        // allocation to a region of three chunks and frees the old one.
        // SAFETY: `record` describes a live allocation from `allocator`.
        let grown =
            unsafe { allocator.realloc(record.ptr, old_layout, 600) }.unwrap();
        let grown_layout = Layout::from_size_align(600, 64).unwrap();
        let grown_record = helpers::Allocation {
            ptr: grown.cast(),
            layout: grown_layout,
            pattern: record.pattern,
        };
        // Only the 128 byte copied from the old allocation are known to
        // carry the pattern. The remaining bytes of the larger allocation are
        // uninitialized, so the check stops at the old size.
        grown_record.assert_pattern(old_layout.size());
        assert_eq!(allocator.usage(), 18.75);

        // Shrinking stays in place, but the two chunks that are no longer
        // backed by the allocation must be released here. Before, they stayed
        // marked as used until the process ended.
        // SAFETY: `grown_record` describes the live replacement allocation.
        let zero =
            unsafe { allocator.realloc(grown_record.ptr, grown_layout, 0) }
                .unwrap();
        assert_eq!(zero.len(), 0);
        // A zero-size allocation still owns one chunk; see `normalize_layout!`.
        assert_eq!(allocator.usage(), 6.25);

        let zero_layout = Layout::from_size_align(0, 1).unwrap();
        // SAFETY: the zero-size result retains the same live allocation.
        unsafe { allocator.deallocate(zero.cast(), zero_layout) };
        assert_eq!(allocator.usage(), 0.0);
    }

    /// Drives allocate, deallocate and realloc in an order that the
    /// hand-written tests do not reach: they check one operation at a time on
    /// an otherwise fresh heap, while the bugs of a bitmap allocator show up
    /// after the heap has become fragmented and the free-chunk hint points
    /// somewhere in the middle.
    ///
    /// Two invariants carry the test. Every allocation is filled with its own
    /// pattern and re-checked before it is touched again, so any overlap
    /// between two live allocations fails here instead of corrupting data
    /// silently. And after everything is freed, `usage` must be back at zero,
    /// so a chunk that is never released fails the test as well.
    ///
    /// The sequence comes from a fixed seed rather than a random one: a
    /// failure is reproducible and bisectable, which a randomised run would
    /// not be. The step count is reduced under Miri, which needs roughly three
    /// orders of magnitude more time per step.
    #[test]
    fn test_deterministic_allocation_lifecycle() {
        #[cfg(miri)]
        const STEPS: usize = 64;
        #[cfg(not(miri))]
        const STEPS: usize = 512;
        let (mut heap, mut bitmap) =
            helpers::create_heap_and_bitmap_vectors_for::<256>(64);
        let mut allocator =
            helpers::allocator_over::<256>(&mut heap, &mut bitmap);
        let mut seed = 0x5eed_u64;
        let mut live: Vec<helpers::Allocation> = Vec::new();

        for step in 0..STEPS {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let index = (seed as usize) % live.len().max(1);
            match (seed >> 32) % 3 {
                0 if !live.is_empty() => {
                    let allocation = live.swap_remove(index);
                    allocation.assert_pattern(allocation.layout.size());
                    // SAFETY: the record retains the original live allocation
                    // and layout.
                    unsafe {
                        allocator.deallocate(allocation.ptr, allocation.layout)
                    };
                }
                1 if !live.is_empty() => {
                    let mut allocation = live.swap_remove(index);
                    allocation.assert_pattern(allocation.layout.size());
                    let new_size = ((seed >> 8) as usize % 700) + 1;
                    // SAFETY: the record retains the original live allocation
                    // and layout.
                    match unsafe {
                        allocator.realloc(
                            allocation.ptr,
                            allocation.layout,
                            new_size,
                        )
                    } {
                        Ok(ptr) => {
                            let preserved =
                                allocation.layout.size().min(new_size);
                            allocation.ptr = ptr.cast();
                            allocation.layout = Layout::from_size_align(
                                new_size,
                                allocation.layout.align(),
                            )
                            .unwrap();
                            allocation.assert_pattern(preserved);
                            allocation.fill();
                            live.push(allocation);
                        }
                        Err(ChunkAllocatorError::OutOfMemory) => {
                            live.push(allocation)
                        }
                        Err(error) => {
                            panic!("unexpected realloc error: {error:?}")
                        }
                    }
                }
                _ => {
                    let size = ((seed >> 8) as usize % 700) + 1;
                    let alignment =
                        [1, 2, 4, 8, 16, 32, 64, 128, 256][(seed as usize) % 9];
                    let layout =
                        Layout::from_size_align(size, alignment).unwrap();
                    if let Ok(ptr) = allocator.allocate(layout) {
                        let allocation = helpers::Allocation {
                            ptr: ptr.cast(),
                            layout,
                            pattern: step as u8,
                        };
                        allocation.fill();
                        live.push(allocation);
                    }
                }
            }
        }

        for allocation in live {
            allocation.assert_pattern(allocation.layout.size());
            // SAFETY: every remaining record is a live allocation with its
            // layout.
            unsafe { allocator.deallocate(allocation.ptr, allocation.layout) };
        }
        assert_eq!(allocator.usage(), 0.0);
    }
}
