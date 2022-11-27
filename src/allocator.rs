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
//! Module for [`ChunkAllocator`].

use crate::chunk_cache::ChunkCacheEntry;
use crate::compiler_hints::UNLIKELY;
use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

/// Zero sized types may trigger this; according to the Rust doc of the `Allocator`
/// trait this is intended. I work around this by changing the size to 1. This makes
/// core simpler.
macro_rules! normalize_layout {
    ($layout:ident) => {
        if UNLIKELY($layout.size() == 0) {
            core::alloc::Layout::from_size_align(1, $layout.align()).unwrap()
        } else {
            $layout
        }
    };
}

/// Possible errors of [`ChunkAllocator`].
#[derive(Debug, Copy, Clone)]
pub enum ChunkAllocatorError {
    /// The backing memory for the heap must be
    /// - not empty
    /// - an multiple of the used chunk size that is a multiple of 8, and
    /// - not start at 0
    /// - be aligned to the chunk size.
    BadHeapMemory,
    /// The number of bits in the backing memory for the heap bitmap
    /// must match the number of chunks in the heap.
    BadBitmapMemory,
    /// The chunk size must be not 0 and a power of 2.
    BadChunkSize,
    /// The heap is either completely full or to fragmented to serve
    /// the request. Also, it may happen that the alignment can't get
    /// guaranteed, because all aligned chunks are already in use.
    OutOfMemory,
}

/// Default chunk size used by [`ChunkAllocator`]. 256 Bytes is a trade-off between fast
/// allocations and efficient memory usage. However, small allocations will take up at least
/// this amount if bytes.
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Low-level chunk allocator that operates on the provided backing memory. Allocates memory
/// with a variant of the strategies next-fit and best-fit.
///
/// The default chunk size is [`DEFAULT_CHUNK_SIZE`]. A large chunk size has the negative impact
/// that small allocations will consume at least one chunk. A small chunk size has the negative
/// impact that the allocation may take slightly longer.
///
/// As this allocator may allocate more memory than required (because of the chunk size),
/// realloc/grow operations are no-ops in certain cases.
#[derive(Debug)]
pub struct ChunkAllocator<'a, const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE> {
    /// Backing memory for heap.
    heap: &'a mut [u8],
    /// Backing memory for bookkeeping.
    bitmap: &'a mut [u8],
    /// Helper to do some initial initialization on the first runtime invocation.
    is_first_alloc: Cell<bool>,
    /// Contains the next free continuous memory region with a minimum length of one chunk.
    /// It might happen that this entry is invalid because the heap is full or the next chunk
    /// after the previous allocation is already in use.
    ///
    /// This optimization mechanism prevents the need to iterate over all chunks everytime which
    /// can take up to tens of thousands of CPU cycles in the worst case (fragmented heap).
    maybe_next_free_chunk: ChunkCacheEntry,
    /// Counts the number of blocks in use.
    chunks_in_use: usize,
}

impl<'a, const CHUNK_SIZE: usize> ChunkAllocator<'a, CHUNK_SIZE> {
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

    /// Creates a new allocator object. Verifies that the provided memory has the correct properties.
    /// Zeroes the bitmap.
    ///
    /// - heap length must be a multiple of `CHUNK_SIZE`
    /// - the heap must be not empty
    /// - the bitmap must match the number of chunks
    /// - the heap must be at least aligned to CHUNK_SIZE.
    ///
    /// It is recommended that the heap and the bitmap both start at page-aligned addresses for
    /// better performance and to enable a faster search for correctly aligned addresses.
    ///
    /// WARNING: During const initialization it is not possible to check the alignment of the
    /// provided buffer. Please make sure that the data is at least aligned to the chunk_size.
    /// The recommended alignment is page-alignment.
    #[inline]
    pub const fn new(
        heap: &'a mut [u8],
        bitmap: &'a mut [u8],
    ) -> Result<Self, ChunkAllocatorError> {
        if CHUNK_SIZE == 0 {
            return Err(ChunkAllocatorError::BadChunkSize);
        }
        if !CHUNK_SIZE.is_power_of_two() {
            return Err(ChunkAllocatorError::BadChunkSize);
        }

        let heap_starts_at_0 = heap.as_ptr().is_null();
        let heap_is_multiple_of_chunk_size = heap.len() % CHUNK_SIZE == 0;

        if heap.is_empty() || heap_starts_at_0 || !heap_is_multiple_of_chunk_size {
            return Err(ChunkAllocatorError::BadHeapMemory);
        }

        // Warning: This is bleeding Edge Rust Compiler magic and may change in the future
        // see: https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
        let offset = heap.as_ptr().align_offset(CHUNK_SIZE);
        // With Rust 1.61 nightly align_offset always returns usize::MAX in const contexts
        // because there is nothing else it could return in that case. There are no addresses
        // at this point.
        let is_in_const_context = offset == usize::MAX;
        if !is_in_const_context {
            assert!(
                offset == 0,
                "the heap must be at least aligned to CHUNK_SIZE"
            );
        }

        // check bitmap memory has correct length
        let chunk_count = heap.len() / CHUNK_SIZE;

        let chunk_count_is_multiple_of_8 = chunk_count % 8 == 0;
        if !chunk_count_is_multiple_of_8 {
            return Err(ChunkAllocatorError::BadHeapMemory);
        }

        let bitmap_chunk_capacity = bitmap.len() * 8;
        let bitmap_covers_all_chunks_exact = chunk_count == bitmap_chunk_capacity;
        if !bitmap_covers_all_chunks_exact {
            return Err(ChunkAllocatorError::BadBitmapMemory);
        }

        Ok(Self {
            heap,
            bitmap,
            is_first_alloc: Cell::new(true),
            // CHUNK_SIZE is minimal alignment and enforced by the constructor.
            maybe_next_free_chunk: ChunkCacheEntry::new(0, CHUNK_SIZE, chunk_count),
            chunks_in_use: 0,
        })
    }

    /// Version of [`Self::new`] that panics instead of returning a result. Useful for globally
    /// static const contexts. The panic will happen during compile time and not during run time.
    /// [`Self::new`] can't be used in such scenarios because `unwrap()` on the
    /// Result is not a const function (yet).
    ///
    /// WARNING: During const initialization it is not possible to check the alignment of the
    /// provided buffer. Please make sure that the data is at least aligned to the chunk_size.
    /// The recommended alignment is page-alignment.
    #[inline]
    pub const fn new_const(heap: &'a mut [u8], bitmap: &'a mut [u8]) -> Self {
        assert!(CHUNK_SIZE > 0, "chunk size must not be zero!");
        assert!(
            CHUNK_SIZE.is_power_of_two(),
            "chunk size must be a power of two!"
        );

        let heap_starts_at_0 = heap.as_ptr().is_null();
        let heap_is_multiple_of_chunk_size = heap.len() % CHUNK_SIZE == 0;

        assert!(
            !heap.is_empty() && !heap_starts_at_0 && heap_is_multiple_of_chunk_size,
            "heap must be not empty and a multiple of the chunk size"
        );

        // Warning: This is bleeding Edge Rust Compiler magic and may change in the future
        // see: https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
        let offset = heap.as_ptr().align_offset(CHUNK_SIZE);
        let is_in_const_context = offset == usize::MAX;
        if !is_in_const_context {
            assert!(
                offset == 0,
                "the heap must be at least aligned to CHUNK_SIZE"
            );
        }

        // check bitmap memory has correct length
        let chunk_count = heap.len() / CHUNK_SIZE;

        let chunk_count_is_multiple_of_8 = chunk_count % 8 == 0;
        assert!(
            chunk_count_is_multiple_of_8,
            "chunk count must be a multiple of 8"
        );

        let bitmap_chunk_capacity = bitmap.len() * 8;
        let bitmap_covers_all_chunks_exact = chunk_count == bitmap_chunk_capacity;
        assert!(
            bitmap_covers_all_chunks_exact,
            "the bitmap must cover the amount of chunks exactly"
        );

        Self {
            heap,
            bitmap,
            is_first_alloc: Cell::new(true),
            // I can't enforce CHUNK_SIZE is minimal alignment here because this does not work in
            // const contexts: see https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
            // Current workaround: Do it lazily on the first allocation.
            maybe_next_free_chunk: ChunkCacheEntry::new(0, 1, chunk_count),
            chunks_in_use: 0,
        }
    }

    /// Capacity in bytes of the allocator.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.heap.len()
    }

    /// Returns number of chunks.
    #[inline]
    pub const fn chunk_count(&self) -> usize {
        // size is a multiple of CHUNK_SIZE;
        // ensured in new()
        self.capacity() / CHUNK_SIZE
    }

    /// Returns the current memory usage in percentage rounded to two decimal places.
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
    /// - `chunk_index` describes the start chunk; i.e. the search space inside the backing storage
    #[inline(always)]
    fn chunk_is_free(&self, chunk_index: usize) -> bool {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index={} is bigger than max chunk index={}",
            chunk_index,
            self.chunk_count() - 1
        );
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        let relevant_bit = (self.bitmap[byte_i] >> bit) & 1;
        relevant_bit == 0
    }

    /// Marks a chunk as used, i.e. write a 1 into the bitmap at the right position.
    #[inline(always)]
    fn mark_chunk_as_used(&mut self, chunk_index: usize) {
        debug_assert!(chunk_index < self.chunk_count());
        if UNLIKELY(!self.chunk_is_free(chunk_index)) {
            panic!(
                "tried to mark chunk {} as used but it is already used",
                chunk_index
            );
        }
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // xor => keep all bits, except bitflip at relevant position
        self.bitmap[byte_i] ^= 1 << bit;
    }

    /// Marks a chunk as free, i.e. write a 0 into the bitmap at the right position.
    #[inline(always)]
    fn mark_chunk_as_free(&mut self, chunk_index: usize) {
        debug_assert!(chunk_index < self.chunk_count());
        if UNLIKELY(self.chunk_is_free(chunk_index)) {
            panic!(
                "tried to mark chunk {} as free but it is already free",
                chunk_index
            );
        }
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // xor => keep all bits, except bitflip at relevant position
        let updated_byte = self.bitmap[byte_i] ^ (1 << bit);
        self.bitmap[byte_i] = updated_byte;
    }

    /// Returns the indices into the bitmap array of a given chunk index.
    #[inline(always)]
    fn chunk_index_to_bitmap_indices(&self, chunk_index: usize) -> (usize, usize) {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index out of range!"
        );
        (chunk_index / 8, chunk_index % 8)
    }

    /// Finds the next available continuous memory region, i.e. coherent available/free chunks.
    /// Returns the beginning index. Does not mark them as used. This is the responsibility
    /// of the caller.
    ///
    /// Uses [`Self::find_next_free_aligned_chunk_by_index`] as helper method.
    ///
    /// # Parameters
    /// - `chunk_num_request` number of chunks that must be all free without gaps in-between; greater than 0
    /// - `alignment` required alignment of the chunk in memory. Must be a power of 2. This usually
    ///               comes from [`core::alloc::Layout`] which already guarantees that it is a power
    ///               of two.
    #[inline(always)]
    fn find_free_continuous_memory_region(
        &mut self,
        chunk_num_request: usize,
        alignment: usize,
    ) -> Result<usize, ChunkAllocatorError> {
        if UNLIKELY(chunk_num_request > self.chunk_count()) {
            // out of memory
            return Err(ChunkAllocatorError::OutOfMemory);
        }

        // We hope that the index and its succeeding chunks stored in the cache fits the requested
        // memory region.
        let start_index = self.maybe_next_free_chunk.index();

        (start_index..(start_index + self.chunk_count()))
            // Cope with wrapping indices (i.e. index 0 follows 31).
            // This will lead to scenarios where it iterates like: 4,5,6,7,0,1,2,3
            // (assuming there are 8 chunks).
            .map(|index| index % self.chunk_count())
            // It only makes sense to start the lookup at chunks that are available.
            .filter(|chunk_index| self.chunk_is_free(*chunk_index))
            // If the heap has 8 chunks and we need 4 but start the search at index 6, then we
            // don't have enough continuous chunks to fulfill the request. Thus, we skip those.
            .filter(|chunk_index| {
                // example: index=0 + request=4 with count=4  => is okay
                *chunk_index + chunk_num_request <= self.chunk_count()
            })
            // ALIGNMENT CHECK BEGIN
            .map(|chunk_index| (chunk_index, unsafe { self.chunk_index_to_ptr(chunk_index) }))
            .filter(|(_, addr)| addr.align_offset(alignment) == 0)
            .map(|(chunk_index, _)| chunk_index)
            // ALIGNMENT CHECK END
            //
            // Now look for the continuous region: are all succeeding chunks free?
            // This is safe because earlier I skipped chunk_indices that are too close to
            // the end. Return the first result.
            .find(|chunk_index| {
                // +1: chunk at chunk_index itself is already free (we checked this earlier)
                // inclusive
                let from = chunk_index + 1;
                // -1: indices start at 0
                // exclusive
                let to = from + chunk_num_request - 1;

                (from..to).all(|index| self.chunk_is_free(index))
            })
            // OK or out of memory
            .ok_or(ChunkAllocatorError::OutOfMemory)
    }

    /// Returns the pointer to the beginning of the chunk.
    #[inline(always)]
    unsafe fn chunk_index_to_ptr(&self, chunk_index: usize) -> *mut u8 {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index out of range!"
        );
        self.heap.as_ptr().add(chunk_index * CHUNK_SIZE) as *mut u8
    }

    /// Returns the chunk index of the given pointer (which points to the beginning of a chunk).
    #[inline(always)]
    unsafe fn ptr_to_chunk_index(&self, ptr: *const u8) -> usize {
        let heap_begin_inclusive = self.heap.as_ptr();
        let heap_end_exclusive = self.heap.as_ptr().add(self.heap.len());
        debug_assert!(
            heap_begin_inclusive <= ptr && ptr < heap_end_exclusive,
            "pointer {:?} is out of range {:?}..{:?} of the allocators backing storage",
            ptr,
            heap_begin_inclusive,
            heap_end_exclusive
        );
        (ptr as usize - heap_begin_inclusive as usize) / CHUNK_SIZE
    }

    /// Calculates the number of required chunks to fulfill an allocation request.
    #[inline(always)]
    const fn calc_required_chunks(&self, size: usize) -> usize {
        assert!(size > 0);
        if size % CHUNK_SIZE == 0 {
            size / CHUNK_SIZE
        } else {
            (size / CHUNK_SIZE) + 1
        }
    }

    /// Performs initialization steps on the first allocation.
    /// - checks heap memory (alignment etc) because this can't be done during const new
    ///   initialization
    /// - zeroes the bitmap
    fn init(&mut self) -> Result<(), ChunkAllocatorError> {
        self.is_first_alloc.replace(false);
        // Zero bitmap
        self.bitmap.fill(0);

        if self.heap.as_ptr().align_offset(4096) != 0 && CHUNK_SIZE < 4096 {
            log::debug!(
                "It is recommended to use the allocator with page-aligned(!) backing memory for the heap."
            );
        }

        // this can't be done in const new constructor
        // see: https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248
        if self.heap.as_ptr().align_offset(self.min_alignment()) != 0 {
            log::error!("The heap is not aligned to at least CHUNK_SIZE. Recommended alignment is 4096 (page-alignment).");
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
    #[must_use = "The pointer must be used and freed eventually to prevent memory leaks."]
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, ChunkAllocatorError> {
        log::trace!("called allocate");
        if UNLIKELY(self.is_first_alloc.get()) {
            self.init()?;
        }

        let layout = normalize_layout!(layout);

        let required_chunks = self.calc_required_chunks(layout.size());

        let index = self.find_free_continuous_memory_region(required_chunks, layout.align());

        if UNLIKELY(index.is_err()) {
            log::warn!(
                "Out of Memory. Can't fulfill the requested layout: {:?}. Current usage is: {}%/{}byte",
                layout,
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

        // Only update "maybe_next_free_chunk" if it doesn't already point to a free location;
        // For example, it could be that it was not used in this allocation.
        //
        // MAKE SURE THIS GETS CALLED AFTER USED CHUNKS ARE MARKED AS SUCH EARLIER.
        if !self.chunk_is_free(self.maybe_next_free_chunk.index()) {
            // at next allocation: continue search at this index
            let next_index = (index + 1) % self.chunk_count();
            // - alignment of chunk_size is always guaranteed.
            // - We do not know yet if the next entry is actually available. We just give the
            //   algorithm an hint where to start for the next search.
            self.maybe_next_free_chunk = ChunkCacheEntry::new(next_index, CHUNK_SIZE, 1);
        }

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

    /// Deallocates the given pointer.
    ///
    /// # Safety
    /// Unsafe if memory gets de-allocated that is still in use.
    #[track_caller]
    #[inline]
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        log::trace!("called deallocate");
        let layout = normalize_layout!(layout);

        let freed_chunks = self.calc_required_chunks(layout.size());

        log::trace!("dealloc: layout={:?}, #chunks={})", layout, freed_chunks);

        let index = self.ptr_to_chunk_index(ptr.as_ptr());
        for i in index..index + freed_chunks {
            self.mark_chunk_as_free(i);
        }
        self.chunks_in_use -= freed_chunks;

        // This helps the next allocation to be faster because we know that this block was just
        // freed. This only works if the next allocation fits into the continuous region of memory.
        //
        // Currently, this prefers the smallest possible continuous region with the lowest
        // possible alignment which prevents fragmentation at the cost of larger lookup times. It
        // assumes/hopes the next allocation only needs as few chunks as possible
        // (ideally a fitting one).
        //
        // Small alignments (1, 2, 4, 8) are common but 4096 (page-alignment) is rather rare.
        // Therefore, it is okay to try to prevent small allocations in addresses with big
        // alignment.

        // 1) freed memory region smaller then cached?
        if freed_chunks < self.maybe_next_free_chunk.chunk_count()
            // 2) if same size: check alignment
            || (freed_chunks == self.maybe_next_free_chunk.chunk_count()
                // does the layout alignment is bigger than the minimum alignment
                // and smaller than the one currently cached
                && layout.align() > CHUNK_SIZE
                && layout.align() < self.maybe_next_free_chunk.alignment())
        {
            self.maybe_next_free_chunk = ChunkCacheEntry::new(index, layout.align(), freed_chunks);
        }
    }

    /// Reallocs the memory. This might be a cheap operation if the new size is still smaller or
    /// equal to the chunk size. Otherwise, this falls back to the default implementation of the
    /// Global allocator from Rust.
    ///
    /// # Safety
    /// Unsafe if memory gets de-allocated that is still in use.
    #[track_caller]
    #[inline]
    pub unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<[u8]>, ChunkAllocatorError> {
        log::trace!("called realloc");

        // zero sized types may trigger this; according to the Rust doc of the `Allocator`
        // trait this is intended. I work around this by changing the size to 1.
        let old_layout = normalize_layout!(old_layout);

        let required_chunks = self.calc_required_chunks(old_layout.size());
        let occupied_size = required_chunks * CHUNK_SIZE;

        // fast return: reuse existing allocation as it is big enough
        if new_size <= occupied_size {
            log::trace!("realloc fast return possible!");
            Ok(NonNull::slice_from_raw_parts(ptr, new_size))
        } else {
            log::trace!("realloc fast return NOT possible!");

            // SAFETY: the caller must ensure that the `new_size` does not overflow.
            // `layout.align()` comes from a `Layout` and is thus guaranteed to be valid.
            let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());
            // SAFETY: the caller must ensure that `new_layout` is greater than zero.
            let new_ptr = self.allocate(new_layout)?;

            // SAFETY: the previously allocated block cannot overlap the newly allocated block.
            // The safety contract for `dealloc` must be upheld by the caller.
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_ptr.as_mut_ptr(),
                core::cmp::min(old_layout.size(), new_size),
            );
            self.deallocate(ptr, old_layout);

            Ok(new_ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::tests::helpers::GlobalPageAlignedAlloc;
    use crate::PageAligned;
    use std::alloc::{AllocError, Allocator, Global};
    use std::cmp::max;
    use std::ptr::NonNull;
    use std::vec::Vec;

    mod helpers {

        use super::*;

        /// Helper struct to let the std vector align stuff at a page boundary.
        /// Forwards requests to the global Rust allocator provided by the standard library.
        pub struct GlobalPageAlignedAlloc;

        unsafe impl Allocator for GlobalPageAlignedAlloc {
            fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                let alignment = max(layout.align(), 4096);
                // unwrap should never fail, because layout.align() is already a power
                // of 2, otherwise the value not exist here.
                let layout = layout.align_to(alignment).unwrap();
                Global.allocate(layout)
            }

            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                let alignment = max(layout.align(), 4096);
                // unwrap should never fail, because layout.align() is already a power
                // of 2, otherwise the value not exist here.
                let layout = layout.align_to(alignment).unwrap();
                Global.deallocate(ptr, layout)
            }
        }

        /// Creates backing memory for the allocator and the bitmap management structure.
        /// Uses the std global allocator for this. The memory is page-aligned.
        pub fn create_heap_and_bitmap_vectors() -> (
            Vec<u8, GlobalPageAlignedAlloc>,
            Vec<u8, GlobalPageAlignedAlloc>,
        ) {
            // 32 chunks with default chunk size = 256 bytes = 2 pages = 2*4096
            const CHUNK_COUNT: usize = 32;
            const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
            let mut heap = Vec::with_capacity_in(HEAP_SIZE, GlobalPageAlignedAlloc);
            (0..heap.capacity()).for_each(|_| heap.push(0));
            const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE / 8;
            let mut heap_bitmap = Vec::with_capacity_in(BITMAP_SIZE, GlobalPageAlignedAlloc);
            (0..heap_bitmap.capacity()).for_each(|_| heap_bitmap.push(0));

            assert_eq!(
                heap.as_ptr().align_offset(DEFAULT_CHUNK_SIZE),
                0,
                "heap must be at least allocated to CHUNK_SIZE"
            );
            assert_eq!(heap.as_ptr().align_offset(4096), 0, "must be page aligned");
            assert_eq!(
                heap_bitmap.as_ptr().align_offset(4096),
                0,
                "must be page aligned"
            );

            (heap, heap_bitmap)
        }
    }

    /// Initializes the allocator with illegal chunk sizes.
    #[test]
    fn test_new_fails_illegal_chunk_size() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();

        let msg = "expected panic because of bad chunk size (is 0 which is illegal)";
        assert!(
            matches!(
                ChunkAllocator::<0>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
                ChunkAllocatorError::BadChunkSize
            ),
            "{}",
            msg
        );
        std::panic::catch_unwind(|| {
            let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
            ChunkAllocator::<0>::new_const(&mut heap, &mut heap_bitmap);
        })
        .expect_err(msg);
        // ------------------------------------------------------------------------------------
        let msg = "expected panic because of bad chunk size (is not a power of 2 which is illegal)";
        assert!(
            matches!(
                ChunkAllocator::<3>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
                ChunkAllocatorError::BadChunkSize
            ),
            "{}",
            msg
        );
        std::panic::catch_unwind(|| {
            let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
            ChunkAllocator::<3>::new_const(&mut heap, &mut heap_bitmap);
        })
        .expect_err(msg);

        // ------------------------------------------------------------------------------------
        let msg = "expected panic because of bad bitmap memory: heap is not big enough for the number of chunks";
        assert!(
            matches!(
                ChunkAllocator::<512>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
                ChunkAllocatorError::BadBitmapMemory
            ),
            "{}",
            msg
        );
        std::panic::catch_unwind(|| {
            let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
            ChunkAllocator::<512>::new_const(&mut heap, &mut heap_bitmap);
        })
        .expect_err(msg);

        // ------------------------------------------------------------------------------------
        let msg = "expected panic because of bad bitmap memory";
        assert!(
            matches!(
                ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut [0]).unwrap_err(),
                ChunkAllocatorError::BadBitmapMemory
            ),
            "{}",
            msg
        );
        std::panic::catch_unwind(|| {
            let (mut heap, _) = helpers::create_heap_and_bitmap_vectors();
            ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new_const(&mut heap, &mut [0]);
        })
        .expect_err(msg);
    }

    /// Initializes the allocator with backing memory gained on the heap.
    #[test]
    fn test_compiles_dynamic() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();

        // check that it compiles
        let mut _alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();
    }

    /// Initializes the allocator with backing memory that is static inside the binary.
    /// This is available during compilation time, i.e. tests that the constructor
    /// is a "const fn".
    #[test]
    fn test_compiles_const() {
        // must be a multiple of 8
        const CHUNK_COUNT: usize = 16;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        static mut HEAP: PageAligned<[u8; HEAP_SIZE]> = PageAligned::new([0; HEAP_SIZE]);
        const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE / 8;
        static mut HEAP_BITMAP: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];

        // check that it compiles
        let mut _alloc: ChunkAllocator =
            unsafe { ChunkAllocator::new(HEAP.deref_mut_const(), &mut HEAP_BITMAP).unwrap() };
    }

    /// Test looks if the allocator ensures that the required chunk count to manage the backing
    /// memory matches the size of the bitmap. Tests the method `chunk_count()`.
    #[test]
    fn test_chunk_count_matches_bitmap() {
        // At minimum there must be 8 chunks that get managed by a bitmap of a size of 1 byte.
        let min_chunk_count = 8;

        // - step by 8 => heap size must be dividable by 8 for the bitmap.
        // - limit 128 chosen arbitrary
        for chunk_count in (min_chunk_count..128).step_by(8) {
            let heap_size: usize = chunk_count * DEFAULT_CHUNK_SIZE;
            let mut heap = Vec::new_in(GlobalPageAlignedAlloc);
            (0..heap_size).for_each(|_| heap.push(0));

            let bitmap_size_exact = if chunk_count % 8 == 0 {
                chunk_count / 8
            } else {
                (chunk_count / 8) + 1
            };
            let mut bitmap = vec![0_u8; bitmap_size_exact];
            let alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut bitmap).unwrap();
            assert_eq!(chunk_count, alloc.chunk_count(),);
        }
    }

    /// Test looks if the allocator ensures that the allocator can not get constructed, if the
    /// bitmap size does not match the chunks perfectly.
    #[test]
    fn test_alloc_new_fails_when_bitmap_doesnt_match() {
        // - skip every 8th element. Hence, the chunk count will not be a multiple of 8.
        // - limit 128 chosen arbitrary
        for chunk_count in (0..128).filter(|chunk_count| *chunk_count % 8 != 0) {
            let heap_size: usize = chunk_count * DEFAULT_CHUNK_SIZE;
            let mut heap = Vec::new_in(GlobalPageAlignedAlloc);
            (0..heap_size).for_each(|_| heap.push(0));
            let bitmap_size_exact = if chunk_count % 8 == 0 {
                chunk_count / 8
            } else {
                (chunk_count / 8) + 1
            };
            let mut bitmap = vec![0_u8; bitmap_size_exact];
            let alloc = ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut bitmap);
            assert!(
                alloc.is_err(),
                "new() must fail, because the bitmap can not exactly cover the available chunks"
            );
        }
    }

    /// Tests the method `chunk_index_to_bitmap_indices()`.
    #[test]
    fn test_chunk_index_to_bitmap_indices() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

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
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        heap_bitmap[0] = 0x2f;
        let alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

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
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let heap_ptr = heap.as_ptr();
        let alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

        unsafe {
            assert_eq!(heap_ptr, alloc.chunk_index_to_ptr(0));
            assert_eq!(
                heap_ptr.add(alloc.chunk_size() * 1),
                alloc.chunk_index_to_ptr(1)
            );
            assert_eq!(
                heap_ptr.add(alloc.chunk_size() * 7),
                alloc.chunk_index_to_ptr(7)
            );
        }
    }

    /// Test to get single chunks of memory. Tests `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_basic() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

        // I made this test for these two properties. Test might need to get adjusted if they change
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        assert_eq!(
            0,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(0);
        alloc.maybe_next_free_chunk = ChunkCacheEntry::new(1, DEFAULT_CHUNK_SIZE, 1);

        assert_eq!(1, alloc.find_free_continuous_memory_region(1, 1).unwrap());
        alloc.maybe_next_free_chunk = ChunkCacheEntry::new(2, DEFAULT_CHUNK_SIZE, 1);
        assert_eq!(
            // 16: 256*16 = 4096 => second page in heap mem that consists of two pages
            16,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(16);
        // makes sure the next search
        alloc.maybe_next_free_chunk = ChunkCacheEntry::new(17, DEFAULT_CHUNK_SIZE, 1);

        assert!(
            alloc.find_free_continuous_memory_region(1, 4096).is_err(),
            "out of memory; only 2 pages of memory"
        );

        // now free the first chunk again, which enables a further 4096 byte aligned allocation
        alloc.mark_chunk_as_free(0);
        assert_eq!(
            0,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
    }

    /// Test to get a continuous region of memory. Tests `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_full_1() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

        // I made this test for these two properties. Test might need to get adjusted if they change
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

    /// Test to get a continuous region of memory. Tests `find_free_continuous_memory_region()`.
    #[test]
    fn test_find_free_continuous_memory_region_full_2() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let mut alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut heap_bitmap).unwrap();

        // I made this test for these two properties. Test might need to get adjusted if they change
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        alloc.mark_chunk_as_used(0);
        alloc.mark_chunk_as_used(1);
        alloc.mark_chunk_as_used(2);
        alloc.mark_chunk_as_used(16);

        assert!(alloc.find_free_continuous_memory_region(
            1,
            4096).is_err(),
                "out of memory! chunks 0 and 16 are occupied; the only available page-aligned addresses"
        );
        assert_eq!(17, alloc.find_free_continuous_memory_region(15, 1).unwrap(),);
    }
}
