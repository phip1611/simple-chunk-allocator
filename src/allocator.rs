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
use core::cell::OnceCell;
use core::error;
use core::fmt;
use core::ptr::{self, NonNull};
use log::{trace, warn};

/// No free run of chunks can satisfy a request.
///
/// A heap with room left can still report this: the run must be free as a
/// whole, and an alignment above `CHUNK_SIZE` also limits where it may start.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OutOfMemory;

impl fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("no free run of chunks satisfies the request")
    }
}

impl error::Error for OutOfMemory {}

/// Default chunk size: 256 bytes.
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Rounds a zero-sized layout up to one byte.
///
/// The `Allocator` documentation calls zero-sized requests intended, but a
/// chunk allocator cannot express "no memory": it would have to hand out zero
/// chunks and could not tell the matching deallocation apart from a real one.
/// Charging one chunk for it keeps every path below dealing with whole chunks.
#[inline]
fn normalize_layout(layout: Layout) -> Layout {
    if layout.size() == 0 {
        Layout::from_size_align(1, layout.align())
            .expect("one byte should fit a layout that already exists")
    } else {
        layout
    }
}

/// Where the chunks and their bitmap sit inside the caller's region.
///
/// This cannot be computed by the constructor: the padding depends on the
/// region's address, and const evaluation cannot observe one. Once derived it
/// never changes, so the pointers are resolved here rather than recomputed at
/// every access.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// First chunk, `CHUNK_SIZE`-aligned - what the padding buys, depending
    /// on the alignment of the region handed in.
    heap: NonNull<u8>,
    /// First bitmap byte, immediately behind the last chunk.
    bitmap: NonNull<u8>,
    /// Number of chunks that fit next to their own bitmap.
    chunk_count: usize,
}

// SAFETY: both pointers address the region that the allocator holding this
// geometry owns exclusively, and they stay valid wherever it is moved to.
unsafe impl Send for Geometry {}

/// Allocates from a single caller-provided memory region in fixed-size chunks.
///
/// The region holds both the chunks and the bitmap that tracks them; the
/// bitmap occupies the tail. Each allocation consumes whole chunks. A larger
/// chunk size reduces bitmap size and search work; a smaller size reduces
/// internal fragmentation.
///
/// The allocator aligns the chunks itself, so the region may start anywhere.
/// See [`Self::new`] for what that costs.
///
/// This type is not synchronized: every operation takes `&mut self`. Use
/// [`crate::GlobalChunkAllocator`] to share one.
///
/// # Example
/// ```rust
/// use core::alloc::Layout;
/// use simple_chunk_allocator::ChunkAllocator;
///
/// let mut region = [0_u8; 4096];
/// // SAFETY: `region` outlives the allocator and nothing else touches it.
/// let mut allocator = unsafe {
///     ChunkAllocator::<256>::new(region.as_mut_ptr(), region.len())
/// };
///
/// let layout = Layout::from_size_align(64, 8).unwrap();
/// let allocation = allocator.allocate(layout).unwrap();
/// assert!(allocator.usage() > 0.0);
///
/// // SAFETY: `allocation` is live and paired with its original layout.
/// unsafe { allocator.deallocate(allocation.cast(), layout) };
/// assert_eq!(allocator.usage(), 0.0);
/// ```
#[derive(Debug)]
pub struct ChunkAllocator<const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE> {
    /// Start of the caller-provided region, of any alignment.
    region: NonNull<u8>,
    /// Length of `region` in bytes.
    region_len: usize,
    /// Derived from `region` on first use; see [`Self::geometry`].
    geometry: OnceCell<Geometry>,
    /// Contains the next free continuous memory region with a minimum length
    /// of one chunk. It might happen that this entry is invalid because
    /// the heap is full or the next chunk after the previous allocation is
    /// already in use.
    /// Whether the bitmap has been cleared; see
    /// [`Self::initialize_bitmap`].
    bitmap_is_initialized: bool,
    /// Where the next search starts. Only a hint: the chunk may well be in
    /// use, and the search re-verifies everything it looks at.
    ///
    /// This optimization mechanism prevents the need to iterate over all
    /// chunks everytime which can take up to tens of thousands of CPU
    /// cycles in the worst case (fragmented heap).
    maybe_next_free_chunk: ChunkCacheEntry,
    /// Counts the number of blocks in use.
    chunks_in_use: usize,
}

// SAFETY: the allocator is the exclusive owner of its backing region, and its
// raw pointer stays valid wherever the allocator is moved to.
unsafe impl<const CHUNK_SIZE: usize> Send for ChunkAllocator<CHUNK_SIZE> {}

impl<const CHUNK_SIZE: usize> ChunkAllocator<CHUNK_SIZE> {
    /// Rejects an invalid chunk size when an allocator is built.
    ///
    /// A power of two is what makes every chunk `CHUNK_SIZE`-aligned once the
    /// heap base is. Zero is covered as well, because `0` is not a power of
    /// two.
    ///
    /// [`Self::new`] is the only place that needs to evaluate this: it is the
    /// only way to obtain an allocator, so no invalid chunk size reaches the
    /// allocating code without passing through it.
    const VALIDATE_CHUNK_SIZE: () = assert!(
        CHUNK_SIZE.is_power_of_two(),
        "CHUNK_SIZE must be a power of two"
    );

    /// Returns the used chunk size.
    #[inline]
    pub const fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }

    /// Returns the alignment every chunk is guaranteed to have.
    ///
    /// An allocation asking for no more than this fits into any free chunk.
    /// A larger alignment is met by every `alignment / CHUNK_SIZE`-th chunk.
    /// Which ones those are shifts with the region, but they exist wherever it
    /// starts, so such a request needs a free run in the right place rather
    /// than a particular alignment of the region.
    pub const fn min_alignment(&self) -> usize {
        CHUNK_SIZE
    }

    /// Creates an allocator over a single caller-provided memory region.
    ///
    /// The region holds the chunks followed by their bitmap; how many fit
    /// follows from `region_len` and `CHUNK_SIZE`. Anything below
    /// [`Self::required_region_size`] for one chunk is rejected: such a region
    /// could only ever produce an allocator that is out of memory on every
    /// request.
    ///
    /// The region needs neither a particular alignment nor initialized
    /// content. The allocator aligns the first chunk itself, which makes every
    /// chunk `CHUNK_SIZE`-aligned at a cost of up to `CHUNK_SIZE - 1` bytes;
    /// a region that is already `CHUNK_SIZE`-aligned - page-aligned is the
    /// easiest way - avoids that. It is not needed for alignment to work, see
    /// [`Self::min_alignment`].
    ///
    /// # Safety
    /// `region` must be non-null and point to `region_len` writable bytes for
    /// as long as the allocator lives, and the allocator must be their only
    /// user for that time.
    #[inline]
    pub const unsafe fn new(region: *mut u8, region_len: usize) -> Self {
        let () = Self::VALIDATE_CHUNK_SIZE;

        assert!(
            region_len >= Self::required_region_size(1),
            "the region must be able to hold at least one chunk"
        );

        Self {
            region: NonNull::new(region)
                .expect("caller should pass a non-null region"),
            region_len,
            geometry: OnceCell::new(),
            // Chunk 0 is CHUNK_SIZE-aligned by construction. The length is the
            // conservative minimum; the first allocation replaces the hint
            // with a real one anyway.
            maybe_next_free_chunk: ChunkCacheEntry::new(0, CHUNK_SIZE, 1),
            bitmap_is_initialized: false,
            chunks_in_use: 0,
        }
    }

    /// Returns the region size that holds at least `chunk_count` chunks,
    /// whatever alignment the region turns out to have.
    ///
    /// Sizing backing storage with this is what keeps a caller from silently
    /// getting fewer chunks than planned.
    ///
    /// # Example
    /// ```rust
    /// use simple_chunk_allocator::ChunkAllocator;
    ///
    /// const SIZE: usize = ChunkAllocator::<256>::required_region_size(64);
    /// static mut REGION: [u8; SIZE] = [0; SIZE];
    ///
    /// // SAFETY: nothing else uses REGION.
    /// let allocator = unsafe {
    ///     ChunkAllocator::<256>::new((&raw mut REGION).cast(), SIZE)
    /// };
    /// assert!(allocator.chunk_count() >= 64);
    /// ```
    #[inline]
    pub const fn required_region_size(chunk_count: usize) -> usize {
        let chunks = chunk_count * CHUNK_SIZE;
        // One bit per chunk, rounded up: the bitmap is addressed in bytes.
        let bitmap = chunk_count.div_ceil(8);
        // The allocator skips up to a chunk to align the first one, so an
        // exactly sized region would come up short unless it happens to be
        // chunk-aligned.
        let alignment_padding = CHUNK_SIZE - 1;

        chunks + bitmap + alignment_padding
    }

    /// Returns the region layout, deriving it on the first call.
    ///
    /// Pure apart from the caching. Clearing the bitmap behind it is
    /// [`Self::initialize_bitmap`]'s job.
    #[inline]
    fn geometry(&self) -> Geometry {
        *self.geometry.get_or_init(|| {
            // The padding can exceed the region, and `align_offset` may even
            // give up and return `usize::MAX`. Clamping keeps every derived
            // pointer inside the region; the chunk area then holds nothing.
            let heap_offset = self
                .region
                .as_ptr()
                .align_offset(CHUNK_SIZE)
                .min(self.region_len);
            let available = self.region_len - heap_offset;

            // A chunk costs CHUNK_SIZE bytes plus one bitmap bit, so eight of
            // them cost a whole number of bytes. Counting in groups of eight
            // and then in eighths of a byte keeps the result exact without
            // overflowing on `available * 8`.
            let bytes_per_eight_chunks = 8 * CHUNK_SIZE + 1;
            let whole_groups = available / bytes_per_eight_chunks;
            let leftover = available % bytes_per_eight_chunks;
            let chunk_count =
                whole_groups * 8 + (leftover * 8) / bytes_per_eight_chunks;

            // SAFETY: both offsets are within the region, and the bitmap
            // directly follows the last chunk.
            unsafe {
                Geometry {
                    heap: self.region.add(heap_offset),
                    bitmap: self
                        .region
                        .add(heap_offset + chunk_count * CHUNK_SIZE),
                    chunk_count,
                }
            }
        })
    }

    /// Clears the bitmap, once.
    ///
    /// The caller's region may hold anything, so the bits have to be zeroed
    /// before they are read as chunk state. Only [`Self::allocate`] needs to
    /// call this: a chunk cannot be freed or resized before it was handed out.
    #[inline]
    fn initialize_bitmap(&mut self) {
        if self.bitmap_is_initialized {
            return;
        }
        self.bitmap_is_initialized = true;

        let geometry = self.geometry();
        // One bit per chunk, rounded up to whole bytes.
        let bitmap_len = geometry.chunk_count.div_ceil(8);
        // SAFETY: the bitmap lies inside the region, which the allocator owns
        // exclusively, and no chunk has been handed out yet.
        unsafe { ptr::write_bytes(geometry.bitmap.as_ptr(), 0, bitmap_len) };
    }

    /// Returns the pointer to the first chunk.
    #[inline(always)]
    fn heap_ptr(&self) -> *mut u8 {
        self.geometry().heap.as_ptr()
    }

    /// Returns the pointer to the first bitmap byte.
    #[inline(always)]
    fn bitmap_ptr(&self) -> *mut u8 {
        self.geometry().bitmap.as_ptr()
    }

    /// Usable capacity in bytes, i.e. without the bitmap and any padding the
    /// allocator had to skip to align the first chunk.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.chunk_count() * CHUNK_SIZE
    }

    /// Returns number of chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.geometry().chunk_count
    }

    /// Returns the share of chunks in use, between `0.0` and `1.0`.
    ///
    /// The zero case is special-cased because an allocator without capacity
    /// would otherwise divide by zero.
    #[inline]
    pub fn usage(&self) -> f32 {
        if self.chunks_in_use == 0 {
            0.0
        } else {
            self.chunks_in_use as f32 / self.chunk_count() as f32
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
            unsafe { (*self.bitmap_ptr().add(byte_i) >> bit) & 1 };
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
        unsafe { *self.bitmap_ptr().add(byte_i) ^= 1 << bit };
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
            let byte = self.bitmap_ptr().add(byte_i);
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
        &self,
        chunk_num_request: usize,
        alignment: usize,
    ) -> Result<usize, OutOfMemory> {
        if chunk_num_request > self.chunk_count() {
            // out of memory
            return Err(OutOfMemory);
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

                // Does the chunk sit at an address that satisfies the
                // requested alignment?
                let ptr = self.chunk_index_to_ptr(chunk_index);
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
            .ok_or(OutOfMemory)
    }

    /// Returns the pointer to the beginning of the chunk.
    #[inline(always)]
    fn chunk_index_to_ptr(&self, chunk_index: usize) -> *mut u8 {
        debug_assert!(
            chunk_index < self.chunk_count(),
            "chunk_index out of range!"
        );
        // SAFETY: `chunk_index` is bounded by `chunk_count`.
        unsafe { self.heap_ptr().add(chunk_index * CHUNK_SIZE) }
    }

    /// Returns the chunk index of the given pointer (which points to the
    /// beginning of a chunk).
    #[inline(always)]
    unsafe fn ptr_to_chunk_index(&self, ptr: *const u8) -> usize {
        let heap_begin_inclusive = self.heap_ptr().cast_const();
        // SAFETY: `capacity` is the length of the chunk area.
        let heap_end_exclusive =
            unsafe { self.heap_ptr().add(self.capacity()) };
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

    /// Allocates memory according to the specific layout.
    #[track_caller]
    #[inline]
    #[must_use = "The pointer must be freed eventually to prevent memory leaks."]
    pub fn allocate(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, OutOfMemory> {
        trace!("called allocate");
        self.initialize_bitmap();
        let layout = normalize_layout(layout);

        let required_chunks = self.calc_required_chunks(layout.size());

        let index = self.find_free_continuous_memory_region(
            required_chunks,
            layout.align(),
        );

        if index.is_err() {
            warn!(
                "Out of memory for {layout:?}; {}/{} chunks in use",
                self.chunks_in_use,
                self.chunk_count()
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

        let heap_ptr = self.chunk_index_to_ptr(index);
        trace!(
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
        trace!("called deallocate");
        let layout = normalize_layout(layout);

        let freed_chunks = self.calc_required_chunks(layout.size());

        trace!("dealloc: layout={:?}, #chunks={})", layout, freed_chunks);

        // SAFETY: callers must pass a pointer returned by this allocator.
        let index = unsafe { self.ptr_to_chunk_index(ptr.as_ptr()) };
        for i in index..index + freed_chunks {
            self.mark_chunk_as_free(i);
        }
        self.chunks_in_use -= freed_chunks;

        // Resume the next search where memory just became available.
        // Allocating and freeing a buffer of the same shape over and over is
        // the common case, and it only stays cheap if the search starts at the
        // region that was just released instead of walking the heap again.
        self.maybe_next_free_chunk =
            ChunkCacheEntry::new(index, layout.align(), freed_chunks);
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
    ) -> Result<NonNull<[u8]>, OutOfMemory> {
        trace!("called realloc");

        // zero sized types may trigger this; according to the Rust doc of the
        // `Allocator` trait this is intended. I work around this by
        // changing the size to 1.
        let old_layout = normalize_layout(old_layout);

        let required_chunks = self.calc_required_chunks(old_layout.size());
        let occupied_size = required_chunks * CHUNK_SIZE;

        // Reuse the allocation when it already has enough space.
        if new_size <= occupied_size {
            // `max(1)`: a shrink to zero keeps the allocation alive, and
            // `normalize_layout` will report one chunk on the matching
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
            trace!("realloc fast return possible!");
            Ok(NonNull::slice_from_raw_parts(ptr, new_size))
        } else {
            trace!("realloc fast return NOT possible!");

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
                ptr::copy_nonoverlapping(
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
    use std::vec::Vec;

    mod helpers {

        use super::*;

        /// Aligned backing memory for an allocator under test.
        ///
        /// Tests that expect a specific chunk count, or that exercise
        /// allocations aligned beyond the chunk size, need to know where their
        /// region starts. Over-allocating a `Vec` and taking an aligned window
        /// out of it is enough to get that.
        #[derive(Debug)]
        pub struct Region {
            buffer: Vec<u8>,
            offset: usize,
            len: usize,
        }

        impl Region {
            /// Creates `len` bytes aligned to `alignment`, every one of them
            /// `fill`.
            ///
            /// Filling with something other than zero is what proves that the
            /// allocator does not rely on a pre-zeroed region.
            pub fn aligned(len: usize, alignment: usize, fill: u8) -> Self {
                let buffer = std::vec![fill; len + alignment];
                let offset = buffer.as_ptr().align_offset(alignment);
                Self {
                    buffer,
                    offset,
                    len,
                }
            }

            /// Creates `len` page-aligned bytes, every one of them `fill`.
            pub fn new(len: usize, fill: u8) -> Self {
                Self::aligned(len, 4096, fill)
            }

            /// Creates memory that holds exactly `chunk_count` chunks plus
            /// their bitmap.
            ///
            /// The alignment has to reach the chunk size, not just a page:
            /// otherwise the allocator skips bytes to align its first chunk
            /// and the region holds one chunk less than asked for.
            pub fn for_chunks<const CHUNK_SIZE: usize>(
                chunk_count: usize,
            ) -> Self {
                Self::aligned(
                    chunk_count * CHUNK_SIZE + chunk_count.div_ceil(8),
                    if CHUNK_SIZE > 4096 { CHUNK_SIZE } else { 4096 },
                    0,
                )
            }

            pub fn as_mut_slice(&mut self) -> &mut [u8] {
                &mut self.buffer[self.offset..self.offset + self.len]
            }
        }

        /// Creates an allocator over the given backing memory.
        ///
        /// Every test keeps its region in scope for as long as the allocator,
        /// so the shim can hide the unsafe block that every construction would
        /// otherwise repeat.
        pub fn allocator_over<const CHUNK_SIZE: usize>(
            region: &mut [u8],
        ) -> ChunkAllocator<CHUNK_SIZE> {
            // SAFETY: the caller keeps `region` alive and untouched for as long
            // as the allocator, which the returned type no longer expresses.
            unsafe { ChunkAllocator::new(region.as_mut_ptr(), region.len()) }
        }
    }

    /// The region has to hold the chunks *and* their bitmap, and the allocator
    /// must not leave usable space behind. Sweeping the start offset covers
    /// the padding it inserts to align the first chunk.
    #[test]
    fn test_geometry_uses_the_region_to_the_last_byte() {
        const CS: usize = 64;
        let mut backing = helpers::Region::new(16 * 1024, 0xff);
        let backing = backing.as_mut_slice();

        for offset in [0, 1, 7, 63, 64, 65, 4095] {
            for len in [2 * CS, 3 * CS, 1000, 4096, 8191] {
                let region = &mut backing[offset..offset + len];
                let alloc = helpers::allocator_over::<CS>(region);
                let chunks = alloc.chunk_count();

                let padding =
                    alloc.heap_ptr() as usize - region.as_ptr() as usize;
                let used = padding + chunks * CS + chunks.div_ceil(8);
                assert!(used <= len, "geometry overruns the region");
                assert!(
                    padding + (chunks + 1) * CS + (chunks + 1).div_ceil(8)
                        > len,
                    "one more chunk would still have fit"
                );
            }
        }
    }

    /// The alignment guarantee the allocator gives to every allocation rests
    /// on this: it aligns the first chunk itself, whatever the caller passed.
    #[test]
    fn test_first_chunk_is_chunk_aligned() {
        const CS: usize = 512;
        let mut backing = helpers::Region::new(8 * 1024, 0);
        let backing = backing.as_mut_slice();

        for offset in [0, 1, 8, 255, 256, 511, 513] {
            let region = &mut backing[offset..];
            let alloc = helpers::allocator_over::<CS>(region);
            assert_eq!(alloc.heap_ptr().align_offset(CS), 0);
            assert_eq!(alloc.min_alignment(), CS);
        }
    }

    /// `chunks_in_use` is a counter kept alongside the bitmap, so the two are
    /// separate records of the same fact and can drift apart. Only the
    /// counter is observable from outside, through `usage`, so a path that
    /// updates one and not the other passes every test that looks at `usage`
    /// alone.
    #[test]
    fn test_chunks_in_use_matches_the_bitmap() {
        const CS: usize = 256;

        fn marked_chunks(alloc: &ChunkAllocator<CS>) -> usize {
            (0..alloc.chunk_count())
                .filter(|index| !alloc.chunk_is_free(*index))
                .count()
        }

        let mut backing = helpers::Region::for_chunks::<CS>(32);
        let mut alloc = helpers::allocator_over::<CS>(backing.as_mut_slice());
        let layout = Layout::from_size_align(CS * 3, 1).unwrap();

        let first = alloc.allocate(layout).unwrap().cast();
        let second = alloc.allocate(layout).unwrap().cast();
        assert_eq!(marked_chunks(&alloc), alloc.chunks_in_use);

        // A shrink releases chunks without going through `deallocate`.
        // SAFETY: `first` is live and was allocated for `layout`.
        let shrunk = unsafe { alloc.realloc(first, layout, CS) }.unwrap();
        assert_eq!(marked_chunks(&alloc), alloc.chunks_in_use);

        // A grow that does not fit in place allocates, copies and frees.
        // SAFETY: `second` is live and was allocated for `layout`.
        let grown = unsafe { alloc.realloc(second, layout, CS * 6) }.unwrap();
        assert_eq!(marked_chunks(&alloc), alloc.chunks_in_use);

        // SAFETY: both pointers are live and paired with their layouts.
        unsafe {
            alloc.deallocate(
                shrunk.cast(),
                Layout::from_size_align(CS, 1).unwrap(),
            );
            alloc.deallocate(
                grown.cast(),
                Layout::from_size_align(CS * 6, 1).unwrap(),
            );
        }
        assert_eq!(marked_chunks(&alloc), alloc.chunks_in_use);
        assert_eq!(alloc.chunks_in_use, 0);
    }

    /// Tests the method `chunk_index_to_bitmap_indices()`.
    #[test]
    fn test_chunk_index_to_bitmap_indices() {
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());

        // chunk 3 gets described by bitmap byte 0 bit 3
        assert_eq!((0, 3), alloc.chunk_index_to_bitmap_indices(3));
        assert_eq!((0, 7), alloc.chunk_index_to_bitmap_indices(7));
        // chunk 8 gets described by bitmap byte 1 bit 0
        assert_eq!((1, 0), alloc.chunk_index_to_bitmap_indices(8));
        assert_eq!((1, 1), alloc.chunk_index_to_bitmap_indices(9));
        assert_eq!((1, 7), alloc.chunk_index_to_bitmap_indices(15));
    }

    /// Marking chunks must affect exactly the addressed bit.
    #[test]
    fn test_chunk_is_free() {
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());

        for index in [0, 1, 2, 3, 5] {
            alloc.mark_chunk_as_used(index);
        }
        assert!(!alloc.chunk_is_free(0));
        assert!(!alloc.chunk_is_free(3));
        assert!(alloc.chunk_is_free(4));
        assert!(!alloc.chunk_is_free(5));
        assert!(alloc.chunk_is_free(6));

        alloc.mark_chunk_as_free(3);
        assert!(alloc.chunk_is_free(3));
        assert!(!alloc.chunk_is_free(2));
    }

    /// Tests the `chunk_index_to_ptr` method.
    #[test]
    fn test_chunk_index_to_ptr() {
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());
        let heap_ptr = alloc.heap_ptr();

        assert_eq!(heap_ptr, alloc.chunk_index_to_ptr(0));
        // SAFETY: all computed pointers remain within the chunk area.
        unsafe {
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
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());

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
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());

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
        let mut backing = helpers::Region::for_chunks::<DEFAULT_CHUNK_SIZE>(32);
        let mut alloc: ChunkAllocator =
            helpers::allocator_over(backing.as_mut_slice());

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
}
