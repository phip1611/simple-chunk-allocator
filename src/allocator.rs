//! Module for [`ChunkAllocator`].

use core::alloc::AllocError;
use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;
use libm;

/// Possible errors of [`ChunkAllocator`].
#[derive(Debug)]
pub enum ChunkAllocatorError {
    /// The backing memory for the heap must be
    /// - an even multiple of [`DEFAULT_ALLOCATOR_CHUNK_SIZE`], and
    /// - a multiple of 8 to be correctly represented by the bitmap.
    BadHeapMemory,
    /// The number of bits in the backing memory for the heap bitmap
    /// must match the number of chunks in the heap.
    BadBitmapMemory,
    /// The chunk size must be a multiple of 8 and not 0.
    BadChunkSize,
    /// The heap is either completely full or to fragmented to serve
    /// the request. Also, it may happen that the alignment can't get
    /// guaranteed, because all aligned chunks are already in use.
    OutOfMemory,
}

/// Default chunk size used by [`ChunkAllocator`].
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Allocator that takes a mutable reference to a continuous region of external memory used as
/// backing storage. Additionally, it needs mutable memory to store its bookkeeping data.
///
/// It allocates memory in chunks of custom length, i.e. `256` or `4096`.
/// Default value is [`DEFAULT_CHUNK_SIZE`].
///
/// It is mandatory to wrap this allocator by a mutex or a similar primitive, if it should be
/// used in a global context. It can take (global) static memory arrays as backing storage.
/// See [`crate::GlobalChunkAllocator`] for an implementation of that.
///
/// This can be used to construct allocators, that manage the heap of `no_std` binaries.
///
/// The allocation strategy is a variant of `Next Fit`
/// (see <https://www.geeksforgeeks.org/partition-allocation-methods-in-memory-management/>).
#[derive(Debug)]
pub struct ChunkAllocator<'a, const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE> {
    /// Backing memory for heap.
    heap: &'a mut [u8],
    /// Backing memory for bookkeeping.
    bitmap: &'a mut [u8],
    /// Helper to do some initial initialization on the first runtime invocation.
    is_first_alloc: Cell<bool>,
    /// Contains the next free chunk, maybe. The first value of the pair probably contains the
    /// next free chunk. This is always the case, if a allocation follows a deallocation. The
    /// deallocation will store the just freed block in this property to accelerate the next
    /// allocation. In case of allocations, the allocation algorithm may set the value to a free
    /// chunk it finds during the search when this iteration doesn't occupy these chunks by itself.
    ///
    /// This optimization mechanism prevents the need to iterate over all chunks everytime which
    /// can take up to tens of thousands of CPU cycles in the worst case (full heap).
    maybe_next_free_chunk: (usize, usize),
}

impl<'a, const CHUNK_SIZE: usize> ChunkAllocator<'a, CHUNK_SIZE> {
    /*/// Returns the default const generic value of `CHUNK_SIZE`.
    #[inline]
    pub const fn default_chunk_size() -> usize {
        // keep in sync with struct definition!
        DEFAULT_ALLOCATOR_CHUNK_SIZE
    }*/

    /// Returns the used chunk size.
    #[inline]
    pub const fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }

    /// Creates a new allocator object. Verifies that the provided memory has the correct properties.
    /// Zeroes the bitmap.
    ///
    /// - heap length must be a multiple of [`CHUNK_SIZE`]
    /// - the heap must be >= 0
    #[inline]
    pub const fn new(
        heap: &'a mut [u8],
        bitmap: &'a mut [u8],
    ) -> Result<Self, ChunkAllocatorError> {
        if CHUNK_SIZE == 0 || CHUNK_SIZE % 8 != 0 {
            return Err(ChunkAllocatorError::BadChunkSize);
        }
        let heap_is_empty = heap.len() == 0;
        let is_not_multiple_of_chunk_size = heap.len() % CHUNK_SIZE != 0;
        let is_not_coverable_by_bitmap = heap.len() < 8 * CHUNK_SIZE;
        if heap_is_empty || is_not_multiple_of_chunk_size || is_not_coverable_by_bitmap {
            return Err(ChunkAllocatorError::BadHeapMemory);
        }

        // check bitmap memory has correct length
        let chunk_count = heap.len() / CHUNK_SIZE;
        let expected_bitmap_length = chunk_count / 8;
        if bitmap.len() != expected_bitmap_length {
            return Err(ChunkAllocatorError::BadBitmapMemory);
        }

        Ok(Self {
            heap,
            bitmap,
            is_first_alloc: Cell::new(true),
            maybe_next_free_chunk: (0, chunk_count),
        })
    }

    /// Capacity in bytes of the allocator.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.heap.len()
    }

    /// Returns number of chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        // size is a multiple of CHUNK_SIZE;
        // ensured in new()
        self.capacity() / CHUNK_SIZE
    }

    /// Returns the current memory usage in percentage.
    #[inline]
    pub fn usage(&self) -> f32 {
        let mut used_chunks = 0;
        let chunk_count = self.chunk_count();
        for chunk_i in 0..chunk_count {
            if !self.chunk_is_free(chunk_i) {
                used_chunks += 1;
            }
        }
        let ratio = used_chunks as f32 / chunk_count as f32;
        libm::roundf(ratio * 10000.0) / 100.0
    }

    /// Returns whether a chunk is free according to the bitmap.
    ///
    /// # Parameters
    /// - `chunk_index` describes the start chunk; i.e. the search space inside the backing storage
    #[inline(always)]
    fn chunk_is_free(&self, chunk_index: usize) -> bool {
        assert!(
            chunk_index < self.chunk_count(),
            "chunk_index={} is bigger than max_chunk_count()={}",
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
        assert!(chunk_index < self.chunk_count());
        if !self.chunk_is_free(chunk_index) {
            panic!(
                "tried to mark chunk {} as used but it is already used",
                chunk_index
            );
        }
        let (byte_i, bit) = self.chunk_index_to_bitmap_indices(chunk_index);
        // xor => keep all bits, except bitflip at relevant position
        self.bitmap[byte_i] = self.bitmap[byte_i] ^ (1 << bit);
    }

    /// Marks a chunk as free, i.e. write a 0 into the bitmap at the right position.
    #[inline(always)]
    fn mark_chunk_as_free(&mut self, chunk_index: usize) {
        assert!(chunk_index < self.chunk_count());
        if self.chunk_is_free(chunk_index) {
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
        assert!(
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
        if
        /*chunk_num_request < 0 || */
        chunk_num_request > self.chunk_count() {
            // out of memory
            return Err(ChunkAllocatorError::OutOfMemory);
        }

        // Use optimization; fast allocate
        // - check if it is really free and already properly allocated
        if self.chunk_is_free(self.maybe_next_free_chunk.0)
            // - check that it fits
            && chunk_num_request <= self.maybe_next_free_chunk.1
            // - check alignment
            && unsafe {
            self.chunk_index_to_ptr(self.maybe_next_free_chunk.0)
                .align_offset(alignment)
                == 0
        } {
            let found_index = self.maybe_next_free_chunk.0;
            // TODO it feels false that this method updates this data structure but
            //  does not mark the blocks as used! I think the upper method should do that!
            self.maybe_next_free_chunk = (
                (self.maybe_next_free_chunk.0 + chunk_num_request) % self.chunk_count(),
                self.maybe_next_free_chunk.1 - chunk_num_request,
            );
            Ok(found_index)
        } else {
            let start_index = self.maybe_next_free_chunk.0;

            let res = (start_index..(start_index + self.chunk_count()))
                // cope with wrapping indices (i.e. index 0 follows 31)
                .map(|index| index % self.chunk_count())
                .filter(|chunk_index| self.chunk_is_free(*chunk_index))
                // If the heap has 8 chunks and we need 4 but start the search at index 6, then we
                // don't have enough continuous chunks to fulfill the request. Thus, we skip those.
                .skip_while(|chunk_index| {
                    *chunk_index + (chunk_num_request - 1) >= self.chunk_count()
                })
                // ALIGNMENT CHECK BEGIN
                .map(|chunk_index| (chunk_index, unsafe { self.chunk_index_to_ptr(chunk_index) }))
                .filter(|(_, addr)| addr.align_offset(alignment) == 0)
                .map(|(chunk_index, _)| chunk_index)
                // ALIGNMENT CHECK END
                // Now look for the continuous region: are all succeeding chunks free?
                // This is safe because earlier I skipped chunk_indices that are too close to
                // the end.
                .filter(|chunk_index| {
                    // +1: chunk at chunk_index itself is already free
                    // -1: num => index
                    (chunk_index + 1..((chunk_index + 1) + (chunk_num_request - 1)))
                        .all(|index| self.chunk_is_free(index))
                })
                // get the first
                .next()
                // OK or out of memory
                .ok_or(ChunkAllocatorError::OutOfMemory);

            if let Ok(index) = res {
                // TODO it feels false that this method updates this data structure but
                //  does not mark the blocks as used! I think the upper method should do that!

                // Only update "maybe_next_free_chunk" if it doesn't already point to a free location
                if !self.chunk_is_free(self.maybe_next_free_chunk.0) || self.maybe_next_free_chunk.1 == 0 {
                    self.maybe_next_free_chunk = ((index + 1) % self.chunk_count(), 1);
                }

            }

            res
        }
    }

    /// Returns the pointer to the beginning of the chunk.
    #[inline(always)]
    unsafe fn chunk_index_to_ptr(&self, chunk_index: usize) -> *mut u8 {
        assert!(
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
        assert!(
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
    fn calc_required_chunks(&self, size: usize) -> usize {
        if size % CHUNK_SIZE == 0 {
            size / CHUNK_SIZE
        } else {
            (size / CHUNK_SIZE) + 1
        }
    }

    #[track_caller]
    #[inline]
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if self.is_first_alloc.get() {
            self.is_first_alloc.replace(false);
            // Zero bitmap
            self.bitmap.fill(0);
        }

        // zero sized types may trigger this; according to the Rust doc of the `Allocator`
        // trait this is intended. I work around this by changing the size to 1.
        let layout = if layout.size() == 0 {
            Layout::from_size_align(1, layout.align()).unwrap()
        } else {
            layout
        };

        let required_chunks = self.calc_required_chunks(layout.size());

        let index = self.find_free_continuous_memory_region(required_chunks, layout.align());

        if let Err(_) = index {
            log::warn!(
                "Out of Memory. Can't fulfill the requested layout: {:?}. Current usage is: {}%/{}byte",
                layout,
                self.usage(),
                ((self.usage() * self.capacity() as f32) as u64)
            );
        }
        let index = index.map_err(|_| AllocError)?;

        for i in index..index + required_chunks {
            self.mark_chunk_as_used(i);
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

    #[track_caller]
    #[inline]
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        // zero sized types may trigger this; according to the Rust doc of the `Allocator`
        // trait this is intended. I work around this by changing the size to 1.
        let layout = if layout.size() == 0 {
            Layout::from_size_align(1, layout.align()).unwrap()
        } else {
            layout
        };

        let freed_chunks = self.calc_required_chunks(layout.size());

        log::trace!("dealloc: layout={:?}, #chunks={})", layout, freed_chunks);

        let index = self.ptr_to_chunk_index(ptr.as_ptr());
        for i in index..index + freed_chunks {
            self.mark_chunk_as_free(i);
        }

        // This helps the next allocation to be faster. Currently, this prefers the smallest
        // possible continuous region. This prevents fragmentation but assumes/hopes the next
        // allocation only needs one single chunk.
        if freed_chunks < self.maybe_next_free_chunk.1 {
            self.maybe_next_free_chunk = (index, freed_chunks);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        assert!(matches!(
            ChunkAllocator::<0>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
            ChunkAllocatorError::BadChunkSize
        ));
        assert!(matches!(
            ChunkAllocator::<1>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
            ChunkAllocatorError::BadChunkSize
        ));
        assert!(matches!(
            ChunkAllocator::<9>::new(&mut heap, &mut heap_bitmap).unwrap_err(),
            ChunkAllocatorError::BadChunkSize
        ));
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
        static mut HEAP: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
        const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE / 8;
        static mut HEAP_BITMAP: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];

        // check that it compiles
        let mut _alloc: ChunkAllocator =
            unsafe { ChunkAllocator::new(&mut HEAP, &mut HEAP_BITMAP).unwrap() };
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
            let mut heap = vec![0_u8; heap_size];
            let bitmap_size_exact = if chunk_count % 8 == 0 {
                chunk_count / 8
            } else {
                (chunk_count / 8) + 1
            };
            let mut bitmap = vec![0_u8; bitmap_size_exact];
            let alloc: ChunkAllocator = ChunkAllocator::new(&mut heap, &mut bitmap).unwrap();
            assert_eq!(
                chunk_count,
                alloc.chunk_count(),
                "the allocator must get constructed successfully because the bitmap size matches the number of chunks"
            );
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
            let mut heap = vec![0_u8; heap_size];
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

        // I made this test for these two properties. Test might need to get adjusted if this
        // changes
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        assert_eq!(
            0,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(0);
        alloc.maybe_next_free_chunk = (1, 1);

        assert_eq!(1, alloc.find_free_continuous_memory_region(1, 1).unwrap());
        alloc.maybe_next_free_chunk = (2, 1);
        assert_eq!(
            // 16: 256*16 = 4096 => second page in heap mem that consists of two pages
            16,
            alloc.find_free_continuous_memory_region(1, 4096).unwrap()
        );
        alloc.mark_chunk_as_used(16);
        // makes sure the next search
        alloc.maybe_next_free_chunk = (17, 1);

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

        // I made this test for these two properties. Test might need to get adjusted if this
        // changes
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

        // I made this test for these two properties. Test might need to get adjusted if this
        // changes
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
