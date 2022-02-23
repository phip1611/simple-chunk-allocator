//! Module for [`ChunkAllocator`].

#![no_std]
#![feature(allocator_api)]
#![feature(const_mut_refs)]
#![feature(const_for)]

#[cfg(test)]
#[macro_use]
extern crate std;

use core::alloc::Layout;
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
}

/// Default chunk size used by [`ChunkAllocator`].
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// First-fit allocator that takes mutable references to arbitrary external memory
/// backing storages. It uses them to manage memory. It is mandatory to wrap
/// this allocator by a mutex or a similar primitive, if it should be used
/// in a global context. It can take (global) static memory arrays as backing
/// storage. It allocates memory in chunks of custom length, i.e. `256` or `4096`.
/// Default value is [`DEFAULT_ALLOCATOR_CHUNK_SIZE`].
///
/// This can be used to construct allocators, that manage the heap for the roottask
/// or virtual memory.
///
/// TODO: In fact, the chunk allocator only needs the bitmap reference, but not the
///  one from the heap. Future work: completely throw this away and instead do some
///  mixture of PAge-Frame-Allocator and Virtual Memory Mapper
///  --- Maybe I can keep this and build it on top of frame allocator and virtual
///      memory allocator.
#[derive(Debug)]
pub struct ChunkAllocator<'a, const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE> {
    /// Backing memory for heap.
    heap: &'a mut [u8],
    /// Backing memory for bookkeeping.
    bitmap: &'a mut [u8],
    /// Helper to do some initial initialization on the first runtime invocation.
    is_first_alloc: bool,
    /// Contains the next free chunk, maybe. The first value of the pair is the index of
    /// the chunk whereas the second one is a hint for the size. This value gets updated
    /// during allocations and allocations. It helps to accelerate the lookup of new
    /// free chunks. For example, if a deallocation happens, it is likely that the next allocation
    /// fit into that space (hopefully). This prevents the need to iterate over all chunks which
    /// can take up to tens of thousands of CPU cycles in the worst case.
    maybe_next_free_chunk: Option<(usize, usize)>,
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
        let expected_bitmap_length = heap.len() / CHUNK_SIZE / 8;
        if bitmap.len() != expected_bitmap_length {
            return Err(ChunkAllocatorError::BadBitmapMemory);
        }

        Ok(Self {
            heap,
            bitmap ,
            is_first_alloc: true,
            maybe_next_free_chunk: Some((0, expected_bitmap_length * 8))
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
        assert!(chunk_index < self.chunk_count());
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

    /// Finds the next free chunk who's pointer has the guaranteed alignment.
    ///
    /// # Parameters
    /// - `alignment` required alignment of the chunk in memory
    ///
    /// # Return
    /// Returns the index of the chunk or `Err` for out of memory.
    #[inline(always)]
    fn find_next_free_aligned_chunk_index(
        &self,
        alignment: usize,
    ) -> Result<usize, ()> {
        let start = self.maybe_next_free_chunk.map(|(index, _)| index).unwrap_or(0);

        #[cfg(test)]
        println!("---------");

        // iterates over all chunks to find a free one; can cope with wrapping at the border
        // i.e. searches indice 16-31 and 0-15 in that order.
        (start..(start + self.chunk_count())).map(|index| index % self.chunk_count())
            .map(|x| {
                #[cfg(test)]
                println!("{}", x);
                x
            })
            .filter(|chunk_index| self.chunk_is_free(*chunk_index))
            .map(|chunk_index| (chunk_index, unsafe { self.chunk_index_to_ptr(chunk_index) }))
            .filter(|(_, addr)| addr.align_offset(alignment) == 0)
            .map(|(chunk_index, _)| chunk_index)
            // get the first
            .next()
            // OK or out of memory
            .ok_or(())
    }

    /*/// Finds the next available chain of available chunks. Returns the
    /// beginning index.
    ///
    /// # Parameters
    /// - `chunk_num` number of chunks that must be all free without gap in-between; greater than 0
    /// - `alignment` required alignment of the chunk in memory
    #[inline(always)]
    fn find_free_coherent_chunks_aligned(
        &self,
        chunk_num: usize,
        alignment: usize,
    ) -> Result<usize, ()> {
        assert!(
            chunk_num > 0,
            "chunk_num must be greater than 0! Allocating 0 blocks makes no sense"
        );
        let mut begin_chunk_i = self.find_next_free_aligned_chunk_index(Some(0), alignment)?;
        loop {
            let out_of_mem_cond = begin_chunk_i + (chunk_num - 1) >= self.chunk_count();
            if out_of_mem_cond {
                break;
            }

            // this var counts how many coherent chunks we found while iterating the bitmap
            let mut coherent_chunk_count = 1;
            for chunk_chain_i in 1..=chunk_num {
                if coherent_chunk_count == chunk_num {
                    return Ok(begin_chunk_i);
                } else if self.chunk_is_free(begin_chunk_i + chunk_chain_i) {
                    coherent_chunk_count += 1;
                } else {
                    break;
                }
            }

            // check again at next free block
            // "+1" because we want to skip the just discovered non-free block
            begin_chunk_i = self
                .find_next_free_aligned_chunk_index(
                    Some(begin_chunk_i + coherent_chunk_count + 1),
                    alignment,
                )
                .unwrap();
        }
        // out of memory
        Err(())
    }*/

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

    /*/// Allocates memory from the backing storage. Performs bookkeeping in the bitmao
    /// data structure. On the very first call, this ensures that the bitmap gets zeroed.
    #[track_caller]
    #[inline]
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        if self.is_first_alloc {
            self.is_first_alloc = false;
            // Zero bitmap
            self.bitmap.fill(0);
        }

        // zero sized types may trigger this
        let layout = if layout.size() == 0 {
            Layout::from_size_align(1, layout.align()).unwrap()
        } else {
            layout
        };

        let required_chunks = if layout.size() % CHUNK_SIZE == 0 {
            layout.size() / CHUNK_SIZE
        } else {
            (layout.size() / CHUNK_SIZE) + 1
        };

        let index = self.find_free_coherent_chunks_aligned(required_chunks, layout.align());

        if let Err(_) = index {
            panic!(
                "Out of Memory. Can't fulfill the requested layout: {:?}. Current usage is: {}%/{}byte",
                layout,
                self.usage(),
                ((self.usage() * self.capacity() as f32) as u64)
            );
        }
        let index = index.unwrap();

        for i in index..index + required_chunks {
            self.mark_chunk_as_used(i);
        }

        self.chunk_index_to_ptr(index)

        TODO next_free_index_optimization
    }

    #[track_caller]
    #[inline]
    pub unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let mut required_chunks = layout.size() / CHUNK_SIZE;
        let modulo = layout.size() % CHUNK_SIZE;
        if modulo != 0 {
            required_chunks += 1;
        }
        // log::debug!("dealloc: layout={:?} ({} chunks]", layout, required_chunks);

        let index = self.ptr_to_chunk_index(ptr as *const u8);
        for i in index..index + required_chunks {
            self.mark_chunk_as_free(i);
        }
        TODO next_free_index_optimization
    }*/
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
        pub struct PageAlignedAlloc;

        unsafe impl Allocator for PageAlignedAlloc {
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
        pub fn create_heap_and_bitmap_vectors(
        ) -> (Vec<u8, PageAlignedAlloc>, Vec<u8, PageAlignedAlloc>) {
            // 32 chunks with default chunk size = 256 bytes = 2 pages = 2*4096
            const CHUNK_COUNT: usize = 32;
            const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
            let mut heap = Vec::with_capacity_in(HEAP_SIZE, PageAlignedAlloc);
            (0..heap.capacity()).for_each(|_| heap.push(0));
            const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE / 8;
            let mut heap_bitmap = Vec::with_capacity_in(BITMAP_SIZE, PageAlignedAlloc);
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
        let mut _alloc = ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut heap_bitmap).unwrap();
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
        let mut _alloc = unsafe { ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut HEAP, &mut HEAP_BITMAP).unwrap() };
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
            let alloc = ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut bitmap).unwrap();
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
        let alloc = ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut heap_bitmap).unwrap();

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
        let alloc = ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut heap_bitmap).unwrap();


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
        let alloc =
            ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut heap_bitmap).unwrap();

        unsafe {
            assert_eq!(heap_ptr, alloc.chunk_index_to_ptr(0));
            assert_eq!(heap_ptr.add(alloc.chunk_size() * 1), alloc.chunk_index_to_ptr(1));
            assert_eq!(heap_ptr.add(alloc.chunk_size() * 7), alloc.chunk_index_to_ptr(7));
        }
    }

    /// Tests the method `find_next_free_chunk_aligned` which is the base for the allocation algorithm.
    #[test]
    fn test_find_next_free_chunk_aligned() {
        let (mut heap, mut heap_bitmap) = helpers::create_heap_and_bitmap_vectors();
        let mut alloc =
            ChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(&mut heap, &mut heap_bitmap).unwrap();

        // I made this test for these two properties. Test might need to get adjusted if this
        // changes
        assert_eq!(alloc.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(alloc.chunk_count(), 32);

        assert_eq!(0, alloc.find_next_free_aligned_chunk_index(4096).unwrap());
        alloc.mark_chunk_as_used(0);
        alloc.maybe_next_free_chunk = Some((1, 1));

        assert_eq!(1, alloc.find_next_free_aligned_chunk_index(1).unwrap());
        alloc.maybe_next_free_chunk = Some((2, 1));
        assert_eq!(16, alloc.find_next_free_aligned_chunk_index(4096).unwrap());
        alloc.mark_chunk_as_used(16);
        // makes sure the next search
        alloc.maybe_next_free_chunk = Some((17, 1));

        assert!(alloc.find_next_free_aligned_chunk_index(4096).is_err(), "out of memory; only 2 pages of memory");

        // now free the first chunk again, which enables a further 4096 byte aligned allocation
        alloc.mark_chunk_as_free(0);
        assert_eq!(0, alloc.find_next_free_aligned_chunk_index(4096).unwrap());
    }

    /*#[test]
    fn test_find_free_coherent_chunks() {
        let heap_size: usize = 24 * DEFAULT_ALLOCATOR_CHUNK_SIZE;
        let mut heap = vec![0_u8; heap_size];
        let mut bitmap = vec![0_u8; heap_size / DEFAULT_ALLOCATOR_CHUNK_SIZE / 8];

        bitmap[0] = 0x0f;
        bitmap[1] = 0x0f;

        let alloc =
            ChunkAllocator::<DEFAULT_ALLOCATOR_CHUNK_SIZE>::new(&mut heap, &mut bitmap).unwrap();

        assert_eq!(4, alloc.find_free_coherent_chunks_aligned(1, 1).unwrap());
        assert_eq!(4, alloc.find_free_coherent_chunks_aligned(2, 1).unwrap());
        assert_eq!(4, alloc.find_free_coherent_chunks_aligned(3, 1).unwrap());
        assert_eq!(4, alloc.find_free_coherent_chunks_aligned(4, 1).unwrap());
        assert_eq!(12, alloc.find_free_coherent_chunks_aligned(5, 1).unwrap());
    }



    #[test]
    fn test_alloc() {
        // must be a multiple of 8; 32 is equivalent to two pages
        const CHUNK_COUNT: usize = 32;
        const HEAP_SIZE: usize = DEFAULT_ALLOCATOR_CHUNK_SIZE * CHUNK_COUNT;
        static mut HEAP: PageAlignedByteBuf<HEAP_SIZE> = PageAlignedByteBuf::new_zeroed();
        const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_ALLOCATOR_CHUNK_SIZE / 8;
        static mut BITMAP: PageAlignedByteBuf<BITMAP_SIZE> = PageAlignedByteBuf::new_zeroed();

        // check that it compiles
        let mut alloc = unsafe {
            ChunkAllocator::<DEFAULT_ALLOCATOR_CHUNK_SIZE>::new(HEAP.get_mut(), BITMAP.get_mut())
                .unwrap()
        };
        assert_eq!(alloc.usage(), 0.0, "allocator must report usage of 0%!");

        let layout1_single_byte = Layout::from_size_align(1, 1).unwrap();
        let layout_page = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap();

        // allocate 1 single byte
        let ptr1 = {
            unsafe {
                let ptr = alloc.alloc(layout1_single_byte.clone());
                assert_eq!(
                    ptr as u64 % PAGE_SIZE as u64,
                    0,
                    "the first allocation must be always page-aligned"
                );
                assert_eq!(alloc.usage(), 3.13, "allocator must report usage of 3.15%!");
                assert!(!alloc.chunk_is_free(0), "the first chunk is taken now!");
                assert!(
                    alloc.chunk_is_free(1),
                    "the second chunk still must be free!"
                );
                ptr
            }
        };

        // allocate 1 page (consumes now the higher half of the available memory)
        let ptr2 = {
            let ptr;
            unsafe {
                ptr = alloc.alloc(layout_page.clone());
                assert_eq!(
                    ptr as u64 % PAGE_SIZE as u64,
                    0,
                    "the second allocation must be page-aligned because this was requested!"
                );
            }
            assert_eq!(
                alloc.usage(),
                3.13 + 50.0,
                "allocator must report usage of 53.13%!"
            );
            (0..CHUNK_COUNT)
                .into_iter()
                .skip(CHUNK_COUNT / 2)
                .for_each(|i| {
                    assert!(!alloc.chunk_is_free(i), "chunk must be in use!");
                });
            ptr
        };

        // free the very first allocation; allocate again; now we should have two allocations
        // of two full pages
        {
            unsafe {
                alloc.dealloc(ptr1, layout1_single_byte);
                let ptr3 = alloc.alloc(layout_page);
                assert_eq!(ptr1, ptr3);
            }

            assert_eq!(
                alloc.usage(),
                100.0,
                "allocator must report usage of 100.0%, because two full pages (=100%) are allocated!"
            );

            // assert that all chunks are taken
            for i in 0..CHUNK_COUNT {
                assert!(!alloc.chunk_is_free(i), "all chunks must be in use!");
            }
        }

        unsafe {
            alloc.dealloc(ptr1, layout_page);
            alloc.dealloc(ptr2, layout_page);
        }
        assert_eq!(alloc.usage(), 0.0, "allocator must report usage of 0%!");
    }

    #[test]
    fn test_alloc_alignment() {
        const TWO_MIB: usize = 0x200000;
        const HEAP_SIZE: usize = 2 * TWO_MIB;
        static mut HEAP: PageAlignedByteBuf<HEAP_SIZE> = PageAlignedByteBuf::new_zeroed();
        const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_ALLOCATOR_CHUNK_SIZE / 8;
        static mut BITMAP: PageAlignedByteBuf<BITMAP_SIZE> = PageAlignedByteBuf::new_zeroed();

        // check that it compiles
        let mut alloc = unsafe {
            ChunkAllocator::<DEFAULT_ALLOCATOR_CHUNK_SIZE>::new(HEAP.get_mut(), BITMAP.get_mut())
                .unwrap()
        };
        let ptr = unsafe { alloc.alloc(Layout::new::<u8>().align_to(TWO_MIB).unwrap()) };
        assert_eq!(ptr as usize % TWO_MIB, 0, "must be aligned!");
    }

    #[test]
    #[should_panic]
    fn test_alloc_out_of_memory() {
        // must be a multiple of 8; 32 is equivalent to two pages
        const CHUNK_COUNT: usize = 32;
        const HEAP_SIZE: usize = DEFAULT_ALLOCATOR_CHUNK_SIZE * CHUNK_COUNT;
        static mut HEAP: PageAlignedByteBuf<HEAP_SIZE> = PageAlignedByteBuf::new_zeroed();
        const BITMAP_SIZE: usize = HEAP_SIZE / DEFAULT_ALLOCATOR_CHUNK_SIZE / 8;
        static mut BITMAP: PageAlignedByteBuf<BITMAP_SIZE> = PageAlignedByteBuf::new_zeroed();

        // check that it compiles
        let mut alloc = unsafe {
            ChunkAllocator::<DEFAULT_ALLOCATOR_CHUNK_SIZE>::new(HEAP.get_mut(), BITMAP.get_mut())
                .unwrap()
        };

        unsafe {
            let _ = alloc.alloc(Layout::from_size_align(16384, PAGE_SIZE).unwrap());
        }
    }*/
}

// TODO für morgen
// der allokator ist so langsam, da er irgendwan ntausende iterationen machen muss
// Idee: Optimierung: bei jeder allokation merkt er sich wo er war, da es wahrscheinlich
// ist, dass er beim nächsten mal da weiter machen kann. Diese Strategie ist zumindest gut
// so lange der heap noch nicht einmal ganz voll war. Das sollte viel Zeit sparen.
//
// Außerdem: Die Bitmap selbst kann im  Heap bereich selbst legen.. vereinfaht die API
// Und Chunk Size auf 512 byte vergrößern? Dass man nur 256 byte braucht ist unwahrscheinlich..
//
// sonst vllt: ein buddy allokator der ganze pages allokiert und dann der 256 byte allokator
// daneben?! (SLAB)
