//! Module for [`GlobalChunkAllocator`].

use crate::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;

/// Synchronized wrapper around [`ChunkAllocator`] that implements the Rust traits
/// [`GlobalAlloc`] which enables the usage as global allocator.
#[derive(Debug)]
pub struct GlobalChunkAllocator<'a, const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE>(
    spin::Mutex<Option<ChunkAllocator<'a, CHUNK_SIZE>>>,
);

impl<'a, const CHUNK_SIZE: usize> GlobalChunkAllocator<'a, CHUNK_SIZE> {
    /// Constructor. Don't forget to call [`Self::init`] first, before actual
    /// allocations happen!
    pub const fn new() -> Self {
        let lock = spin::Mutex::new(None);
        Self(lock)
    }

    /// Initializes the underlying [`ChunkAllocator`].
    pub fn init(&self, heap: &'a mut [u8], bitmap: &'a mut [u8]) -> Result<(), ()> {
        let mut lock = self.0.lock();
        if lock.is_some() {
            Err(())
        } else {
            let alloc = ChunkAllocator::<CHUNK_SIZE>::new(heap, bitmap).unwrap();
            lock.replace(alloc);
            Ok(())
        }
    }

    /// Returns the usage of the allocator in percentage.
    pub fn usage(&self) -> f32 {
        self.0.lock().as_ref().unwrap().usage()
    }

    /// Returns an instance of [`AllocatorApiGlue`].
    pub fn allocator_api_glue<'b>(&'b self) -> AllocatorApiGlue<'a, 'b, CHUNK_SIZE> {
        AllocatorApiGlue(self)
    }
}

unsafe impl<'a, const CHUNK_SIZE: usize> GlobalAlloc for GlobalChunkAllocator<'a, CHUNK_SIZE> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0
            .lock()
            .as_mut()
            .expect("call init first")
            .allocate(layout)
            .unwrap()
            .as_mut_ptr()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.0
            .lock()
            .as_mut()
            .expect("call init first")
            .deallocate(NonNull::new(ptr).unwrap(), layout)
    }
}

/// Helper struct generated by [`GlobalChunkAllocator`] that can be used in structs that accept
/// a custom instance of a specific allocator implementation.
///
/// This is mainly relevant for testing my stuff. In a real environment, an instance of
/// [`GlobalChunkAllocator`] is likely registered a global allocator for Rust. Thus, `Vec` and
/// others default to the `Global` allocator, which will use [`GlobalChunkAllocator`].
///
/// # Example
/// ```ignore
/// const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * 8;
/// const CHUNK_COUNT: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE;
/// const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
///
/// static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
/// static mut BITMAP_MEM: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];
/// static ALLOCATOR: GlobalChunkAllocator = GlobalChunkAllocator::new();
/// unsafe {
///     ALLOCATOR.init(HEAP_MEM.as_mut_slice(), BITMAP_MEM.as_mut_slice()).unwrap()
/// };
///
/// // Vector from the standard library will use my custom allocator instead.
/// let vec = Vec::<u8, _>::with_capacity_in(123, ALLOCATOR.allocator_api_glue());
/// ```
#[derive(Debug)]
pub struct AllocatorApiGlue<'a, 'b, const CHUNK_SIZE: usize>(
    &'a GlobalChunkAllocator<'b, CHUNK_SIZE>,
);

unsafe impl<'a, 'b, const CHUNK_SIZE: usize> Allocator for AllocatorApiGlue<'a, 'b, CHUNK_SIZE> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0
             .0
            .lock()
            .as_mut()
            .expect("call init first")
            .allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0
             .0
            .lock()
            .as_mut()
            .expect("call init first")
            .deallocate(ptr, layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;


    /// Uses [`GlobalChunkAllocator`] against the Rust Allocator API to test
    /// the underlying [`ChunkAllocator`]. This is like an "integration" test
    /// whereas the other tests in the other module are unit tests.
    #[test]
    fn test_allocator_with_allocator_api() {
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * 8;
        const CHUNK_COUNT: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE;
        const BITMAP_SIZE: usize = CHUNK_COUNT / 8;

        static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
        static mut BITMAP_MEM: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];
        static ALLOCATOR: GlobalChunkAllocator = GlobalChunkAllocator::new();
        unsafe {
            ALLOCATOR.init(HEAP_MEM.as_mut_slice(), BITMAP_MEM.as_mut_slice()).unwrap()
        };

        assert_eq!(0.0, ALLOCATOR.usage());
        let vec1 = Vec::<u8, _>::with_capacity_in(DEFAULT_CHUNK_SIZE * 2, ALLOCATOR.allocator_api_glue());
        assert_eq!(25.0, ALLOCATOR.usage());
        let vec2 = Vec::<u8, _>::with_capacity_in(DEFAULT_CHUNK_SIZE * 6, ALLOCATOR.allocator_api_glue());
        assert_eq!(100.0, ALLOCATOR.usage());

        // I can't test it like this :( Because of the design of the types of the Rust standard library,
        // they fail if an allocation can't be satisfied. However, they do not throw a normal kind
        // of panic but trigger the "rust alloc error" hook, which terminates the program in a
        // different manner.
        //let alloc_res = std::panic::catch_unwind(|| {
        //    let _vec = Vec::<u8, _>::with_capacity_in(1, ALLOCATOR.allocator_api_glue());
        //});
        //assert!(panic_oom.is_err(), "allocator is out of memory");


        drop(vec1);
        assert_eq!(75.0, ALLOCATOR.usage());
        let vec3 = Vec::<u8, _>::with_capacity_in(DEFAULT_CHUNK_SIZE * 1, ALLOCATOR.allocator_api_glue());
        assert_eq!(87.5, ALLOCATOR.usage());

        drop(vec2);
        drop(vec3);
        assert_eq!(0.0, ALLOCATOR.usage());
    }
}
