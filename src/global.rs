//! Module for [`GlobalChunkAllocator`].

use core::alloc::{Allocator, AllocError, GlobalAlloc, Layout};
use core::ptr::NonNull;
use crate::{ChunkAllocator, DEFAULT_CHUNK_SIZE};

/// Synchronized wrapper around [`ChunkAllocator`] that implements the Rust traits
/// [`Allocator`] and [`GlobalAlloc`].
pub struct GlobalChunkAllocator<'a, const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE>(
    spin::Mutex<ChunkAllocator<'a, CHUNK_SIZE>>,
);

unsafe impl<'a> Allocator for GlobalChunkAllocator<'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.lock().allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.lock().deallocate(ptr, layout)
    }
}

unsafe impl<'a> GlobalAlloc for GlobalChunkAllocator<'a> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0.lock().allocate(layout).unwrap().as_mut_ptr()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.0.lock().deallocate(NonNull::new(ptr).unwrap(), layout)
    }
}
