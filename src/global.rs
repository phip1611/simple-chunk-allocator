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
//! Module for [`GlobalChunkAllocator`].

use crate::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;

/// The default number of chunks. If the default chunk size of
/// [`DEFAULT_CHUNK_SIZE`] gets used then this equals to 1MiB of memory.
pub const DEFAULT_CHUNK_AMOUNT: usize = 4096;

/// Synchronized high-level wrapper around [`ChunkAllocator`] that implements
/// the Rust traits [`GlobalAlloc`] which enables the usage as global allocator.
/// The method [`GlobalChunkAllocator::allocator_api_glue`] returns an object of
/// type [`AllocatorApiGlue`] which can be used with the `allocator_api`
/// feature.
///
/// ```rust
/// use simple_chunk_allocator::{
///     heap, heap_bitmap, GlobalChunkAllocator, PageAligned,
/// };
///
/// // The macros help to get a correctly sized arrays types.
/// // I page-align them for better caching and to improve the availability of
/// // page-aligned addresses.
///
/// /// Backing storage for the heap (1 MiB).
///
/// /// heap!: first argument is chunk amount; second is chunk size.
/// ///        If no arguments are provided it falls back to defaults.
/// ///        Example: `heap!(chunks=16, chunksize=256)`.
/// static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
/// /// Backing storage for heap bookkeeping bitmap.
/// /// heap_bitmap!: first argument is amount of chunks.
/// ///               If no argument is provided it falls back to a default.
/// ///               Example: `heap_bitmap!(chunks=16)`.
/// static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();
///
/// // The backing memory must be CHUNK_SIZE-aligned; page alignment is preferable.
/// #[global_allocator]
/// static ALLOCATOR: GlobalChunkAllocator =
///     unsafe {
///         GlobalChunkAllocator::new_raw(
///             core::ptr::slice_from_raw_parts_mut(
///                 core::ptr::addr_of_mut!(HEAP).cast(),
///                 1048576,
///             ),
///             core::ptr::slice_from_raw_parts_mut(
///                 core::ptr::addr_of_mut!(HEAP_BITMAP).cast(),
///                 512,
///             ),
///         )
///     };
/// ```
#[derive(Debug)]
pub struct GlobalChunkAllocator<
    'a,
    const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE,
>(spin::Mutex<ChunkAllocator<'a, CHUNK_SIZE>>);

impl<'a, const CHUNK_SIZE: usize> GlobalChunkAllocator<'a, CHUNK_SIZE> {
    /// High-level constructor around
    /// [`ChunkAllocator::<CHUNK_SIZE>::new_const`].
    ///
    /// It is recommended that the heap and the bitmap both start at
    /// page-aligned addresses for better performance and to enable a faster
    /// search for correctly aligned addresses.
    ///
    /// WARNING: During const initialization it is not possible to check the
    /// alignment of the provided buffer. Please make sure that the data is
    /// at least aligned to the chunk_size. The recommended alignment is
    /// page-alignment.
    #[inline]
    pub const fn new(heap: &'a mut [u8], bitmap: &'a mut [u8]) -> Self {
        let inner_alloc = ChunkAllocator::<CHUNK_SIZE>::new_const(heap, bitmap);
        let inner_alloc = spin::Mutex::new(inner_alloc);
        Self(inner_alloc)
    }

    /// Creates a global allocator from raw backing-memory slices in a const
    /// context.
    ///
    /// # Safety
    /// The pointers must be valid, non-null, uniquely owned slices for the
    /// lifetime `'a`.
    #[inline]
    pub const unsafe fn new_raw(heap: *mut [u8], bitmap: *mut [u8]) -> Self {
        // SAFETY: required validity and exclusivity are guaranteed by the
        // caller.
        let inner_alloc =
            unsafe { ChunkAllocator::<CHUNK_SIZE>::new_raw(heap, bitmap) };
        Self(spin::Mutex::new(inner_alloc))
    }

    /// Wrapper around [`ChunkAllocator::usage`].
    #[inline]
    pub fn usage(&self) -> f32 {
        self.0.lock().usage()
    }

    /// Returns an instance of [`AllocatorApiGlue`].
    #[inline]
    pub const fn allocator_api_glue<'b>(
        &'b self,
    ) -> AllocatorApiGlue<'a, 'b, CHUNK_SIZE> {
        AllocatorApiGlue(self)
    }
}

// SAFETY: the mutex serializes access to the exclusively owned backing memory.
unsafe impl<'a, const CHUNK_SIZE: usize> GlobalAlloc
    for GlobalChunkAllocator<'a, CHUNK_SIZE>
{
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0
            .lock()
            .allocate(layout)
            .map_or(core::ptr::null_mut(), |allocation| allocation.as_mut_ptr())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `GlobalAlloc::dealloc` requires a valid allocation from this
        // allocator.
        unsafe { self.0.lock().deallocate(NonNull::new(ptr).unwrap(), layout) }
    }

    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        // SAFETY: `GlobalAlloc::realloc` requires a valid allocation from this
        // allocator.
        unsafe {
            self.0
                .lock()
                .realloc(NonNull::new(ptr).unwrap(), layout, new_size)
                .map_or(core::ptr::null_mut(), |allocation| {
                    allocation.as_mut_ptr()
                })
        }
    }
}

/// Helper struct generated by [`GlobalChunkAllocator`] that can be used in
/// structs that accept a custom instance of a specific allocator
/// implementation.
///
/// This is mainly relevant for testing the allocator. In a real environment, an
/// instance of [`GlobalChunkAllocator`] is likely registered a global allocator
/// for Rust. Thus, `Vec` and others default to the `Global` allocator, which
/// will use [`GlobalChunkAllocator`].
///
/// # Example
/// ```ignore
/// // ...
///
/// // Vector from the standard library will use my custom allocator instead.
/// let vec = Vec::<u8, _>::with_capacity_in(
///     123,
///     ALLOCATOR.allocator_api_glue(),
/// );
/// ```
#[derive(Debug)]
pub struct AllocatorApiGlue<'a, 'b, const CHUNK_SIZE: usize>(
    &'a GlobalChunkAllocator<'b, CHUNK_SIZE>,
);

// SAFETY: methods delegate to the mutex-protected `GlobalChunkAllocator`.
unsafe impl<'a, 'b, const CHUNK_SIZE: usize> Allocator
    for AllocatorApiGlue<'a, 'b, CHUNK_SIZE>
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut this = self.0.0.lock();
        ChunkAllocator::allocate(&mut *this, layout).map_err(|error| {
            log::error!("ChunkAllocatorError: {:?}", error);
            AllocError
        })
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut this = self.0.0.lock();
        // SAFETY: `Allocator::deallocate` requires a valid allocation from
        // `self`.
        unsafe { ChunkAllocator::deallocate(&mut *this, ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        assert!(
            old_layout.align() >= new_layout.align(),
            "change of alignment currenly not supported"
        );
        let mut this = self.0.0.lock();
        // SAFETY: `Allocator::grow` requires a valid allocation from `self`.
        unsafe { this.realloc(ptr, old_layout, new_layout.size()) }.map_err(
            |err| {
                log::error!("realloc error: {err:?}");
                AllocError
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PageAligned;
    use std::time::Instant;
    use std::vec::Vec;

    /// Uses [`GlobalChunkAllocator`] against the Rust Allocator API to test
    /// the underlying [`ChunkAllocator`]. This is like an "integration" test
    /// whereas the other tests in the other module are unit tests.
    #[test]
    fn test_allocator_with_allocator_api() {
        const CHUNK_COUNT: usize = 8;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
        static mut HEAP_MEM: PageAligned<[u8; HEAP_SIZE]> =
            PageAligned::new([0; HEAP_SIZE]);
        static mut BITMAP_MEM: PageAligned<[u8; BITMAP_SIZE]> =
            PageAligned::new([0; BITMAP_SIZE]);
        // SAFETY: these statics are exclusively owned by `ALLOCATOR`.
        static ALLOCATOR: GlobalChunkAllocator = unsafe {
            GlobalChunkAllocator::new_raw(
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(HEAP_MEM).cast(),
                    HEAP_SIZE,
                ),
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(BITMAP_MEM).cast(),
                    BITMAP_SIZE,
                ),
            )
        };

        assert_eq!(0.0, ALLOCATOR.usage());
        let vec1 = Vec::<u8, _>::with_capacity_in(
            DEFAULT_CHUNK_SIZE * 2,
            ALLOCATOR.allocator_api_glue(),
        );
        assert_eq!(25.0, ALLOCATOR.usage());
        let vec2 = Vec::<u8, _>::with_capacity_in(
            DEFAULT_CHUNK_SIZE * 6,
            ALLOCATOR.allocator_api_glue(),
        );
        assert_eq!(100.0, ALLOCATOR.usage());

        // I can't test it like this :( Because of the design of the types of
        // the Rust standard library, they fail if an allocation can't
        // be satisfied. However, they do not throw a normal kind
        // of panic but trigger the "rust alloc error" hook, which terminates
        // the program in a different manner.
        //let alloc_res = std::panic::catch_unwind(|| {
        //    let _vec = Vec::<u8, _>::with_capacity_in(1,
        // ALLOCATOR.allocator_api_glue());
        //});
        //assert!(panic_oom.is_err(), "allocator is out of memory");

        drop(vec1);
        assert_eq!(75.0, ALLOCATOR.usage());
        let vec3 = Vec::<u8, _>::with_capacity_in(
            DEFAULT_CHUNK_SIZE * 1,
            ALLOCATOR.allocator_api_glue(),
        );
        assert_eq!(87.5, ALLOCATOR.usage());

        drop(vec2);
        drop(vec3);
        assert_eq!(0.0, ALLOCATOR.usage());
    }

    /// Uses [`GlobalChunkAllocator`] against the Rust Allocator API to test
    /// if the realloc optimization works and is used.
    #[test]
    #[ignore = "performance benchmark"]
    fn test_allocator_fast_realloc_works() {
        const CHUNK_COUNT: usize = 32;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
        static mut HEAP_MEM: PageAligned<[u8; HEAP_SIZE]> =
            PageAligned::new([0; HEAP_SIZE]);
        static mut BITMAP_MEM: PageAligned<[u8; BITMAP_SIZE]> =
            PageAligned::new([0; BITMAP_SIZE]);
        // SAFETY: these statics are exclusively owned by `ALLOCATOR`.
        static ALLOCATOR: GlobalChunkAllocator = unsafe {
            GlobalChunkAllocator::new_raw(
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(HEAP_MEM).cast(),
                    HEAP_SIZE,
                ),
                core::ptr::slice_from_raw_parts_mut(
                    core::ptr::addr_of_mut!(BITMAP_MEM).cast(),
                    BITMAP_SIZE,
                ),
            )
        };

        // I run the allocation N times to measure the duration of it. This way
        // I can figure out if the shortcut was taken or not.
        const RUNS: usize = 10000;

        // TEST WITH FAST REALLOC
        let begin = Instant::now();
        for _ in 0..RUNS {
            let mut vec = Vec::<u8, _>::new_in(ALLOCATOR.allocator_api_glue());
            for i in 0..DEFAULT_CHUNK_SIZE {
                vec.resize(i, 42);
            }
        }
        let avg_duration_with_fast_realloc =
            (Instant::now() - begin).as_secs_f64() / RUNS as f64;

        // TEST WITHOUT FAST REALLOC
        let begin = Instant::now();
        for _ in 0..RUNS {
            let mut vec = Vec::<u8, _>::new_in(ALLOCATOR.allocator_api_glue());
            // realloc optimization can not be used; always requires one more
            // chunk
            vec.resize(DEFAULT_CHUNK_SIZE * 1 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 2 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 3 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 4 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 5 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 6 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 7 + 1, 42);
        }
        let avg_duration_without_fast_realloc =
            (Instant::now() - begin).as_secs_f64() / RUNS as f64;

        // almost always 3.6 or so but I use 2.6 so that test is not flaky
        // TODO the whole test is weird and probably not completely useful
        const FASTER_FACTOR_THRESHOLD: f64 = 2.0;
        let faster_factor =
            avg_duration_without_fast_realloc / avg_duration_with_fast_realloc;
        dbg!(
            avg_duration_without_fast_realloc / avg_duration_with_fast_realloc
        );
        dbg!(
            avg_duration_with_fast_realloc,
            avg_duration_without_fast_realloc
        );
        assert!(faster_factor >= FASTER_FACTOR_THRESHOLD);
    }
}
