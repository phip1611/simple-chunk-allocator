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
//! Module for [`GlobalChunkAllocator`].

use crate::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;

/// Default heap size in chunks (1 MiB with [`DEFAULT_CHUNK_SIZE`]).
pub const DEFAULT_CHUNK_AMOUNT: usize = 4096;

/// Thread-safe [`ChunkAllocator`] wrapper for use as a global allocator.
///
/// [`Self::allocator_api_glue`] exposes the nightly [`Allocator`] trait. See
/// the crate-level documentation for a complete `#[global_allocator]` setup.
#[derive(Debug)]
pub struct GlobalChunkAllocator<const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE>(
    spin::Mutex<ChunkAllocator<CHUNK_SIZE>>,
);

impl<const CHUNK_SIZE: usize> GlobalChunkAllocator<CHUNK_SIZE> {
    /// Creates a global allocator over a single caller-provided region.
    ///
    /// # Safety
    /// `region` and `region_len` must meet [`ChunkAllocator::new`]
    /// requirements for the allocator lifetime.
    #[inline]
    pub const unsafe fn new(region: *mut u8, region_len: usize) -> Self {
        // SAFETY: required validity and exclusivity are guaranteed by the
        // caller.
        let inner_alloc =
            unsafe { ChunkAllocator::<CHUNK_SIZE>::new(region, region_len) };
        Self(spin::Mutex::new(inner_alloc))
    }

    /// Wrapper around [`ChunkAllocator::required_region_size`].
    #[inline]
    pub const fn required_region_size(chunk_count: usize) -> usize {
        ChunkAllocator::<CHUNK_SIZE>::required_region_size(chunk_count)
    }

    /// Wrapper around [`ChunkAllocator::usage`].
    #[inline]
    pub fn usage(&self) -> f32 {
        self.0.lock().usage()
    }

    /// Returns an instance of [`AllocatorApiGlue`].
    #[inline]
    pub const fn allocator_api_glue(&self) -> AllocatorApiGlue<'_, CHUNK_SIZE> {
        AllocatorApiGlue(self)
    }
}

// SAFETY: the mutex serializes access to the exclusively owned backing memory.
unsafe impl<const CHUNK_SIZE: usize> GlobalAlloc
    for GlobalChunkAllocator<CHUNK_SIZE>
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

/// [`Allocator`] implementation backed by a [`GlobalChunkAllocator`].
///
/// Use this when the allocator is not registered as `#[global_allocator]` and
/// only selected collections should allocate from it.
///
/// # Example
/// ```rust
/// #![feature(allocator_api)]
/// use simple_chunk_allocator::GlobalChunkAllocator;
///
/// type Allocator = GlobalChunkAllocator<256>;
///
/// const REGION_SIZE: usize = Allocator::required_region_size(16);
/// static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];
///
/// // SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
/// static ALLOCATOR: Allocator =
///     unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };
///
/// // The vector allocates from ALLOCATOR; everything else keeps using the
/// // registered global allocator.
/// let mut vec = Vec::<u8, _>::with_capacity_in(
///     123,
///     ALLOCATOR.allocator_api_glue(),
/// );
/// vec.push(42);
/// assert!(ALLOCATOR.usage() > 0.0);
/// ```
#[derive(Debug)]
pub struct AllocatorApiGlue<'a, const CHUNK_SIZE: usize>(
    &'a GlobalChunkAllocator<CHUNK_SIZE>,
);

// SAFETY: methods delegate to the mutex-protected `GlobalChunkAllocator`.
unsafe impl<const CHUNK_SIZE: usize> Allocator
    for AllocatorApiGlue<'_, CHUNK_SIZE>
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut this = self.0.0.lock();
        ChunkAllocator::allocate(&mut *this, layout).map_err(|error| {
            log::error!("allocation failed: {error:?}");
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
            "change of alignment currently not supported"
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
    use std::time::Instant;
    use std::vec::Vec;

    /// Page-aligned backing memory.
    ///
    /// The allocator accepts any alignment, but a chunk-aligned region is the
    /// only one whose chunk count is exactly `SIZE / CHUNK_SIZE`, which the
    /// usage assertions below rely on.
    #[repr(align(4096))]
    struct Region<const SIZE: usize>([u8; SIZE]);

    /// Uses [`GlobalChunkAllocator`] against the Rust Allocator API to test
    /// the underlying [`ChunkAllocator`]. This is like an "integration" test
    /// whereas the other tests in the other module are unit tests.
    #[test]
    fn test_allocator_with_allocator_api() {
        const CHUNK_COUNT: usize = 8;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const REGION_SIZE: usize = HEAP_SIZE + CHUNK_COUNT.div_ceil(8);
        static mut REGION: Region<REGION_SIZE> = Region([0; REGION_SIZE]);
        // SAFETY: the static is exclusively owned by `ALLOCATOR`.
        static ALLOCATOR: GlobalChunkAllocator = unsafe {
            GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
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
            DEFAULT_CHUNK_SIZE,
            ALLOCATOR.allocator_api_glue(),
        );
        assert_eq!(87.5, ALLOCATOR.usage());

        drop(vec2);
        drop(vec3);
        assert_eq!(0.0, ALLOCATOR.usage());
    }

    #[test]
    fn test_global_alloc_returns_null_on_out_of_memory() {
        const CHUNK_COUNT: usize = 8;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const REGION_SIZE: usize = HEAP_SIZE + CHUNK_COUNT.div_ceil(8);
        static mut REGION: Region<REGION_SIZE> = Region([0; REGION_SIZE]);
        // SAFETY: the static is exclusively owned by `ALLOCATOR`.
        static ALLOCATOR: GlobalChunkAllocator = unsafe {
            GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
        };
        let layout = Layout::from_size_align(HEAP_SIZE, 1).unwrap();

        // SAFETY: `layout` is valid and the returned pointer is deallocated
        // below.
        let ptr = unsafe { GlobalAlloc::alloc(&ALLOCATOR, layout) };
        assert!(!ptr.is_null());
        // SAFETY: this valid request cannot fit while `ptr` is live.
        assert!(unsafe { GlobalAlloc::alloc(&ALLOCATOR, layout) }.is_null());
        // SAFETY: `ptr` is the live allocation returned for `layout`.
        unsafe { GlobalAlloc::dealloc(&ALLOCATOR, ptr, layout) };
    }

    /// Uses [`GlobalChunkAllocator`] against the Rust Allocator API to test
    /// if the realloc optimization works and is used.
    #[test]
    #[ignore = "performance benchmark"]
    fn test_allocator_fast_realloc_works() {
        const CHUNK_COUNT: usize = 32;
        const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
        const REGION_SIZE: usize = HEAP_SIZE + CHUNK_COUNT.div_ceil(8);
        static mut REGION: Region<REGION_SIZE> = Region([0; REGION_SIZE]);
        // SAFETY: the static is exclusively owned by `ALLOCATOR`.
        static ALLOCATOR: GlobalChunkAllocator = unsafe {
            GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
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
            let _ = vec;
        }
        let avg_duration_with_fast_realloc =
            (Instant::now() - begin).as_secs_f64() / RUNS as f64;

        // TEST WITHOUT FAST REALLOC
        let begin = Instant::now();
        for _ in 0..RUNS {
            let mut vec = Vec::<u8, _>::new_in(ALLOCATOR.allocator_api_glue());
            // realloc optimization can not be used; always requires one more
            // chunk
            #[allow(clippy::identity_op)]
            vec.resize(DEFAULT_CHUNK_SIZE * 1 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 2 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 3 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 4 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 5 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 6 + 1, 42);
            vec.resize(DEFAULT_CHUNK_SIZE * 7 + 1, 42);

            let _ = vec;
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
