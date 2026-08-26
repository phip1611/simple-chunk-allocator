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
use core::ptr::{self, NonNull};
use log::error;

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

    /// Wrapper around [`ChunkAllocator::chunk_size`].
    #[inline]
    pub const fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }

    /// Wrapper around [`ChunkAllocator::min_alignment`].
    #[inline]
    pub const fn min_alignment(&self) -> usize {
        CHUNK_SIZE
    }

    /// Wrapper around [`ChunkAllocator::capacity`].
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.lock().capacity()
    }

    /// Wrapper around [`ChunkAllocator::chunk_count`].
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.0.lock().chunk_count()
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
            .map_or(ptr::null_mut(), |allocation| {
                allocation.cast::<u8>().as_ptr()
            })
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
                .map_or(ptr::null_mut(), |allocation| {
                    allocation.cast::<u8>().as_ptr()
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

impl<const CHUNK_SIZE: usize> AllocatorApiGlue<'_, CHUNK_SIZE> {
    /// Resizes an allocation, in place where the chunks it already owns are
    /// enough.
    ///
    /// Shared by [`Allocator::grow`] and [`Allocator::shrink`], which differ
    /// only in the direction the caller promises to resize in.
    ///
    /// # Safety
    /// `ptr` must be a live allocation from this allocator, and `old_layout`
    /// must be the layout it was made with.
    #[inline]
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // `ChunkAllocator::realloc` keeps the original alignment, so a
        // stricter one cannot be served. Refuse it rather than panic: an
        // allocator that unwinds is far harder to use than one that reports
        // failure, and the trait allows reporting it.
        if new_layout.align() > old_layout.align() {
            error!(
                "cannot resize from alignment {} to {}",
                old_layout.align(),
                new_layout.align()
            );
            return Err(AllocError);
        }

        let mut this = self.0.0.lock();
        // SAFETY: the caller guarantees a live allocation and its layout.
        unsafe { this.realloc(ptr, old_layout, new_layout.size()) }.map_err(
            |error| {
                error!("resize failed: {error:?}");
                AllocError
            },
        )
    }
}

// SAFETY: methods delegate to the mutex-protected `GlobalChunkAllocator`.
unsafe impl<const CHUNK_SIZE: usize> Allocator
    for AllocatorApiGlue<'_, CHUNK_SIZE>
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut this = self.0.0.lock();
        ChunkAllocator::allocate(&mut *this, layout).map_err(|error| {
            error!("allocation failed: {error:?}");
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
        // SAFETY: `Allocator::grow` requires a valid allocation from `self`.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }

    /// Overrides the default, which always allocates, copies and frees.
    ///
    /// An allocation that shrinks within the chunks it already owns keeps
    /// them, so the copy is avoidable; only the chunks behind the new size are
    /// released. Rounding up to whole chunks makes that the common case.
    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `Allocator::shrink` requires a valid allocation from `self`.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }
}
