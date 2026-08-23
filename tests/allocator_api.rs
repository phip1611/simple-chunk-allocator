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
//! Exercises [`GlobalChunkAllocator`] without registering it globally.
//!
//! This is the second way the crate is meant to be used: the allocator stays
//! a normal `static` and only the collections that name it allocate from it.
//! Unlike `tests/global_allocator.rs`, nothing else allocates here, so these
//! tests can assert on exact usage numbers.

#![feature(allocator_api)]

use core::alloc::{Allocator, GlobalAlloc, Layout};
use simple_chunk_allocator::{DEFAULT_CHUNK_SIZE, GlobalChunkAllocator};

mod common;

use common::StaticRegion;

const CHUNK_COUNT: usize = 8;
const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * CHUNK_COUNT;
const REGION_SIZE: usize = HEAP_SIZE + CHUNK_COUNT.div_ceil(8);

/// A chunk-aligned region of eight chunks must report exactly those, so that
/// every allocation below moves `usage` by a known eighth.
#[test]
fn usage_follows_live_collections() {
    static mut REGION: StaticRegion<REGION_SIZE> =
        StaticRegion([0; REGION_SIZE]);
    // SAFETY: `ALLOCATOR` is the only user of `REGION`, and this test runs
    // once. Every test declares its own so that they do not share usage.
    static ALLOCATOR: GlobalChunkAllocator = unsafe {
        GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
    };

    assert_eq!(ALLOCATOR.chunk_count(), CHUNK_COUNT);
    assert_eq!(ALLOCATOR.capacity(), HEAP_SIZE);
    assert_eq!(ALLOCATOR.usage(), 0.0);

    let vec1 = Vec::<u8, _>::with_capacity_in(
        DEFAULT_CHUNK_SIZE * 2,
        ALLOCATOR.allocator_api_glue(),
    );
    assert_eq!(ALLOCATOR.usage(), 0.25);
    let vec2 = Vec::<u8, _>::with_capacity_in(
        DEFAULT_CHUNK_SIZE * 6,
        ALLOCATOR.allocator_api_glue(),
    );
    assert_eq!(ALLOCATOR.usage(), 1.0);

    drop(vec1);
    assert_eq!(ALLOCATOR.usage(), 0.75);
    let vec3 = Vec::<u8, _>::with_capacity_in(
        DEFAULT_CHUNK_SIZE,
        ALLOCATOR.allocator_api_glue(),
    );
    assert_eq!(ALLOCATOR.usage(), 0.875);

    drop(vec2);
    drop(vec3);
    assert_eq!(ALLOCATOR.usage(), 0.0);
}

/// `GlobalAlloc` has no error type: exhaustion has to arrive as a null
/// pointer, not as a panic.
///
/// A collection cannot stand in for this. When an allocation fails, the
/// standard library calls the allocation error handler, which aborts the
/// process instead of unwinding, so `catch_unwind` cannot observe it.
#[test]
fn global_alloc_returns_null_on_out_of_memory() {
    static mut REGION: StaticRegion<REGION_SIZE> =
        StaticRegion([0; REGION_SIZE]);
    // SAFETY: `ALLOCATOR` is the only user of `REGION`, and this test runs
    // once. Every test declares its own so that they do not share usage.
    static ALLOCATOR: GlobalChunkAllocator = unsafe {
        GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
    };

    let layout = Layout::from_size_align(HEAP_SIZE, 1).unwrap();

    // SAFETY: `layout` is valid and the returned pointer is deallocated below.
    let ptr = unsafe { GlobalAlloc::alloc(&ALLOCATOR, layout) };
    assert!(!ptr.is_null());
    // SAFETY: this valid request cannot fit while `ptr` is live.
    assert!(unsafe { GlobalAlloc::alloc(&ALLOCATOR, layout) }.is_null());
    // SAFETY: `ptr` is the live allocation returned for `layout`.
    unsafe { GlobalAlloc::dealloc(&ALLOCATOR, ptr, layout) };
}

/// Growing inside the chunks an allocation already owns must reuse them.
///
/// Allocations are rounded up to whole chunks, so most growth steps of a
/// `Vec` need no new memory at all. Skipping the copy is what makes that
/// cheap, and the observable difference is that the pointer stays put.
#[test]
fn realloc_reuses_the_chunks_an_allocation_already_owns() {
    static mut REGION: StaticRegion<REGION_SIZE> =
        StaticRegion([0; REGION_SIZE]);
    // SAFETY: `ALLOCATOR` is the only user of `REGION`, and this test runs
    // once. Every test declares its own so that they do not share usage.
    static ALLOCATOR: GlobalChunkAllocator = unsafe {
        GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
    };

    let layout = Layout::from_size_align(1, 1).unwrap();

    // SAFETY: `layout` is valid; the pointer is reallocated and freed below.
    let ptr = unsafe { GlobalAlloc::alloc(&ALLOCATOR, layout) };
    assert!(!ptr.is_null());
    assert_eq!(ALLOCATOR.usage(), 1.0 / CHUNK_COUNT as f32);

    // Still within the first chunk.
    // SAFETY: `ptr` is live and paired with `layout`.
    let grown = unsafe {
        GlobalAlloc::realloc(&ALLOCATOR, ptr, layout, DEFAULT_CHUNK_SIZE)
    };
    assert_eq!(grown, ptr, "growth inside the chunk must not move the data");
    assert_eq!(ALLOCATOR.usage(), 1.0 / CHUNK_COUNT as f32);

    // One byte beyond it, which needs a second chunk and therefore a copy.
    let grown_layout = Layout::from_size_align(DEFAULT_CHUNK_SIZE, 1).unwrap();
    // SAFETY: `grown` is live and paired with `grown_layout`.
    let moved = unsafe {
        GlobalAlloc::realloc(
            &ALLOCATOR,
            grown,
            grown_layout,
            DEFAULT_CHUNK_SIZE + 1,
        )
    };
    assert!(!moved.is_null());
    assert_eq!(ALLOCATOR.usage(), 2.0 / CHUNK_COUNT as f32);

    let moved_layout =
        Layout::from_size_align(DEFAULT_CHUNK_SIZE + 1, 1).unwrap();
    // SAFETY: `moved` is the live allocation for `moved_layout`.
    unsafe { GlobalAlloc::dealloc(&ALLOCATOR, moved, moved_layout) };
    assert_eq!(ALLOCATOR.usage(), 0.0);
}

/// Growing a collection is the only way `Allocator::grow` gets called, and no
/// other test reaches it: `with_capacity_in` allocates once and never resizes.
#[test]
fn growing_a_collection_moves_it_only_when_it_has_to() {
    static mut REGION: StaticRegion<REGION_SIZE> =
        StaticRegion([0; REGION_SIZE]);
    // SAFETY: `ALLOCATOR` is the only user of `REGION`, and this test runs
    // once. Every test declares its own so that they do not share usage.
    static ALLOCATOR: GlobalChunkAllocator = unsafe {
        GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
    };

    // Starting at one element forces a growth step per doubling. The first
    // steps stay inside the chunk the allocation already owns; the later ones
    // have to move it, and both paths must preserve the contents.
    let mut values = Vec::with_capacity_in(1, ALLOCATOR.allocator_api_glue());
    for value in 0..512_u16 {
        values.push(value);
    }
    assert!(values.iter().copied().eq(0..512));

    values.truncate(8);
    values.shrink_to_fit();
    assert!(values.iter().copied().eq(0..8));
    assert_eq!(ALLOCATOR.usage(), 1.0 / CHUNK_COUNT as f32);

    drop(values);
    assert_eq!(ALLOCATOR.usage(), 0.0);
}

/// An allocation cannot gain alignment on the way: the underlying `realloc`
/// keeps the one it was made with. `grow` has to report that instead of
/// unwinding, which a caller in a `no_std` binary could not recover from.
#[test]
fn grow_refuses_an_alignment_it_cannot_provide() {
    static mut REGION: StaticRegion<REGION_SIZE> =
        StaticRegion([0; REGION_SIZE]);
    // SAFETY: `ALLOCATOR` is the only user of `REGION`, and this test runs
    // once. Every test declares its own so that they do not share usage.
    static ALLOCATOR: GlobalChunkAllocator = unsafe {
        GlobalChunkAllocator::new((&raw mut REGION).cast(), REGION_SIZE)
    };

    let glue = ALLOCATOR.allocator_api_glue();
    let layout = Layout::from_size_align(8, 8).unwrap();
    let ptr = glue.allocate(layout).unwrap().cast::<u8>();
    let stricter = Layout::from_size_align(16, DEFAULT_CHUNK_SIZE * 2).unwrap();

    // SAFETY: `ptr` is live, made for `layout`, and `stricter` is larger.
    assert!(unsafe { glue.grow(ptr, layout, stricter) }.is_err());
    assert_eq!(
        ALLOCATOR.usage(),
        1.0 / CHUNK_COUNT as f32,
        "the refused grow must leave the allocation alone"
    );

    // SAFETY: `ptr` is still the live allocation made for `layout`.
    unsafe { glue.deallocate(ptr, layout) };
    assert_eq!(ALLOCATOR.usage(), 0.0);
}
