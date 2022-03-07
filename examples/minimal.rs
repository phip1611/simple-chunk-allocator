#![feature(const_mut_refs)]
#![feature(allocator_api)]

use simple_chunk_allocator::{heap, heap_bitmap, GlobalChunkAllocator, PageAligned};

// The macros help to get a correctly sized arrays types.
// I page-align them for better caching and to improve the availability of
// page-aligned addresses.

/// Backing storage for heap (1Mib). (read+write) static memory in final executable.
static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
/// Backing storage for heap bookkeeping bitmap. (read+write) static memory in final executable.
static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();

#[global_allocator]
static ALLOCATOR: GlobalChunkAllocator =
    unsafe { GlobalChunkAllocator::new(HEAP.deref_mut_const(), HEAP_BITMAP.deref_mut_const()) };

fn main() {
    // at this point, the allocator already got used a bit by the Rust runtime that executes
    // before main() gets called. This is not the case if a `no_std` binary gets produced.
    let old_usage = ALLOCATOR.usage();
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    assert!(ALLOCATOR.usage() > old_usage);

    // use "allocator_api"-feature. You can use this if "ALLOCATOR" is not registered as
    // the global allocator. Otherwise, it is already the default.
    let _boxed = Box::new_in([1, 2, 3], ALLOCATOR.allocator_api_glue());
}
