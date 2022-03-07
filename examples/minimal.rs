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
