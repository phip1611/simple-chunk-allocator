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
#![feature(allocator_api)]

use simple_chunk_allocator::{
    GlobalChunkAllocator, PageAligned, heap, heap_bitmap,
};

/// Page-aligned heap storage (1 MiB).
///
/// Without arguments the macros fall back to `DEFAULT_CHUNK_AMOUNT` chunks of
/// `DEFAULT_CHUNK_SIZE` bytes. Both accept the geometry explicitly, e.g.
/// `heap!(chunks = 16, chunksize = 256)`.
static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
/// One bitmap bit per heap chunk, e.g. `heap_bitmap!(chunks = 16)`.
static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();

#[global_allocator]
// SAFETY: these statics are exclusively owned by `ALLOCATOR`.
static ALLOCATOR: GlobalChunkAllocator = unsafe {
    GlobalChunkAllocator::new_raw(
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(HEAP).cast(),
            1048576,
        ),
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(HEAP_BITMAP).cast(),
            512,
        ),
    )
};

fn main() {
    // The Rust runtime already allocated before `main` was entered, so the
    // usage is compared against the current value instead of zero. A `no_std`
    // binary starts with an empty heap.
    let old_usage = ALLOCATOR.usage();

    #[allow(clippy::vec_init_then_push)]
    {
        let mut vec = Vec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert!(ALLOCATOR.usage() > old_usage);
    }

    // Use the allocator API explicitly when it is not globally registered.
    let _boxed = Box::new_in([1, 2, 3], ALLOCATOR.allocator_api_glue());
}
