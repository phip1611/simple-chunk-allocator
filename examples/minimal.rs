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

use simple_chunk_allocator::GlobalChunkAllocator;

/// The allocator type, named once so that the chunk size is stated once.
/// Without an argument it uses `DEFAULT_CHUNK_SIZE`.
type Allocator = GlobalChunkAllocator;

/// Backing memory: 1 MiB worth of chunks plus the bitmap that tracks them.
///
/// The region may start at any address, and the allocator skips up to
/// `CHUNK_SIZE - 1` bytes to align the first chunk. `required_region_size`
/// budgets for that, so the 4096 chunks are there in any case.
const CHUNKS: usize = 4096;
const REGION_SIZE: usize = Allocator::required_region_size(CHUNKS);
static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];

#[global_allocator]
// SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
static ALLOCATOR: Allocator =
    unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };

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
