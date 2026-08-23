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
//! # Simple Chunk Allocator
//!
//! A nightly-only `no_std` allocator that manages fixed-size chunks in
//! caller-provided static memory.
//!
//! [`ChunkAllocator`] manages a heap in fixed-size chunks. For a global
//! allocator, use [`GlobalChunkAllocator`] with static, chunk-aligned heap and
//! bitmap storage. The bitmap needs exactly one bit for each heap chunk.
//!
//! The crate requires nightly for `allocator_api`. See the README for sizing
//! guidance.
//!
//! ## Highlights
//!
//! - ✅ `no_std` allocator with test coverage
//! - ✅ uses static memory as backing storage (no paging/page table
//!   manipulations)
//! - ✅ allocation strategy is a combination of next-fit and best-fit
//! - ✅ reasonably fast with low code complexity
//! - ✅ const compatibility (no runtime `init()` required)
//! - ✅ efficient in scenarios where the heap is a few dozen megabytes in size
//! - ✅ user-friendly API
//!
//! ## Example
//!
//! The macros derive the storage types from the chunk geometry, so the same
//! constants describe the arrays and the slices handed to the allocator:
//!
//! ```rust
//! #![feature(allocator_api)]
//! use simple_chunk_allocator::{
//!     DEFAULT_CHUNK_AMOUNT, DEFAULT_CHUNK_SIZE, GlobalChunkAllocator,
//!     PageAligned, heap, heap_bitmap,
//! };
//!
//! const CHUNKS: usize = DEFAULT_CHUNK_AMOUNT;
//! const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;
//!
//! static mut HEAP: PageAligned<[u8; CHUNKS * CHUNK_SIZE]> =
//!     heap!(chunks = CHUNKS, chunksize = CHUNK_SIZE);
//! static mut BITMAP: PageAligned<[u8; CHUNKS / 8]> =
//!     heap_bitmap!(chunks = CHUNKS);
//!
//! #[global_allocator]
//! // SAFETY: ALLOCATOR exclusively owns both statics for the whole program.
//! static ALLOCATOR: GlobalChunkAllocator<CHUNK_SIZE> = unsafe {
//!     GlobalChunkAllocator::new(
//!         core::ptr::slice_from_raw_parts_mut(
//!             core::ptr::addr_of_mut!(HEAP).cast(),
//!             CHUNKS * CHUNK_SIZE,
//!         ),
//!         core::ptr::slice_from_raw_parts_mut(
//!             core::ptr::addr_of_mut!(BITMAP).cast(),
//!             CHUNKS / 8,
//!         ),
//!     )
//! };
//!
//! fn main() {
//!     // In a hosted binary the runtime already allocated before `main`.
//!     let old_usage = ALLOCATOR.usage();
//!     let mut values = Vec::new();
//!     values.push(42);
//!     assert!(ALLOCATOR.usage() > old_usage);
//! }
//! ```
//!
//! [`AllocatorApiGlue`] serves the same allocator to individual collections
//! when it is not registered globally.
//!
//! ## Implementation
//!
//! The bookkeeping is a bitmap with one bit per chunk, stored in memory the
//! caller provides. The allocator therefore owns no memory of its own and
//! never touches paging or page tables.
//!
//! An allocation of `n` bytes occupies `ceil(n / CHUNK_SIZE)` consecutive
//! chunks. Small allocations thus occupy a whole chunk, which is the price for
//! keeping the bookkeeping at a single bit per chunk.
//!
//! The search for those chunks starts at a cached hint and takes the first run
//! that is long enough and whose start address satisfies the requested
//! alignment (next-fit). Chunks start at `CHUNK_SIZE`-aligned addresses, so
//! alignments up to the chunk size always fit; larger alignments make the
//! search skip candidates. Deallocation moves the hint to the freed region
//! when that region is smaller than the cached one, which biases the next
//! allocation towards the smallest recent gap (best-fit) and slows down
//! fragmentation.
//!
//! Everything needed to build an allocator is `const`, so the allocator and
//! its backing memory can be set up at compile time and need no runtime
//! `init()`. Heap alignment is the one property that a const context cannot
//! check; it is validated on the first allocation instead.
//!
//! ## Safety
//!
//! Constructors that accept raw slices require exclusive ownership of valid,
//! non-overlapping backing storage for the allocator lifetime. Deallocation and
//! reallocation require a live pointer from the same allocator and its original
//! layout.

#![no_std]
#![deny(
    clippy::all,
    clippy::cargo,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks,
    // clippy::restriction,
    // clippy::pedantic
)]
// Allow a few noisy lints that do not improve this crate.
#![allow(
    clippy::suboptimal_flops,
    clippy::redundant_pub_crate,
    clippy::fallible_impl_from
)]
#![deny(missing_debug_implementations)]
#![deny(rustdoc::all)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]

#[macro_use]
mod macros;
mod allocator;
mod chunk_cache;
mod global;
mod page_aligned;

pub use allocator::*;
pub use global::*;
pub use page_aligned::PageAligned;

#[cfg(test)]
#[macro_use]
extern crate std;
