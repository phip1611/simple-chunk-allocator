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
//! caller-provided memory.
//!
//! [`ChunkAllocator`] takes a single contiguous region and splits it into
//! chunks plus the bitmap that tracks them. [`GlobalChunkAllocator`] wraps it
//! behind a lock for `#[global_allocator]` use. The region needs no particular
//! alignment and no initialization; see [`ChunkAllocator::new`].
//!
//! The crate requires nightly for `allocator_api`. See the README for sizing
//! guidance.
//!
//! ## Highlights
//!
//! - ✅ `no_std` allocator with test coverage
//! - ✅ uses a single caller-provided region as backing storage (no paging/page
//!   table manipulations)
//! - ✅ next-fit allocation that reuses the most recently freed region first
//! - ✅ reasonably fast with low code complexity
//! - ✅ const compatibility (no runtime `init()` required)
//! - ✅ efficient in scenarios where the heap is a few dozen megabytes in size
//! - ✅ small API: one constructor, no macros, no alignment wrappers
//!
//! ## Example
//!
//! The allocator manages one contiguous region: the chunks first, their
//! bitmap at the end. [`ChunkAllocator::required_region_size`] turns a chunk
//! count into the region size that is guaranteed to hold it.
//!
//! ```rust
//! #![feature(allocator_api)]
//! use simple_chunk_allocator::GlobalChunkAllocator;
//!
//! /// Named once, so that the chunk size is stated once.
//! type Allocator = GlobalChunkAllocator;
//!
//! const REGION_SIZE: usize = Allocator::required_region_size(4096);
//! static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];
//!
//! #[global_allocator]
//! // SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
//! static ALLOCATOR: Allocator =
//!     unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };
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
//! The bookkeeping is a bitmap with one bit per chunk. It lives at the end of
//! the same region as the chunks, so the allocator owns no memory of its own
//! and never touches paging or page tables. How many chunks fit therefore
//! depends on the region: each one costs `CHUNK_SIZE` bytes plus a bitmap bit.
//!
//! An allocation of `n` bytes occupies `ceil(n / CHUNK_SIZE)` consecutive
//! chunks. Small allocations thus occupy a whole chunk, which is the price for
//! keeping the bookkeeping at a single bit per chunk.
//!
//! The search starts at a hint and takes the first run that is long enough
//! and whose start address satisfies the requested alignment (next-fit).
//! Chunks start at `CHUNK_SIZE`-aligned addresses, so alignments up to the
//! chunk size always fit; a larger one is met by every
//! `alignment / CHUNK_SIZE`-th chunk.
//!
//! Deallocation points the hint at the freed region, so a buffer allocated
//! and freed over and over is handed back the memory it just released
//! instead of being searched for.
//!
//! Everything needed to build an allocator is `const`, so the allocator and
//! its backing memory can be set up at compile time and need no runtime
//! `init()`. The one thing a const context cannot do is look at an address,
//! which is what deciding where the chunks start requires. That happens on
//! first use instead, along with zeroing the bitmap.
//!
//! ## Safety
//!
//! The constructors require exclusive ownership of a valid backing region for
//! the allocator lifetime. Deallocation and reallocation require a live
//! pointer from the same allocator and its original layout.

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

mod allocator;
mod chunk_cache;
mod global;

pub use allocator::*;
pub use global::*;

#[cfg(test)]
#[macro_use]
extern crate std;
