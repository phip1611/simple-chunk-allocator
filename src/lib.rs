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
//! # Simple Chunk Allocator
//!
//! A simple `no_std` allocator written in Rust that manages memory in fixed-size chunks/blocks. Useful for basic `no_std`
//! binaries where you want to manage a heap of a few megabytes without complex features such as paging/page table
//! management. Instead, this allocator gets a fixed/static memory region and allocates memory from there. This memory
//! region can be contained inside the executable file that uses this allocator. See examples down below.
//!
//! ⚠ _There probably exist better solutions for large-scale applications that have better performance by using a
//! more complex algorithm. However, this is good for simple `no_std` binaries and hopefully also for educational
//! purposes. It helped me to understand a lot about allocators._ ⚠
//!
//! ## TL;DR
//! - ✅ `no_std` allocator with test coverage
//! - ✅ uses static memory as backing storage (no paging/page table manipulations)
//! - ✅ allocation strategy is a combination of next-fit and best-fit
//! - ✅ reasonable fast with low code complexity
//! - ✅ const compatibility (no runtime `init()` required)
//! - ✅ efficient in scenarios where heap is a few dozens megabytes in size
//! - ✅ user-friendly API
//!
//! The inner and low-level `ChunkAllocator` can be used as `#[global_allocator]` with the synchronized wrapper type
//! `GlobalChunkAllocator`. Both can be used with the `allocator_api` feature. The latter enables the usage in several
//! types of the Rust standard library, such as `Vec::new_in` or `BTreeMap::new_in`. This is primarily interesting for
//! testing but may also enable other interesting use-cases.
//!
//! The focus is on `const` compatibility. The allocator and the backing memory can get initialized during compile time
//! and need no runtime `init()` call or similar. This means that if the compiler accepts it then the allocation will
//! also work during runtime. However, you can also create allocator objects during runtime.
//!
//! The inner and low-level `ChunkAllocator` is a chunk allocator or also called fixed-size block allocator. It uses a
//! mixture of the strategies next-fit and a best-fit. It tries to use the smallest gap for an allocation request to
//! prevent fragmentation but this is no guarantee. Each allocation is a trade-off between a low allocation time and
//! preventing fragmentation. The default chunk size is `256 bytes` but this can be changed as compile time const generic.
//! Having a fixed-size block allocator enables an easy bookkeeping algorithm through a bitmap but has as consequence that
//! small allocations, such as `64 byte` will take at least one chunk/block of the chosen block size.
//!
//! This project originates from my [Diplom thesis project](https://github.com/phip1611/diplomarbeit-impl). Since I
//! originally had lots of struggles to create this (my first ever allocator), I outsourced it for better testability and
//! to share my knowledge and findings with others in the hope that someone can learn from it in any way.
//!
//!
//! # Minimal Code Example
//!
//! ```rust
//! #![feature(const_mut_refs)]
//! #![feature(allocator_api)]
//!
//! use simple_chunk_allocator::{heap, heap_bitmap, GlobalChunkAllocator, PageAligned};
//!
//! // The macros help to get a correctly sized arrays types.
//! // I page-align them for better caching and to improve the availability of
//! // page-aligned addresses.
//!
//! /// Backing storage for heap (1Mib). (read+write) static memory in final executable.
//! ///
//! /// heap!: first argument is chunk amount, second argument is size of each chunk.
//! ///        If no arguments are provided it falls back to defaults.
//! ///        Example: `heap!(chunks=16, chunksize=256)`.
//! static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
//! /// Backing storage for heap bookkeeping bitmap. (read+write) static memory in final executable.
//! ///
//! /// heap_bitmap!: first argument is amount of chunks.
//! ///               If no argument is provided it falls back to a default.
//! ///               Example: `heap_bitmap!(chunks=16)`.
//! static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();
//!
//! // please make sure that the backing memory is at least CHUNK_SIZE aligned; better page-aligned
//! #[global_allocator]
//! static ALLOCATOR: GlobalChunkAllocator =
//!     unsafe { GlobalChunkAllocator::new(HEAP.deref_mut_const(), HEAP_BITMAP.deref_mut_const()) };
//!
//! fn main() {
//!     // at this point, the allocator already got used a bit by the Rust runtime that executes
//!     // before main() gets called. This is not the case if a `no_std` binary gets produced.
//!     let old_usage = ALLOCATOR.usage();
//!     let mut vec = Vec::new();
//!     vec.push(1);
//!     vec.push(2);
//!     vec.push(3);
//!     assert!(ALLOCATOR.usage() > old_usage);
//!
//!     // use "allocator_api"-feature. You can use this if "ALLOCATOR" is not registered as
//!     // the global allocator. Otherwise, it is already the default.
//!     let _boxed = Box::new_in([1, 2, 3], ALLOCATOR.allocator_api_glue());
//! }
//! ```

#![no_std]
#![deny(
    clippy::all,
    clippy::cargo,
    clippy::nursery,
    // clippy::restriction,
    // clippy::pedantic
)]
// now allow a few rules which are denied by the above statement
// --> they are ridiculous and not necessary
#![allow(
    clippy::suboptimal_flops,
    clippy::redundant_pub_crate,
    clippy::fallible_impl_from
)]
#![deny(missing_debug_implementations)]
#![deny(rustdoc::all)]
#![feature(allocator_api)]
#![feature(const_mut_refs)]
#![feature(const_for)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(const_ptr_is_null)]
#![feature(core_intrinsics)]
#![feature(const_align_offset)]

#[macro_use]
mod macros;
mod allocator;
mod chunk_cache;
mod compiler_hints;
mod global;
mod page_aligned;

pub use allocator::*;
pub use global::*;
pub use macros::*;
pub use page_aligned::PageAligned;

#[cfg(test)]
#[macro_use]
extern crate std;
