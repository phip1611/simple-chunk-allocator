//! A simple `no_std` allocator written in Rust that manages memory in fixed-size chunks.
//! [`ChunkAllocator`] can be used as `#[global_allocator]` with the synchronized wrapper type
//! [`GlobalChunkAllocator`] as well as with the `allocator_api` feature. The latter enables the
//! usage in several types of the Rust standard library, such as `Vec` or `BTreeMap`. This is
//! primarily interesting for testing but may also enable other use cases. This project originates
//! from my [Diplom thesis project](https://github.com/phip1611/diplomarbeit-impl).
//!
//! Because I had lots of struggles to create this (my first ever allocator), I outsourced it for
//! better testability and to share my knowledge and findings with others in the hope that someone
//! can learn from it in any way.
//!
//! This is a chunk allocator or also called fixed-size block allocator. It uses a mixture of the
//! strategies next-fit and a best-fit. It tries to use the smallest gap for an allocation request
//! to prevent fragmentation but this is no guarantee. Each allocation is a trade-off between a low
//! allocation time and preventing fragmentation. The default chunk size is `256 bytes` but this
//! can be changed as compile time const generic. Having a fixed-size block allocator enables an
//! easy bookkeeping algorithm (a bitmap) but has as consequence that small allocations, such as
//! `64 byte` will take at least one chunk/block of the chosen block size.
//!
//! # Example
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
//! static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
//! /// Backing storage for heap bookkeeping bitmap. (read+write) static memory in final executable.
//! static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();
//!
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

#[macro_use]
mod macros;
mod allocator;
mod global;
mod page_aligned;

pub use allocator::*;
pub use global::*;
pub use macros::*;
pub use page_aligned::PageAligned;

#[cfg(test)]
#[macro_use]
extern crate std;
