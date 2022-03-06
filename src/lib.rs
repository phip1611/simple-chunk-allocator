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
//! # Code Example
//! ```ignore
//! #![feature(allocator_api)]
//!
//! use simple_chunk_allocator::{DEFAULT_CHUNK_SIZE, GlobalChunkAllocator};
//!
//! const HEAP_SIZE: usize = DEFAULT_CHUNK_SIZE * 8;
//! const CHUNK_COUNT: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE;
//! const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
//!
//! static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
//! static mut BITMAP_MEM: [u8; BITMAP_SIZE] = [0; BITMAP_SIZE];
//!
//! #[global_allocator]
//! static ALLOCATOR: GlobalChunkAllocator = GlobalChunkAllocator::new();
//!
//! fn entry() {
//!     unsafe {
//!         ALLOCATOR.init(HEAP_MEM.as_mut_slice(), BITMAP_MEM.as_mut_slice())
//!           .unwrap()
//!     };
//!
//!     // use global allocator
//!     let _boxed_array = Box::new([0, 1, 2, 3, 4]);
//!     let _msg = String::from("hello, world");
//!
//!     // example: allocator_api-feature
//!     let _vec = Vec::<u8, _ >::with_capacity_in(123, ALLOCATOR.allocator_api_glue());
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

mod allocator;
mod global;

pub use allocator::*;
pub use global::*;

#[cfg(test)]
#[macro_use]
extern crate std;
