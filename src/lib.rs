//! Module for [`ChunkAllocator`].

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
