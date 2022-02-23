//! Module for [`ChunkAllocator`].

#![no_std]
#![feature(allocator_api)]
#![feature(const_mut_refs)]
#![feature(const_for)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]

mod allocator;
mod global;

use allocator::*;
use global::*;

#[cfg(test)]
#[macro_use]
extern crate std;


