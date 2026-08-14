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
//! Fixed-chunk allocation for caller-provided `no_std` memory.
//!
//! [`ChunkAllocator`] manages a heap in fixed-size chunks. For a global
//! allocator, use [`GlobalChunkAllocator`] with static, chunk-aligned heap and
//! bitmap storage. The bitmap needs exactly one bit for each heap chunk.
//!
//! The crate requires nightly for `allocator_api`. See the README for setup,
//! sizing guidance, and full examples.
//!
//! # Safety
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
