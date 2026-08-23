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
//! Runs a whole test binary on top of [`GlobalChunkAllocator`].
//!
//! Registering it as `#[global_allocator]` is the crate's main use case and
//! the one path no unit test can take, because a process has exactly one
//! global allocator. Everything here - the test harness included - allocates
//! through it, so a bug in the bookkeeping shows up as corrupted data or a
//! failed allocation rather than as a failed assertion about internals.
//!
//! Together these allocate far more than the heap holds, so chunks that are
//! never released would exhaust it. That makes reuse an implicit precondition
//! of every test here; `tests/lifecycle.rs` checks it directly, where no other
//! thread is allocating at the same time.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use simple_chunk_allocator::GlobalChunkAllocator;

/// Named once, so that the chunk size is stated once.
type Allocator = GlobalChunkAllocator<64>;

/// Miri interprets every instruction, and the allocator scans its bitmap chunk
/// by chunk, so the heap and the workloads below are scaled down there. What
/// the tests check does not change; only how much of it they do.
#[cfg(miri)]
const SCALE: usize = 32;
#[cfg(not(miri))]
const SCALE: usize = 1;

/// A small chunk size and a generous region: the harness allocates in parallel
/// with the tests, and running out would abort the binary instead of failing a
/// test.
const REGION_SIZE: usize = Allocator::required_region_size(128 * 1024 / SCALE);
static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];

#[global_allocator]
// SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
static ALLOCATOR: Allocator =
    unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };

/// The region is a plain `[u8; N]` with no alignment guarantee, so this also
/// covers the padding the allocator inserts to align its first chunk.
#[test]
fn reports_a_usable_heap() {
    assert!(ALLOCATOR.capacity() >= 128 * 1024 / SCALE * 64);
    assert!(ALLOCATOR.usage() > 0.0, "the harness has allocated already");
}

/// Repeated growth is the operation that exercises realloc hardest: every
/// push beyond the capacity moves the buffer and must carry the contents
/// along.
#[test]
fn growing_collections_keep_their_contents() {
    let count = 10_000_u32 / SCALE as u32;
    let mut values = Vec::new();
    for value in 0..count {
        values.push(value);
    }
    assert!(values.iter().copied().eq(0..count));

    values.shrink_to_fit();
    assert!(values.iter().copied().eq(0..count));
}

/// Nodes of a map are many small, independent allocations with a lifetime that
/// does not follow the order they were made in, which is what fragments a
/// chunk heap.
#[test]
fn interleaved_allocations_stay_independent() {
    let count = 2_000_u32 / SCALE as u32;
    let mut map: BTreeMap<u32, String> = BTreeMap::new();
    for key in 0..count {
        let mut value = String::new();
        write!(value, "value-{key}").unwrap();
        map.insert(key, value);
    }

    // Dropping every second entry leaves holes for the next round to reuse.
    map.retain(|key, _| key.is_multiple_of(2));
    for key in (0..count).filter(|key| !key.is_multiple_of(2)) {
        map.insert(key, format!("again-{key}"));
    }

    for key in 0..count {
        let expected = if key.is_multiple_of(2) {
            format!("value-{key}")
        } else {
            format!("again-{key}")
        };
        assert_eq!(map[&key], expected);
    }
}

/// Boxed values with an alignment above the chunk size only fit at the few
/// addresses that satisfy them, so this drives the allocator's alignment
/// search rather than its free-chunk hint.
#[test]
fn over_aligned_allocations_are_aligned() {
    #[repr(align(4096))]
    struct PageAligned(u64);

    let values: Vec<_> =
        (0..16_u64).map(|i| Box::new(PageAligned(i))).collect();
    for (index, value) in values.iter().enumerate() {
        assert_eq!(
            std::ptr::from_ref::<PageAligned>(&**value).align_offset(4096),
            0
        );
        assert_eq!(value.0, index as u64);
    }
}
