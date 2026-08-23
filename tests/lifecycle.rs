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
//! Drives [`ChunkAllocator`] through complete allocation cycles using nothing
//! but its public API.
//!
//! The unit tests in the crate check the pieces in isolation and reach into
//! private state to do so. These tests answer the other question: does the
//! surface the crate actually exposes carry a real workload from an empty heap
//! to a full one and back?

use core::alloc::Layout;
use simple_chunk_allocator::{ChunkAllocator, OutOfMemory};

mod common;

use common::{Allocation, Region};

const CHUNK_SIZE: usize = 256;

/// Creates an allocator over `region`.
///
/// # Safety
/// `region` must outlive the returned allocator and must not be used
/// otherwise.
unsafe fn allocator_over<const CHUNK_SIZE: usize>(
    region: &mut [u8],
) -> ChunkAllocator<CHUNK_SIZE> {
    // SAFETY: forwarded to the caller.
    unsafe { ChunkAllocator::new(region.as_mut_ptr(), region.len()) }
}

/// A region too small for one chunk is a mistake the constructor rejects, so
/// that a `static` sized wrong fails to compile rather than yielding an
/// allocator that is out of memory on every request.
#[test]
fn a_region_too_small_for_a_chunk_is_rejected() {
    for len in [0, 1, CHUNK_SIZE, 2 * CHUNK_SIZE - 1] {
        std::panic::catch_unwind(move || {
            let mut region = vec![0_u8; len];
            // SAFETY: `region` outlives the allocator and is not used
            // otherwise.
            let _allocator =
                unsafe { allocator_over::<CHUNK_SIZE>(&mut region) };
        })
        .expect_err("region too small for one chunk");
    }
}

/// Storage sized with `required_region_size` must deliver the requested chunks
/// wherever it ends up in memory.
#[test]
fn required_region_size_covers_every_alignment() {
    let mut backing = Region::new(
        ChunkAllocator::<CHUNK_SIZE>::required_region_size(64) + CHUNK_SIZE,
        0,
    );

    for chunk_count in [1, 7, 8, 9, 64] {
        let len =
            ChunkAllocator::<CHUNK_SIZE>::required_region_size(chunk_count);
        for offset in [0, 1, CHUNK_SIZE - 1, CHUNK_SIZE] {
            let region = &mut backing.as_mut_slice()[offset..offset + len];
            // SAFETY: `backing` outlives the allocator and the window is used
            // by nothing else.
            let allocator = unsafe { allocator_over::<CHUNK_SIZE>(region) };
            assert!(
                allocator.chunk_count() >= chunk_count,
                "asked for {chunk_count} chunks at offset {offset}, got {}",
                allocator.chunk_count()
            );
        }
    }
}

/// A region handed over by a bootloader holds whatever was there before, so
/// the allocator has to bring its own bitmap into a defined state. If it did
/// not, the chunks a dirty bitmap claims as used would never be handed out.
#[test]
fn dirty_region_starts_out_empty() {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(16);
    backing.as_mut_slice().fill(0xff);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };

    assert_eq!(allocator.usage(), 0.0);
    let whole_heap = Layout::from_size_align(allocator.capacity(), 1).unwrap();
    assert!(
        allocator.allocate(whole_heap).is_ok(),
        "a single chunk left over from the previous content would break this"
    );
}

/// Fills a heap of `chunk_count` chunks completely, checks that nothing else
/// fits and that no allocation lost its contents, then empties it again.
///
/// Every allocation carries its own byte pattern, so an overlap between two of
/// them - or between the last chunk and the bitmap that sits right behind it -
/// shows up as a foreign byte. Miri cannot see either: the whole region is a
/// single allocation to it, so overruns and use-after-free inside it are
/// invisible. These patterns are the only check for that.
fn fill_and_empty_heap<const CHUNK_SIZE: usize>(chunk_count: usize) {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(chunk_count);
    let region = backing.as_mut_slice();
    let region_start = region.as_ptr() as usize;
    let region_end = region_start + region.len();
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator = unsafe { allocator_over::<CHUNK_SIZE>(region) };
    assert_eq!(allocator.chunk_count(), chunk_count);
    let layout = Layout::from_size_align(CHUNK_SIZE, 1).unwrap();

    let allocations: Vec<_> = (0..chunk_count)
        .map(|index| Allocation {
            ptr: allocator.allocate(layout).unwrap().cast(),
            layout,
            // Never zero, so that a byte left over from the bitmap stands out.
            pattern: index as u8 | 0x80,
        })
        .collect();
    allocations.iter().for_each(Allocation::fill);

    for allocation in &allocations {
        let begin = allocation.ptr.as_ptr() as usize;
        assert!(
            begin >= region_start && begin + CHUNK_SIZE <= region_end,
            "chunk at {begin:#x} leaves the region \
             {region_start:#x}..{region_end:#x}"
        );
    }

    assert_eq!(allocator.usage(), 1.0);
    assert_eq!(allocator.allocate(layout), Err(OutOfMemory));

    for allocation in allocations {
        allocation.assert_pattern(layout.size());
        // SAFETY: every record is a live allocation with its layout.
        unsafe { allocator.deallocate(allocation.ptr, allocation.layout) };
    }

    assert_eq!(allocator.usage(), 0.0);
    let whole_heap = Layout::from_size_align(allocator.capacity(), 1).unwrap();
    assert!(
        allocator.allocate(whole_heap).is_ok(),
        "the heap must be one continuous free region again"
    );
}

/// The bitmap shares its region with the chunks, so a geometry that is one
/// chunk too generous would let the last allocation scribble over the
/// bookkeeping.
///
/// A chunk count that is not a multiple of eight is the interesting case: the
/// bitmap rounds up to whole bytes, leaving spare bits in the last one.
/// Treating those as chunks would hand out memory past the end of the region.
#[test]
fn allocations_never_reach_the_bitmap() {
    for chunk_count in [1, 2, 3, 7, 8, 9, 13, 16, 17, 31, 64] {
        fill_and_empty_heap::<CHUNK_SIZE>(chunk_count);
    }
}

/// `CHUNK_SIZE` is a const generic that callers pick freely, and the extremes
/// change what the allocator does. At 1 the bitmap costs an eighth of the
/// region and there is no alignment guarantee beyond a single byte; at a page
/// or more every chunk is page-aligned and the bitmap is a rounding error.
#[test]
fn every_chunk_size_carries_a_full_cycle() {
    fill_and_empty_heap::<1>(64);
    fill_and_empty_heap::<2>(37);
    fill_and_empty_heap::<8>(9);
    fill_and_empty_heap::<64>(23);
    fill_and_empty_heap::<4096>(5);
    fill_and_empty_heap::<8192>(3);
}

/// A freed region becomes the starting point of the next search, so an
/// allocation of the same shape lands exactly where the old one was.
///
/// Without that the search resumes wherever the previous allocation left off
/// and walks the heap looking for a run it has just been handed back, which is
/// the expensive way to answer a question the allocator already knew.
#[test]
fn a_freed_region_is_handed_out_again_first() {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(16);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let layout = Layout::from_size_align(CHUNK_SIZE * 4, 1).unwrap();

    let first = allocator.allocate(layout).unwrap().cast::<u8>();
    let second = allocator.allocate(layout).unwrap().cast::<u8>();
    assert_ne!(first, second);

    // SAFETY: `first` is live and was allocated for `layout`.
    unsafe { allocator.deallocate(first, layout) };
    let reused = allocator.allocate(layout).unwrap().cast::<u8>();
    assert_eq!(reused, first, "the freed region must be offered again");

    // SAFETY: both pointers are live and were allocated for `layout`.
    unsafe {
        allocator.deallocate(second, layout);
        allocator.deallocate(reused, layout);
    }
    assert_eq!(allocator.usage(), 0.0);
}

/// Allocating many times the heap's size in sequence only works if every
/// deallocation gives its chunks back. A leak would run out during the loop.
#[test]
fn freed_chunks_are_reused() {
    const CHUNKS: usize = 16;
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(CHUNKS);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let layout = Layout::from_size_align(CHUNK_SIZE * 4, 1).unwrap();

    // Sixteen heaps' worth in total, four under Miri, which interprets every
    // step of the search.
    #[cfg(miri)]
    let rounds = CHUNKS;
    #[cfg(not(miri))]
    let rounds = CHUNKS * 4;

    for round in 0..rounds {
        let record = Allocation {
            ptr: allocator.allocate(layout).unwrap().cast(),
            layout,
            pattern: round as u8,
        };
        record.fill();
        record.assert_pattern(layout.size());
        // SAFETY: `record` is the only live allocation and carries its layout.
        unsafe { allocator.deallocate(record.ptr, record.layout) };
    }
    assert_eq!(allocator.usage(), 0.0);
}

/// Every chunk is handed out exactly once, and a freed chunk becomes
/// available again.
#[test]
fn allocate_respects_boundaries_and_reuses_chunks() {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(8);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let layout = Layout::from_size_align(CHUNK_SIZE, 1).unwrap();

    let mut allocations: Vec<_> = (0..8)
        .map(|_| allocator.allocate(layout).unwrap())
        .collect();
    assert_eq!(allocator.allocate(layout), Err(OutOfMemory));
    assert_eq!(allocator.usage(), 1.0);

    let ptr = allocations.pop().unwrap().cast();
    // SAFETY: `ptr` is the most recent live allocation for `layout`.
    unsafe { allocator.deallocate(ptr, layout) };
    assert!(allocator.allocate(layout).is_ok());
}

/// Alignments up to the chunk size are guaranteed; beyond it they depend on
/// the region, which this test aligns to a page on purpose.
#[test]
fn allocate_honors_requested_alignment() {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(64);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let mut allocations = Vec::new();

    for alignment in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let layout = Layout::from_size_align(1, alignment).unwrap();
        let allocation = allocator.allocate(layout).unwrap();
        assert_eq!(
            allocation.as_ptr().cast::<u8>().align_offset(alignment),
            0,
            "allocation for alignment {alignment} is misaligned"
        );
        allocations.push((allocation.cast(), layout));
    }

    for (ptr, layout) in allocations {
        // SAFETY: each pointer is live and paired with its original layout.
        unsafe { allocator.deallocate(ptr, layout) };
    }
    assert_eq!(allocator.usage(), 0.0);
}

/// Follows one allocation through grow and shrink and checks the chunk
/// bookkeeping after every step. The heap has 16 chunks, so every expected
/// usage below is `chunks_in_use / 16`.
#[test]
fn realloc_preserves_data_and_releases_chunks() {
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(16);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let old_layout = Layout::from_size_align(128, 64).unwrap();
    let record = Allocation {
        ptr: allocator.allocate(old_layout).unwrap().cast(),
        layout: old_layout,
        pattern: 0xa5,
    };
    record.fill();
    // 128 byte occupy a single chunk.
    assert_eq!(allocator.usage(), 1.0 / 16.0);

    // 600 byte do not fit into the occupied chunk, so this moves the
    // allocation to a region of three chunks and frees the old one.
    // SAFETY: `record` describes a live allocation from `allocator`.
    let grown =
        unsafe { allocator.realloc(record.ptr, old_layout, 600) }.unwrap();
    let grown_layout = Layout::from_size_align(600, 64).unwrap();
    let grown_record = Allocation {
        ptr: grown.cast(),
        layout: grown_layout,
        pattern: record.pattern,
    };
    // Only the 128 byte copied from the old allocation are known to carry the
    // pattern. The remaining bytes of the larger allocation are uninitialized,
    // so the check stops at the old size.
    grown_record.assert_pattern(old_layout.size());
    assert_eq!(allocator.usage(), 3.0 / 16.0);

    // Shrinking stays in place, but the two chunks that are no longer backed
    // by the allocation must be released here.
    // SAFETY: `grown_record` describes the live replacement allocation.
    let zero = unsafe { allocator.realloc(grown_record.ptr, grown_layout, 0) }
        .unwrap();
    assert_eq!(zero.len(), 0);
    // A zero-size allocation still owns one chunk, because the allocator
    // rounds a zero-size layout up to one byte.
    assert_eq!(allocator.usage(), 1.0 / 16.0);

    let zero_layout = Layout::from_size_align(0, 1).unwrap();
    // SAFETY: the zero-size result retains the same live allocation.
    unsafe { allocator.deallocate(zero.cast(), zero_layout) };
    assert_eq!(allocator.usage(), 0.0);
}

/// An alignment above the chunk size does not need the region to carry that
/// alignment.
///
/// Chunk `i` sits at `base + i * CHUNK_SIZE` and the allocator makes `base`
/// chunk-aligned, so every `alignment / CHUNK_SIZE`-th chunk meets a larger
/// alignment. Which ones those are shifts with the region, but they are always
/// there: a page-aligned allocation comes out of an unaligned region.
#[test]
fn over_aligned_allocations_do_not_need_an_over_aligned_region() {
    // A page-aligned buffer to skew deliberately, so that every case below
    // starts at a known distance from a page boundary.
    let mut backing = Region::new(64 * 1024, 0);
    let page = backing.as_mut_slice();

    for skew_chunks in [0, 1, 2, 7, 15] {
        let skew = skew_chunks * CHUNK_SIZE;
        let region = &mut page[skew..skew + 32 * 1024];
        assert_eq!(
            region.as_ptr().align_offset(4096) == 0,
            skew_chunks == 0,
            "only the unskewed region is page aligned"
        );

        // SAFETY: `backing` outlives the allocator and the window is used by
        // nothing else.
        let mut allocator = unsafe { allocator_over::<CHUNK_SIZE>(region) };
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        let allocation = allocator
            .allocate(layout)
            .unwrap_or_else(|_| panic!("skew of {skew_chunks} chunks"));
        assert_eq!(
            allocation.as_ptr().cast::<u8>().align_offset(4096),
            0,
            "skew of {skew_chunks} chunks"
        );
    }
}

/// A grow that cannot be satisfied has to leave the original allocation
/// exactly as it was.
///
/// This is where an allocator loses memory: releasing the old chunks before
/// the new ones are secured leaks them, and a half-updated allocation hands
/// the caller a pointer to memory it no longer owns.
#[test]
fn a_failed_grow_leaves_the_original_allocation_alone() {
    const CHUNKS: usize = 8;
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(CHUNKS);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let layout = Layout::from_size_align(CHUNK_SIZE, 1).unwrap();

    let record = Allocation {
        ptr: allocator.allocate(layout).unwrap().cast(),
        layout,
        pattern: 0x5a,
    };
    record.fill();
    // Occupy the rest, so that there is nowhere to move the allocation to.
    let rest: Vec<_> = (1..CHUNKS)
        .map(|_| allocator.allocate(layout).unwrap().cast())
        .collect();
    assert_eq!(allocator.usage(), 1.0);

    // SAFETY: `record` is a live allocation from `allocator`. `realloc` only
    // consumes the pointer when it succeeds, and this call cannot.
    let failed =
        unsafe { allocator.realloc(record.ptr, layout, CHUNK_SIZE * 2) };
    assert_eq!(failed.unwrap_err(), OutOfMemory);
    record.assert_pattern(layout.size());
    assert_eq!(
        allocator.usage(),
        1.0,
        "a failed grow must not change the bookkeeping"
    );

    // Still exactly one chunk, so freeing it releases exactly one.
    // SAFETY: `record` is still the live allocation for `layout`.
    unsafe { allocator.deallocate(record.ptr, layout) };
    assert_eq!(allocator.usage(), (CHUNKS - 1) as f32 / CHUNKS as f32);

    for ptr in rest {
        // SAFETY: each pointer is live and was allocated for `layout`.
        unsafe { allocator.deallocate(ptr, layout) };
    }
    assert_eq!(allocator.usage(), 0.0);
}

/// The `Allocator` documentation permits zero-sized requests. A chunk
/// allocator cannot hand out nothing, so it charges a whole chunk - and two
/// such allocations must still not share one.
#[test]
fn zero_sized_allocations_occupy_a_chunk_of_their_own() {
    const CHUNKS: usize = 8;
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(CHUNKS);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let layout = Layout::from_size_align(0, 1).unwrap();

    let first = allocator.allocate(layout).unwrap().cast::<u8>();
    assert_eq!(allocator.usage(), 1.0 / CHUNKS as f32);
    let second = allocator.allocate(layout).unwrap().cast::<u8>();
    assert_ne!(first, second, "two live allocations share a chunk");
    assert_eq!(allocator.usage(), 2.0 / CHUNKS as f32);

    // SAFETY: both are live allocations made for `layout`.
    unsafe {
        allocator.deallocate(first, layout);
        allocator.deallocate(second, layout);
    }
    assert_eq!(allocator.usage(), 0.0);
}

/// Drives allocate, deallocate and realloc in an order that the hand-written
/// tests do not reach: they check one operation at a time on an otherwise
/// fresh heap, while the bugs of a bitmap allocator show up after the heap has
/// become fragmented and the free-chunk hint points somewhere in the middle.
///
/// Two invariants carry the test. Every allocation is filled with its own
/// pattern and re-checked before it is touched again, so any overlap between
/// two live allocations fails here instead of corrupting data silently. And
/// after everything is freed, `usage` must be back at zero, so a chunk that is
/// never released fails the test as well.
///
/// The sequence comes from a fixed seed rather than a random one: a failure is
/// reproducible and bisectable, which a randomised run would not be. The step
/// count is reduced under Miri, which needs roughly three orders of magnitude
/// more time per step.
#[test]
fn deterministic_allocation_lifecycle() {
    #[cfg(miri)]
    const STEPS: usize = 64;
    #[cfg(not(miri))]
    const STEPS: usize = 512;
    let mut backing = Region::for_chunks::<CHUNK_SIZE>(64);
    // SAFETY: `backing` outlives the allocator and is not used meanwhile.
    let mut allocator =
        unsafe { allocator_over::<CHUNK_SIZE>(backing.as_mut_slice()) };
    let mut seed = 0x5eed_u64;
    let mut live: Vec<Allocation> = Vec::new();

    for step in 0..STEPS {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let index = (seed as usize) % live.len().max(1);
        match (seed >> 32) % 3 {
            0 if !live.is_empty() => {
                let allocation = live.swap_remove(index);
                allocation.assert_pattern(allocation.layout.size());
                // SAFETY: the record retains the original live allocation and
                // layout.
                unsafe {
                    allocator.deallocate(allocation.ptr, allocation.layout)
                };
            }
            1 if !live.is_empty() => {
                let mut allocation = live.swap_remove(index);
                allocation.assert_pattern(allocation.layout.size());
                let new_size = ((seed >> 8) as usize % 700) + 1;
                // SAFETY: the record retains the original live allocation and
                // layout.
                match unsafe {
                    allocator.realloc(
                        allocation.ptr,
                        allocation.layout,
                        new_size,
                    )
                } {
                    Ok(ptr) => {
                        let preserved = allocation.layout.size().min(new_size);
                        allocation.ptr = ptr.cast();
                        allocation.layout = Layout::from_size_align(
                            new_size,
                            allocation.layout.align(),
                        )
                        .unwrap();
                        allocation.assert_pattern(preserved);
                        allocation.fill();
                        live.push(allocation);
                    }
                    Err(OutOfMemory) => live.push(allocation),
                }
            }
            _ => {
                let size = ((seed >> 8) as usize % 700) + 1;
                let alignment =
                    [1, 2, 4, 8, 16, 32, 64, 128, 256][(seed as usize) % 9];
                let layout = Layout::from_size_align(size, alignment).unwrap();
                if let Ok(ptr) = allocator.allocate(layout) {
                    let allocation = Allocation {
                        ptr: ptr.cast(),
                        layout,
                        pattern: step as u8,
                    };
                    allocation.fill();
                    live.push(allocation);
                }
            }
        }
    }

    for allocation in live {
        allocation.assert_pattern(allocation.layout.size());
        // SAFETY: every remaining record is a live allocation with its layout.
        unsafe { allocator.deallocate(allocation.ptr, allocation.layout) };
    }
    assert_eq!(allocator.usage(), 0.0);
}
