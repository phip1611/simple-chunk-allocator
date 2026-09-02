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
//! Fast-path round trips of this allocator, plus deterministic workload
//! replays against `linked_list_allocator` and `talc`.
//!
//! Fairness rules of the comparison: every contender runs through its raw,
//! unsynchronized `&mut` API, gets an identically sized region, pays its own
//! metadata out of it, and replays the identical fixed-seed operation
//! sequence. The chunk allocator rounds requests up to whole chunks; that is
//! a property under test, not a harness artifact.

#![deny(clippy::undocumented_unsafe_blocks)]

use core::alloc::Layout;
use core::ptr::NonNull;
use core::time::Duration;
use criterion::measurement::WallTime;
use criterion::{
    Bencher, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
    criterion_group, criterion_main,
};
use simple_chunk_allocator::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use std::hint::black_box;
use talc::source::Claim;

/// Chunks in the benchmark heap: 8 MiB.
const HEAP_CHUNKS: usize = 32768;

type TalcAllocator = talc::base::Talc<Claim, talc::DefaultBinning>;

/// Aligned backing memory, non-zero filled to prove the allocator needs no
/// pre-zeroed region. Trimmed copy of `tests/common` (benches cannot import
/// test modules).
struct Region {
    buffer: Vec<u8>,
    offset: usize,
}

impl Region {
    /// Holds exactly [`HEAP_CHUNKS`] chunks plus their bitmap. Aligned to
    /// 4096 so no chunk is lost to leading padding.
    fn new() -> Self {
        let len = HEAP_CHUNKS * DEFAULT_CHUNK_SIZE + HEAP_CHUNKS.div_ceil(8);
        let buffer = vec![0xAB; len + 4096];
        let offset = buffer.as_ptr().align_offset(4096);
        Self { buffer, offset }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.buffer.len() - 4096;
        &mut self.buffer[self.offset..self.offset + len]
    }
}

/// Advances the seed and returns it. Fixed-seed sequences keep every run
/// reproducible and comparable across commits.
const fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

/// Uniform driver over the raw, unsynchronized contenders.
trait RawAllocator {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>>;

    /// # Safety
    /// `ptr` must be a live allocation from `self` for `layout`, freed once.
    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout);
}

impl<const CHUNK_SIZE: usize> RawAllocator for ChunkAllocator<CHUNK_SIZE> {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        ChunkAllocator::allocate(self, layout)
            .ok()
            .map(NonNull::cast)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded caller contract.
        unsafe { ChunkAllocator::deallocate(self, ptr, layout) }
    }
}

impl RawAllocator for linked_list_allocator::Heap {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        self.allocate_first_fit(layout).ok()
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded caller contract.
        unsafe { linked_list_allocator::Heap::deallocate(self, ptr, layout) }
    }
}

impl RawAllocator for TalcAllocator {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: the workloads never produce zero-sized layouts.
        unsafe { TalcAllocator::allocate(self, layout) }
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded caller contract.
        unsafe { TalcAllocator::deallocate(self, ptr.as_ptr(), layout) }
    }
}

/// One step of a generated workload.
#[derive(Clone, Copy)]
enum Op {
    Alloc { slot: usize, layout: Layout },
    Free { slot: usize },
}

struct Workload {
    ops: Vec<Op>,
    slot_count: usize,
}

/// Deterministic alloc/free sequence oscillating around `target_percent`
/// heap occupancy, ending on an empty heap so a replay leaves the allocator
/// as it started. Occupancy is tracked with a conservative per-allocation
/// footprint, so no contender ever runs out of memory.
fn generate(
    seed: u64,
    churn_ops: usize,
    usable_bytes: usize,
    target_percent: usize,
    sizes: &[(usize, u32)],
    aligns: &[usize],
) -> Workload {
    // Whole chunks here, header and alignment slack on the free-list
    // designs.
    let footprint =
        |size: usize| size.next_multiple_of(DEFAULT_CHUNK_SIZE).max(size + 64);
    let target_bytes = usable_bytes / 100 * target_percent;
    let mut seed = seed;
    let mut ops = Vec::new();
    let mut live: Vec<(usize, usize)> = Vec::new();
    let mut live_bytes = 0_usize;
    let mut next_slot = 0_usize;

    for _ in 0..churn_ops {
        let roll = lcg(&mut seed) % 4;
        // Allocate with p=3/4 below the target, free with p=3/4 above it.
        let allocates = if live.is_empty() {
            true
        } else if live_bytes < target_bytes {
            roll != 0
        } else {
            roll == 0
        };

        if allocates {
            let size = draw_weighted(&mut seed, sizes);
            let align = aligns[(lcg(&mut seed) as usize) % aligns.len()];
            let layout = Layout::from_size_align(size, align).unwrap();
            ops.push(Op::Alloc {
                slot: next_slot,
                layout,
            });
            live.push((next_slot, footprint(size)));
            live_bytes += footprint(size);
            next_slot += 1;
        } else {
            // `swap_remove` guarantees each slot is freed exactly once.
            let index = (lcg(&mut seed) as usize) % live.len();
            let (slot, bytes) = live.swap_remove(index);
            live_bytes -= bytes;
            ops.push(Op::Free { slot });
        }
    }

    while !live.is_empty() {
        let index = (lcg(&mut seed) as usize) % live.len();
        let (slot, _) = live.swap_remove(index);
        ops.push(Op::Free { slot });
    }

    Workload {
        ops,
        slot_count: next_slot,
    }
}

fn draw_weighted(seed: &mut u64, sizes: &[(usize, u32)]) -> usize {
    let total: u32 = sizes.iter().map(|(_, weight)| *weight).sum();
    let mut roll = (lcg(seed) % u64::from(total)) as u32;
    for (size, weight) in sizes {
        if roll < *weight {
            return *size;
        }
        roll -= *weight;
    }
    unreachable!("the roll is below the summed weights")
}

/// Mixed sizes weighted toward small allocations, ~70 % occupancy.
fn churn_workload() -> Workload {
    generate(
        0x5eed,
        4096,
        HEAP_CHUNKS * DEFAULT_CHUNK_SIZE,
        70,
        &[(64, 40), (256, 25), (1024, 20), (4096, 10), (16384, 5)],
        &[1, 8, 16, 64],
    )
}

/// Bimodal sizes over a long churn - the hole-dominated pattern where the
/// designs diverge the most. Sits right below the occupancy where the chunk
/// allocator starts failing requests for contiguous runs.
fn fragmentation_workload() -> Workload {
    generate(
        0xf7a6,
        8192,
        HEAP_CHUNKS * DEFAULT_CHUNK_SIZE,
        70,
        &[(64, 60), (2048, 40)],
        &[1, 8, 64],
    )
}

/// Replays `ops`, returning the number of failed allocations. `black_box`
/// on both pointer directions keeps the alloc/free pairs alive.
fn replay<A: RawAllocator>(
    alloc: &mut A,
    ops: &[Op],
    slots: &mut [Option<(NonNull<u8>, Layout)>],
) -> usize {
    let mut failed = 0;
    for op in ops {
        match *op {
            Op::Alloc { slot, layout } => {
                match alloc.allocate(black_box(layout)) {
                    Some(ptr) => slots[slot] = Some((black_box(ptr), layout)),
                    None => failed += 1,
                }
            }
            Op::Free { slot } => {
                if let Some((ptr, layout)) = slots[slot].take() {
                    // SAFETY: `ptr` came from `alloc` for `layout` and is
                    // freed only here.
                    unsafe { alloc.deallocate(black_box(ptr), layout) };
                }
            }
        }
    }
    failed
}

/// Allocation plus immediate deallocation: the happy path, with an
/// unchanged heap across iterations.
fn ping_pong<A: RawAllocator>(
    b: &mut Bencher<'_, WallTime>,
    alloc: &mut A,
    layout: Layout,
) {
    b.iter(|| {
        let ptr = alloc.allocate(black_box(layout)).unwrap();
        // SAFETY: `ptr` was just returned for `layout`.
        unsafe { alloc.deallocate(black_box(ptr), layout) };
    });
}

enum Scenario<'a> {
    /// 64 B ping-pong on an empty heap.
    RoundTrip,
    Replay(&'a Workload),
}

/// Registers `scenario` for one contender. Generic, so every timed loop is
/// monomorphized - no dynamic dispatch is measured.
fn run_scenario<A: RawAllocator>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    alloc: &mut A,
    scenario: &Scenario<'_>,
) {
    match *scenario {
        Scenario::RoundTrip => {
            let layout = Layout::from_size_align(64, 8).unwrap();
            group.bench_function(BenchmarkId::from_parameter(id), |b| {
                ping_pong(b, alloc, layout);
            });
        }
        Scenario::Replay(workload) => {
            let mut slots = vec![None; workload.slot_count];
            // An untimed replay proves the workload fits this heap.
            assert_eq!(replay(alloc, &workload.ops, &mut slots), 0);
            group.bench_function(BenchmarkId::from_parameter(id), |b| {
                b.iter(|| replay(alloc, &workload.ops, &mut slots));
            });
        }
    }
}

/// Runs `scenario` once per contender, each over its own fresh region.
fn for_each_contender(
    group: &mut BenchmarkGroup<'_, WallTime>,
    scenario: &Scenario<'_>,
) {
    {
        let mut region = Region::new();
        let slice = region.as_mut_slice();
        // SAFETY: `region` outlives the allocator and has no other user.
        let mut alloc: ChunkAllocator<DEFAULT_CHUNK_SIZE> =
            unsafe { ChunkAllocator::new(slice.as_mut_ptr(), slice.len()) };
        run_scenario(group, "simple-chunk-allocator", &mut alloc, scenario);
    }
    {
        let mut region = Region::new();
        let slice = region.as_mut_slice();
        // SAFETY: `region` outlives the allocator and has no other user.
        let mut heap = unsafe {
            linked_list_allocator::Heap::new(slice.as_mut_ptr(), slice.len())
        };
        run_scenario(group, "linked-list-allocator", &mut heap, scenario);
    }
    {
        let mut region = Region::new();
        let slice = region.as_mut_slice();
        // SAFETY: `region` outlives the allocator and has no other user.
        let claim = unsafe { Claim::new(slice.as_mut_ptr(), slice.len()) };
        let mut talc = TalcAllocator::new(claim);
        run_scenario(group, "talc", &mut talc, scenario);
    }
}

/// This crate's hint fast path on an empty heap, per layout shape.
fn round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip");
    let cases = [
        ("single_chunk", Layout::from_size_align(64, 8).unwrap()),
        ("multi_chunk", Layout::from_size_align(4096, 8).unwrap()),
        ("over_aligned", Layout::from_size_align(256, 4096).unwrap()),
    ];
    for (name, layout) in cases {
        let mut region = Region::new();
        let slice = region.as_mut_slice();
        // SAFETY: `region` outlives the allocator and has no other user.
        let mut alloc: ChunkAllocator<DEFAULT_CHUNK_SIZE> =
            unsafe { ChunkAllocator::new(slice.as_mut_ptr(), slice.len()) };
        group.bench_function(name, |b| ping_pong(b, &mut alloc, layout));
    }
    group.finish();
}

fn compare_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_round_trip");
    for_each_contender(&mut group, &Scenario::RoundTrip);
    group.finish();
}

fn compare_replay(c: &mut Criterion, name: &str, workload: &Workload) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(workload.ops.len() as u64));
    group.sample_size(60);
    group.measurement_time(Duration::from_secs(10));
    for_each_contender(&mut group, &Scenario::Replay(workload));
    group.finish();
}

fn compare_mixed_churn(c: &mut Criterion) {
    compare_replay(c, "compare_mixed_churn", &churn_workload());
}

fn compare_fragmentation_churn(c: &mut Criterion) {
    compare_replay(c, "compare_fragmentation_churn", &fragmentation_workload());
}

criterion_group!(
    benches,
    round_trip,
    compare_round_trip,
    compare_mixed_churn,
    compare_fragmentation_churn
);
criterion_main!(benches);
