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
#![feature(allocator_api)]
#![feature(slice_ptr_get)]

use rand::Rng;
use simple_chunk_allocator::{GlobalChunkAllocator, DEFAULT_CHUNK_SIZE};
use std::alloc::{Allocator, Layout};
use std::time::Instant;

/// This is already enough to fill the corresponding heaps.
const BENCH_DURATION: f64 = 10.0;

/// 160 MiB heap size.
const HEAP_SIZE: usize = 0xa000000;
/// Backing memory for heap management.
static mut HEAP_MEMORY: PageAlignedBytes<HEAP_SIZE> = PageAlignedBytes([0; HEAP_SIZE]);

/// ChunkAllocator specific stuff.
const CHUNK_COUNT: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE;
const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
static mut HEAP_BITMAP_MEMORY: PageAlignedBytes<BITMAP_SIZE> = PageAlignedBytes([0; BITMAP_SIZE]);

/// Benchmark that helps me to check how the search time for new chunks
/// gets influenced when the heap is getting full. The benchmark fills the heap
/// until it is 100% full. During that process, it randomly allocates new memory
/// with different alignments. Furthermore, it makes random deallocations of already
/// allocated space to provoke fragmentation.
///
/// Execute with `cargo run --release --example bench`. Or to get even better performance,
/// execute it with `RUSTFLAGS="-C target-cpu=native" cargo run --example bench --release`
///
fn main() {
    let chunk_allocator = unsafe {
        GlobalChunkAllocator::<DEFAULT_CHUNK_SIZE>::new(
            HEAP_MEMORY.0.as_mut_slice(),
            HEAP_BITMAP_MEMORY.0.as_mut_slice(),
        )
    };

    let mut linked_list_allocator = unsafe {
        linked_list_allocator::LockedHeap::new(HEAP_MEMORY.0.as_mut_ptr() as _, HEAP_SIZE)
    };

    let bench_res_1 = benchmark_allocator(&mut chunk_allocator.allocator_api_glue());
    let bench_res_2 = benchmark_allocator(&mut linked_list_allocator);

    print_bench_results("Chunk Allocator", &bench_res_1);
    println!();
    print_bench_results("Linked List Allocator", &bench_res_2);
}

fn benchmark_allocator(alloc: &mut dyn Allocator) -> BenchRunResults {
    let now_fn = || unsafe { x86::time::rdtscp().0 };

    let mut all_allocations = Vec::new();
    let mut all_deallocations = Vec::new();
    let mut all_alloc_measurements = Vec::new();

    let powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128];
    let mut rng = rand::thread_rng();

    // run for 10s
    let bench_begin_time = Instant::now();
    while bench_begin_time.elapsed().as_secs_f64() <= BENCH_DURATION {
        let alignment_i = rng.gen_range(0..powers_of_two.len());
        let size = rng.gen_range(64..16384);
        let layout = Layout::from_size_align(size, powers_of_two[alignment_i]).unwrap();
        let alloc_begin = now_fn();
        let alloc_res = alloc.allocate(layout);
        let alloc_ticks = now_fn() - alloc_begin;
        all_alloc_measurements.push(alloc_ticks);
        all_allocations.push(Some((layout, alloc_res)));

        // now free an arbitrary amount again to simulate intense heap usage
        // Every ~10th iteration I free 7 existing allocations; the heap will slowly grow until it is full
        let count_all_allocations_not_freed_yet =
            all_allocations.iter().filter(|x| x.is_some()).count();
        let count_allocations_to_free =
            if count_all_allocations_not_freed_yet > 10 && rng.gen_range(0..10) == 0 {
                7
            } else {
                0
            };

        all_allocations
            .iter_mut()
            .filter(|x| x.is_some())
            // .take() important; so that we don't allocate the same allocation multiple times ;)
            .map(|x| x.take().unwrap())
            .filter(|(_, res)| res.is_ok())
            .map(|(layout, res)| (layout, res.unwrap()))
            .take(count_allocations_to_free)
            .for_each(|(layout, allocation)| unsafe {
                // println!("dealloc: layout={:?}", layout);
                all_deallocations.push((layout, allocation));
                alloc.deallocate(allocation.as_non_null_ptr(), layout);
            });
    }

    // sort
    all_alloc_measurements.sort_by(|x1, x2| x1.cmp(x2));

    BenchRunResults {
        allocation_attempts: all_allocations.len() as _,
        successful_allocations: all_allocations
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.as_ref().unwrap())
            .map(|(_layout, res)| res.is_ok())
            .count() as _,
        deallocations: all_deallocations.len() as _,
        allocation_measurements: all_alloc_measurements,
    }
}

fn print_bench_results(bench_name: &str, res: &BenchRunResults) {
    println!("RESULTS OF BENCHMARK: {bench_name}");
    println!(
        "    {:6} allocations, {:6} successful_allocations, {:6} deallocations",
        res.allocation_attempts, res.successful_allocations, res.deallocations
    );
    println!(
        "    median={:6} ticks, average={:6} ticks, min={:6} ticks, max={:6} ticks",
        res.allocation_measurements[res.allocation_measurements.len() / 2],
        res.allocation_measurements.iter().sum::<u64>()
            / (res.allocation_measurements.len() as u64),
        res.allocation_measurements.iter().min().unwrap(),
        res.allocation_measurements.iter().max().unwrap(),
    );
}

/// Result of a bench run.
struct BenchRunResults {
    /// Number of attempts of allocations.
    allocation_attempts: u64,
    /// Number of successful successful_allocations.
    successful_allocations: u64,
    /// Number of deallocations.
    deallocations: u64,
    /// Sorted vector of the amount of clock ticks per allocation.
    allocation_measurements: Vec<u64>,
}

#[repr(align(4096))]
struct PageAlignedBytes<const N: usize>([u8; N]);
