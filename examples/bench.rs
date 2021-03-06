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
use simple_chunk_allocator::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use std::alloc::{AllocError, Allocator, Layout};
use std::ptr::NonNull;

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
    // 160 MiB
    const HEAP_SIZE: usize = 0xa000000;
    const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;
    const CHUNK_COUNT: usize = HEAP_SIZE / CHUNK_SIZE;
    const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
    let mut heap = Vec::with_capacity_in(HEAP_SIZE, PageAlignedGlobalAlloc);
    (0..heap.capacity()).for_each(|_| heap.push(0));
    let mut heap_bitmap = Vec::with_capacity_in(BITMAP_SIZE, PageAlignedGlobalAlloc);
    (0..heap_bitmap.capacity()).for_each(|_| heap_bitmap.push(0));
    let mut alloc =
        ChunkAllocator::<CHUNK_SIZE>::new(heap.as_mut_slice(), heap_bitmap.as_mut_slice()).unwrap();

    let now_fn = || unsafe { x86::time::rdtscp() };

    let mut all_allocations = Vec::new();
    let mut all_deallocations = Vec::new();
    let mut all_alloc_measurements = Vec::new();

    let powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128];
    let mut rng = rand::thread_rng();
    while alloc.usage() < 100.0 {
        let alignment_i = rng.gen_range(0..powers_of_two.len());
        let size = rng.gen_range(64..16384);
        let layout = Layout::from_size_align(size, powers_of_two[alignment_i]).unwrap();
        let begin = now_fn();
        let alloc_res = alloc.allocate(layout);
        let ticks = now_fn() - begin;
        all_alloc_measurements.push(ticks);
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

        println!(
            "usage={:6.2}%, ticks={:8}, success={:5}, layout={:?}",
            alloc.usage(),
            ticks,
            alloc_res.is_ok(),
            layout
        );
    }

    all_alloc_measurements.sort_by(|x1, x2| x1.cmp(x2));
    println!(
        "Stats: {:6} allocations, {:6} deallocations, chunk_size={}, #chunks={}",
        all_alloc_measurements.len(),
        all_deallocations.len(),
        alloc.chunk_size(),
        alloc.chunk_count()
    );
    println!(
        "        median={} ticks, average={} ticks, min={} ticks, max={} ticks",
        all_alloc_measurements[all_alloc_measurements.len() / 2],
        all_alloc_measurements.iter().sum::<u64>() / (all_alloc_measurements.len() as u64),
        all_alloc_measurements.iter().min().unwrap(),
        all_alloc_measurements.iter().max().unwrap(),
    );
}

struct PageAlignedGlobalAlloc;

unsafe impl Allocator for PageAlignedGlobalAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        std::alloc::System.allocate(layout.align_to(4096).unwrap())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::System.deallocate(ptr, layout)
    }
}
