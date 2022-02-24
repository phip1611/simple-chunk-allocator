#![feature(allocator_api)]
#![feature(slice_ptr_get)]

use rand::Rng;
use simple_chunk_allocator::{ChunkAllocator, DEFAULT_CHUNK_SIZE};
use std::alloc::{AllocError, Allocator, Layout};
use std::ptr::NonNull;

/// Benchmark that helps me to check how the search time for new blocks
/// develops when the heap is getting full.
///
/// Execute with: `cargo run --release --example bench`
fn main() {
    // 16 MB
    const HEAP_SIZE: usize = 0x1000000;
    const CHUNK_COUNT: usize = HEAP_SIZE / DEFAULT_CHUNK_SIZE;
    const BITMAP_SIZE: usize = CHUNK_COUNT / 8;
    let mut heap = Vec::with_capacity_in(HEAP_SIZE, PageAlignedGlobalAlloc);
    (0..heap.capacity()).for_each(|_| heap.push(0));
    let mut heap_bitmap = Vec::with_capacity_in(BITMAP_SIZE, PageAlignedGlobalAlloc);
    (0..heap_bitmap.capacity()).for_each(|_| heap_bitmap.push(0));
    let mut alloc: ChunkAllocator =
        ChunkAllocator::new(heap.as_mut_slice(), heap_bitmap.as_mut_slice()).unwrap();

    let now_fn = || unsafe { x86::time::rdtscp() };

    let mut all_allocations = Vec::new();
    let mut all_alloc_measurements = Vec::new();

    let powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128];
    let mut rng = rand::thread_rng();
    while alloc.usage() < 99.9 {
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
        let count_all_allocations_not_freed_yet = all_allocations.iter().filter(|x| x.is_some()).count();
        let count_allocations_to_free = if count_all_allocations_not_freed_yet > 10 && rng.gen_range(0..10) == 0 {
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
                alloc.deallocate(allocation.as_non_null_ptr(), layout);
            });

        println!(
            "usage={:5.2}%, ticks={:6}, success={:5}, layout={:?}",
            alloc.usage(),
            ticks,
            alloc_res.is_ok(),
            layout
        );
    }

    all_alloc_measurements.sort_by(|x1, x2| x1.cmp(x2));
    println!(
        "Stats: {:6} allocations, median={} ticks, average={} ticks, min={} ticks, max={} ticks",
        all_alloc_measurements.len(),
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
