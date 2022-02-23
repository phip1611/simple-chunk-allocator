#![feature(allocator_api)]

use std::alloc::{Allocator, AllocError, Layout};
use std::ptr::NonNull;
use rand::Rng;
use simple_chunk_allocator::{ChunkAllocator, DEFAULT_CHUNK_SIZE};

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
    let mut alloc: ChunkAllocator = ChunkAllocator::new(heap.as_mut_slice(), heap_bitmap.as_mut_slice()).unwrap();


    let now_fn = || unsafe { x86::time::rdtscp() };

    let powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128];
    let mut rng = rand::thread_rng();
    while alloc.usage() < 99.0 {
        let alignment_i = rng.gen_range(0..powers_of_two.len());
        let size = rng.gen_range(0..256);
        let layout = Layout::from_size_align(size, powers_of_two[alignment_i]).unwrap();
        let begin = now_fn();
        let alloc_res = alloc.allocate(layout);
        let ticks = now_fn() - begin;
        println!("usage={:3}%, ticks={:6}, success={:5}, layout={:?}", alloc.usage(), ticks, alloc_res.is_ok(), layout);
    }
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
