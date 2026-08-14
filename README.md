# Simple Chunk Allocator

A nightly-only `no_std` allocator that manages fixed-size chunks in
caller-provided static memory.

It is suitable for small static heaps in kernels, bootloaders, and freestanding
binaries.

## Highlights

- ✅ `no_std` allocator with test coverage
- ✅ uses static memory as backing storage (no paging/page table manipulations)
- ✅ allocation strategy is a combination of next-fit and best-fit
- ✅ reasonably fast with low code complexity
- ✅ const compatibility (no runtime `init()` required)
- ✅ efficient in scenarios where the heap is a few dozen megabytes in size
- ✅ user-friendly API

## When to use it

Use this crate when the heap size is known in advance and predictable memory
use matters more than fine-grained allocation efficiency. Every allocation uses
whole chunks, so a small allocation can consume up to one complete chunk.

The default chunk size is 256 bytes. Choose a larger chunk size for faster
searches and a smaller bitmap; choose a smaller chunk size to waste less space
on small allocations.

## Requirements

- Rust nightly, because the crate uses the unstable allocator API.
- A non-empty heap whose length is a multiple of the chunk size.
- A chunk count divisible by eight.
- A bitmap with exactly one bit per heap chunk.
- Heap storage aligned to the chunk size. Page alignment is recommended.

## Global allocator

The `heap!` and `heap_bitmap!` macros create page-aligned backing storage. Both
derive their size from the chunk geometry, so a single pair of constants
describes the arrays and the slices passed to `new_raw`:

```rust
use simple_chunk_allocator::{
    GlobalChunkAllocator, PageAligned, heap, heap_bitmap,
};

const CHUNKS: usize = 4096;
const CHUNK_SIZE: usize = 256;

static mut HEAP: PageAligned<[u8; CHUNKS * CHUNK_SIZE]> =
    heap!(chunks = CHUNKS, chunksize = CHUNK_SIZE);
static mut BITMAP: PageAligned<[u8; CHUNKS / 8]> =
    heap_bitmap!(chunks = CHUNKS);

#[global_allocator]
// SAFETY: ALLOCATOR exclusively owns both statics for the whole program.
static ALLOCATOR: GlobalChunkAllocator<CHUNK_SIZE> = unsafe {
    GlobalChunkAllocator::new_raw(
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(HEAP).cast(),
            CHUNKS * CHUNK_SIZE,
        ),
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(BITMAP).cast(),
            CHUNKS / 8,
        ),
    )
};

fn main() {
    let mut values = Vec::new();
    values.push(42);
}
```

`new_raw` is unsafe because the allocator cannot verify the storage lifetime,
exclusivity, or overlap. See its API documentation for the complete contract.

## Direct use and allocator API

`GlobalChunkAllocator::allocator_api_glue` exposes a value that implements the
nightly `Allocator` trait. Use it when the allocator is not registered globally
and only selected collections should allocate from it:

```rust
let mut vec = Vec::<u8, _>::with_capacity_in(
    123,
    ALLOCATOR.allocator_api_glue(),
);
vec.push(42);
```

The inner `ChunkAllocator` can also be driven directly with `allocate`,
`deallocate`, and `realloc`. It is not synchronized, so it needs `&mut self`.
`new` validates the storage instead of deferring the alignment check to the
first allocation:

```rust
// The heap must be at least CHUNK_SIZE-aligned; `PageAligned` ensures that.
let mut heap = PageAligned::new([0_u8; 16 * CHUNK_SIZE]);
let mut bitmap = PageAligned::new([0_u8; 16 / 8]);
let mut allocator = ChunkAllocator::<CHUNK_SIZE>::new(
    heap.as_mut_slice(),
    bitmap.as_mut_slice(),
)
.unwrap();

let layout = Layout::from_size_align(64, 8).unwrap();
let allocation = allocator.allocate(layout).unwrap();
// SAFETY: `allocation` is live and paired with its original layout.
unsafe { allocator.deallocate(allocation.cast(), layout) };
```

Only deallocate or reallocate pointers returned by the same allocator, using
the original layout. These operations are unsafe because a mismatched pointer
or layout can corrupt the allocator.

## Performance

TODO: The previous benchmark numbers were measured against an older version of
`examples/bench.rs` and are therefore removed. New numbers follow once the
benchmark has been reworked.

## Examples

The `examples/minimal.rs` program demonstrates a hosted global allocator. The
`freestanding-linux-example` directory contains an x86_64 Linux binary without
libc. It is illustrative only; normal Linux programs should use the operating
system allocator.
