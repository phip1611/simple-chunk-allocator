# Simple Chunk Allocator

`simple-chunk-allocator` is a nightly-only `no_std` allocator for a fixed,
caller-provided memory region. It manages allocations in fixed-size chunks and
is suitable for small static heaps in kernels, bootloaders, and freestanding
binaries.

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

The `heap!` and `heap_bitmap!` macros create page-aligned backing storage. Use
`GlobalChunkAllocator` with `new_raw` for static storage:

```rust
use simple_chunk_allocator::{
    heap, heap_bitmap, GlobalChunkAllocator, PageAligned,
};

static mut HEAP: PageAligned<[u8; 4096]> = heap!(chunks = 16, chunksize = 256);
static mut BITMAP: PageAligned<[u8; 2]> = heap_bitmap!(chunks = 16);

#[global_allocator]
static ALLOCATOR: GlobalChunkAllocator = unsafe {
    // SAFETY: ALLOCATOR exclusively owns both static regions for its lifetime.
    GlobalChunkAllocator::new_raw(
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(HEAP).cast(),
            4096,
        ),
        core::ptr::slice_from_raw_parts_mut(
            core::ptr::addr_of_mut!(BITMAP).cast(),
            2,
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

`ChunkAllocator` can be used directly with `allocate`, `deallocate`, and
`realloc`. `GlobalChunkAllocator::allocator_api_glue` exposes a value that
implements the nightly `Allocator` trait for types such as `Vec<T, _>`.

Only deallocate or reallocate pointers returned by the same allocator, using
the original layout. These operations are unsafe because a mismatched pointer
or layout can corrupt the allocator.

## Examples

The `examples/minimal.rs` program demonstrates a hosted global allocator. The
`freestanding-linux-example` directory contains an x86_64 Linux binary without
libc. It is illustrative only; normal Linux programs should use the operating
system allocator.
