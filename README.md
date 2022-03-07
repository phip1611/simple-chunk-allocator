# Simple Chunk Allocator

A simple `no_std` allocator written in Rust that manages memory in fixed-size chunks/blocks. Useful for basic `no_std`
binaries where you want to manage a heap of a few megabytes without complex features such as paging/page table
management. Instead, this allocator gets a fixed/static memory region and allocates memory from there. This memory
region can be contained inside the executable file that uses this allocator. See examples down below.

⚠ _There probably exist better solutions for large-scale applications that have better performance by using a
more complex algorithm. However, this is good for simple `no_std` binaries and hopefully also for educational
purposes. It helped me to understand a lot about allocators._ ⚠

## TL;DR
- ✅ `no_std` allocator with test coverage
- ✅ uses static memory as backing storage (no paging/page table manipulations)
- ✅ allocation strategy is a combination of next-fit and best-fit
- ✅ reasonable fast with low code complexity
- ✅ const compatibility (no runtime `init()` required)
- ✅ efficient in scenarios where heap is a few dozens megabytes in size
- ✅ user-friendly API

The inner and low-level `ChunkAllocator` can be used as `#[global_allocator]` with the synchronized wrapper type
`GlobalChunkAllocator`. Both can be used with the `allocator_api` feature. The latter enables the usage in several
types of the Rust standard library, such as `Vec::new_in` or `BTreeMap::new_in`. This is primarily interesting for
testing but may also enable other interesting use-cases.

The focus is on `const` compatibility. The allocator and the backing memory can get initialized during compile time
and need no runtime `init()` call or similar. This means that if the compiler accepts it then the allocation will
also work during runtime. However, you can also create allocator objects during runtime.

The inner and low-level `ChunkAllocator` is a chunk allocator or also called fixed-size block allocator. It uses a
mixture of the strategies next-fit and a best-fit. It tries to use the smallest gap for an allocation request to
prevent fragmentation but this is no guarantee. Each allocation is a trade-off between a low allocation time and
preventing fragmentation. The default chunk size is `256 bytes` but this can be changed as compile time const generic.
Having a fixed-size block allocator enables an easy bookkeeping algorithm through a bitmap but has as consequence that
small allocations, such as `64 byte` will take at least one chunk/block of the chosen block size.

This project originates from my [Diplom thesis project](https://github.com/phip1611/diplomarbeit-impl). Since I
originally had lots of struggles to create this (my first ever allocator), I outsourced it for better testability and
to share my knowledge and findings with others in the hope that someone can learn from it in any way.


# Minimal Code Example

```rust
#![feature(const_mut_refs)]
#![feature(allocator_api)]
#![feature(const_ptr_is_null)]

use simple_chunk_allocator::{heap, heap_bitmap, GlobalChunkAllocator, PageAligned};

// The macros help to get a correctly sized arrays types.
// I page-align them for better caching and to improve the availability of
// page-aligned addresses.

/// Backing storage for heap (1Mib). (read+write) static memory in final executable.
static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
/// Backing storage for heap bookkeeping bitmap. (read+write) static memory in final executable.
static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();

#[global_allocator]
static ALLOCATOR: GlobalChunkAllocator =
    unsafe { GlobalChunkAllocator::new(HEAP.deref_mut_const(), HEAP_BITMAP.deref_mut_const()) };

fn main() {
    // at this point, the allocator already got used a bit by the Rust runtime that executes
    // before main() gets called. This is not the case if a `no_std` binary gets produced.
    let old_usage = ALLOCATOR.usage();
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    assert!(ALLOCATOR.usage() > old_usage);

    // use "allocator_api"-feature. You can use this if "ALLOCATOR" is not registered as
    // the global allocator. Otherwise, it is already the default.
    let _boxed = Box::new_in([1, 2, 3], ALLOCATOR.allocator_api_glue());
}
```

## MSRV
This crate only builds with the nightly version. I developed it with version `1.61.0-nightly` (2022-03-05).

## Performance
I executed my example `bench` in release mode on an Intel i7-1165G7 CPU and a heap of `160MB` to get the results listed
below. I used `RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench` to get maximum performance.
The benchmark simulates a heavy usage of the heap in a single-threaded program with many random allocations and
deallocations. The allocations very in their alignment. The table below shows the results of this benchmark as number
of clock cycles. Increasing the chunk size reduces the size of the bookkeeping bitmap which accelerates lookup.
However, a smaller chunk size occupies less heap when only very small allocations are required.

| Chunk Size    | # allocations | median | average | min | max   |
|---------------|---------------|--------|---------|-----|-------|
| 128           | 66960         | 858    | 884     | 129 | 46016 |
| 256 [DEFAULT] | 68371         | 503    | 517     | 110 | 38047 |
| 512           | 63154         | 355    | 366     | 102 | 40107 |

The results vary slightly because each run gets influenced by some randomness.
