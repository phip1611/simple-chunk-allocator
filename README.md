# Simple Chunk Allocator

A simple `no_std` allocator written in Rust that manages memory in fixed-size chunks/blocks. Useful for basic `no_std`
binaries where you want to manage a heap of a few megabytes without complex features such as paging/page table
management. Instead, this allocator gets a fixed/static memory region and allocates memory from there. This memory
region can be contained inside the executable file that uses this allocator. See examples down below.

⚠ _Other allocators with different properties (for example better memory utilization but less
performance) do exist. The README of the repository contains a section that discusses how this allocator
relates to other existing allocators on <crates.io>._ ⚠

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

## Minimal Code Example
```rust
#![feature(const_mut_refs)]
#![feature(allocator_api)]

use simple_chunk_allocator::{heap, heap_bitmap, GlobalChunkAllocator, PageAligned};

// The macros help to get a correctly sized arrays types.
// I page-align them for better caching and to improve the availability of
// page-aligned addresses.

/// Backing storage for heap (1Mib). (read+write) static memory in final executable.
///
/// heap!: first argument is chunk amount, second argument is size of each chunk.
///        If no arguments are provided it falls back to defaults.
///        Example: `heap!(chunks=16, chunksize=256)`.
static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
/// Backing storage for heap bookkeeping bitmap. (read+write) static memory in final executable.
///
/// heap_bitmap!: first argument is amount of chunks.
///               If no argument is provided it falls back to a default.
///               Example: `heap_bitmap!(chunks=16)`.
static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();

// please make sure that the backing memory is at least CHUNK_SIZE aligned; better page-aligned
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

## Another Code Example (Free Standing Linux Binary)
This is an excerpt. The code can be found in the GitHub repository in `freestanding-linux-example`.
```rust
static mut HEAP: PageAligned<[u8; 256]> = heap!(chunks = 16, chunksize = 16);
static mut HEAP_BITMAP: PageAligned<[u8; 2]> = heap_bitmap!(chunks = 16);

// please make sure that the backing memory is at least CHUNK_SIZE aligned; better page-aligned
#[global_allocator]
static ALLOCATOR: GlobalChunkAllocator<16> =
    unsafe { GlobalChunkAllocator::<16>::new(HEAP.deref_mut_const(), HEAP_BITMAP.deref_mut_const()) };

/// Referenced as entry by linker argument. Entry into the code.
#[no_mangle]
fn start() -> ! {
    write!(StdoutWriter, "Hello :)\n").unwrap();
    let mut vec = Vec::new();
    (0..10).for_each(|x| vec.push(x));
    write!(StdoutWriter, "vec: {:#?}\n", vec).unwrap();
    exit();
}
```

## MSRV
This crate only builds with the nightly version of Rust because it uses many nightly-only features. I developed it
with version `1.61.0-nightly` (2022-03-05). Older nightly versions might work. So far there is no stable Rust
compiler version that compiles this.

## Performance
The default CHUNK_SIZE is 256 bytes. It is a tradeoff between performance and efficient memory usage.

I executed my example `bench` in release mode on an Intel i7-1165G7 CPU and a heap of `160MB` to get the results listed
below. I used `RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench` to excute the benchmark with
maximum performance. The benchmark simulates a heavy usage of the heap in a single-threaded program with many random
allocations and deallocations. The benchmark stops when the heap is close to 100%. The allocations vary in their alignment.
The table below shows the results of this benchmark as number of clock cycles.

*Info: Since I measured those values, I slightly changed the benchmark.*

| Chunk Size    | # Chunks | # allocations | # deallocations | median | average  | min | max   |
|---------------|----------|---------------|-----------------|--------|----------|-----|-------|
| 128           | 1310720  | 68148         | 47915           | 955    | 1001     | 126 | 57989 |
| 256 [DEFAULT] | 655360   | 71842         | 51744           | 592    | 619      | 121 | 53578 |
| 512           | 327680   | 66672         | 46858           | 373    | 401      | 111 | 54403 |

The results vary slightly because each run gets influenced by some randomness. One can see that the performance
gets slower with a growing number of chunks. Increasing the chunk size reduces the size of the bookkeeping bitmap which
accelerates the lookup. However, a smaller chunk size occupies less heap when only very small allocations are required.

Note that performance is better than listed above when the heap is used less frequently and does not run full.

## Differences to Other Allocators
### good_memory_allocator (galloc)
**Update November 2022**: I recently found [this new project](https://github.com/MaderNoob/galloc)
and, from a first glance, I recommend to use this crate instead of mine for production usage. It has
impressive performance and heap utilization at the costs of more complicated code. The repository
includes interesting performance numbers from galloc, simple-chunk-allocator (this crate), and
linked-list-allocator.

### linked-list-allocator
**Update November 2022**: I wrote this paragraph before I found out about galloc. I left it
unchanged.

The [linked-list-allocator](https://github.com/rust-osdev/linked-list-allocator) is among the few
other well-suited and maintained general-purpose no-std allocator I could find on crates.io.

**Advantages of my chunk allocator:**
- much faster median allocation time
- much faster average allocation time [ONLY IF HEAP IS NOT CLOSE TO BEEING FULL]
- optimized realloc in certain cases (almost a no-op in some situations)
- uses relatively easy algorithm (but needs dedicated heap and book-keeping backing storage)

**Advantages of *linked-list-allocator*:**
- better memory utilization (less fragmentation)
- better worst-case allocation time in most test runs
- better average allocation time [ONLY IF HEAP IS CLOSE TO BEEING FULL]
- only needs a single chunk of memory and manages the heap with the backing-memory itself

**Benchmark Comparision**:
I ran `$ cargo run --example bench --release` against both allocators and obtained the following results. The benchmark
performs random allocations of different sizes and alignments and also deallocates some of the older allocations. Over
time, the heap becomes full, which is why the number of successful allocations has a higher delta to the attempted
allocations.

Runtime: 1s (most time lot's of heap available)
```
RESULTS OF BENCHMARK: Chunk Allocator
     53360 allocations,  16211 successful_allocations,  37149 deallocations
    median=   878 ticks, average=  1037 ticks, min=   158 ticks, max= 7178941 ticks

RESULTS OF BENCHMARK: Linked List Allocator
     31627 allocations,   9374 successful_allocations,  22253 deallocations
    median= 18582 ticks, average= 44524 ticks, min=    71 ticks, max=44126026 ticks
```

We see that as long as most allocations are done on a heap with lots of space available, the chunk
allocator is faster in median and average performance.

Runtime: 10s (most time heap almost full)
```
RESULTS OF BENCHMARK: Chunk Allocator
     74909 allocations,  23753 successful_allocations,  51156 deallocations
    median=   961 ticks, average=273362 ticks, min=   167 ticks, max=53330953 ticks

RESULTS OF BENCHMARK: Linked List Allocator
     81884 allocations,  24792 successful_allocations,  57092 deallocations
    median=100196 ticks, average=179495 ticks, min=    69 ticks, max=43937820 ticks
```

We see that when the heap is almost full, the chunk allocator has a faster median performance
but a worse worst-case allocation time. The linked list allocator performs better on average (but not on median)
when it is close to beeing full.
