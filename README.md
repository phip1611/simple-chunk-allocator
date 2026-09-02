# Simple Chunk Allocator

A `no_std` allocator that manages fixed-size chunks in a single
caller-provided memory region.

It is suitable for small static heaps in kernels, bootloaders, and freestanding
binaries.

## Highlights

- ✅ `no_std` allocator with test coverage
- ✅ uses one caller-provided region as backing storage (no paging/page table
  manipulations)
- ✅ next-fit allocation that reuses the most recently freed region first
- ✅ reasonably fast with low code complexity
- ✅ const compatibility (no runtime `init()` required)
- ✅ efficient in scenarios where the heap is a few dozen megabytes in size
- ✅ small API: one constructor, no macros, no alignment wrappers

## When to use it

Use this crate when the heap size is known in advance and predictable memory
use matters more than fine-grained allocation efficiency. Every allocation uses
whole chunks, so a small allocation can consume up to one complete chunk.

The default chunk size is 256 bytes. Choose a larger chunk size for faster
searches and a smaller bitmap; choose a smaller chunk size to waste less space
on small allocations.

## Requirements

- Rust 1.85 or newer (MSRV).
- Optional: the `unstable` cargo feature enables the integration with the
  nightly `allocator_api` and requires a nightly toolchain.
- One writable memory region. Its length decides how many chunks fit, and it
  needs neither a particular alignment nor initialized content.

The bitmap lives at the end of that region, so a region holds slightly fewer
chunks than `len / chunk_size`. Use `required_region_size` to size storage for
a given chunk count.

## Alignment

The allocator aligns the first chunk itself, which makes every chunk aligned to
the chunk size and costs up to `chunk_size - 1` bytes of padding. Passing a
region that is already chunk-aligned - a page-aligned one is the easiest way -
avoids that.

An allocation asking for more alignment than the chunk size is served by every
`alignment / chunk_size`-th chunk. Which ones those are shifts with the region,
but they exist wherever it starts: page-aligned allocations come out of a
merely chunk-aligned region. Such a request needs a free run in the right
place, which is a question of heap size and fragmentation, not of alignment.

## Global allocator

```rust
use simple_chunk_allocator::GlobalChunkAllocator;

/// Named once, so that the chunk size is stated once.
type Allocator = GlobalChunkAllocator<256>;

const REGION_SIZE: usize = Allocator::required_region_size(4096);
static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];

#[global_allocator]
// SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
static ALLOCATOR: Allocator =
    unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };

fn main() {
    let mut values = Vec::new();
    values.push(42);
}
```

`new` is unsafe because the allocator cannot verify the region's lifetime or
that it is its only user. See its API documentation for the complete contract.

## Direct use and allocator API

With the `unstable` cargo feature (requires a nightly toolchain),
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
`deallocate`, and `realloc`. It is not synchronized, so it needs `&mut self`:

```rust
let mut region = [0_u8; 4096];
// SAFETY: `region` outlives the allocator and nothing else touches it.
let mut allocator = unsafe {
    ChunkAllocator::<256>::new(region.as_mut_ptr(), region.len())
};

let layout = Layout::from_size_align(64, 8).unwrap();
let allocation = allocator.allocate(layout).unwrap();
// SAFETY: `allocation` is live and paired with its original layout.
unsafe { allocator.deallocate(allocation.cast(), layout) };
```

The snippets above are shortened. The API documentation carries the same
examples in full, where they are compiled and run as doctests.

Only deallocate or reallocate pointers returned by the same allocator, using
the original layout. These operations are unsafe because a mismatched pointer
or layout can corrupt the allocator.

## Testing

```
cargo test                              # stable, default features
cargo +nightly test --all-features
cargo +nightly miri test --all-features
MIRIFLAGS="-Zmiri-many-seeds=0..12" cargo +nightly miri test --all-features
```

Miri checks the crate's own pointer arithmetic, aliasing and alignment, and it
covers every test including the doctests. It cannot check what the allocator
promises its callers: the whole region is a single allocation to Miri, so an
overrun from one chunk into the next, or a write to a chunk after it was freed,
is invisible to it. Only an access that leaves the region entirely is reported.

That is why the tests give every allocation its own byte pattern and read it
back: an overlap between two live allocations shows up as a foreign byte. Those
checks are not redundant with Miri, they are the only check for that property.

## Performance

TODO: The previous benchmark numbers were measured against an older version of
`examples/bench.rs` and are therefore removed. New numbers follow once the
benchmark has been reworked.

## Cargo features

The crate has no default features. The `unstable` feature enables the
`allocator_api` integration (`AllocatorApiGlue`) and requires a nightly
toolchain.

## Examples

The `examples/minimal.rs` program demonstrates a hosted global allocator; it
uses the allocator API, so it needs
`cargo +nightly run --example minimal --features unstable`. The
`freestanding-linux-example` directory contains an x86_64 Linux binary without
libc. It is illustrative only; normal Linux programs should use the operating
system allocator.
