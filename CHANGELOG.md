# Changelog

## Unreleased

- All UB fixed.
- Fixed some bugs
- Improved documentation
- **Breaking:** an invalid `CHUNK_SIZE` is a compile error instead of a
  runtime one. `ChunkAllocatorError::BadChunkSize` is removed.
- **Breaking:** one constructor remains. `ChunkAllocator::new` and
  `GlobalChunkAllocator::new` are `const unsafe fn`; `new_const` and `new_raw`
  are removed.
- **Breaking:** the lifetime parameter of `ChunkAllocator` and
  `GlobalChunkAllocator` is gone. The obligation it stood for is part of the
  safety contract of `new`.
- **Breaking:** the heap may have any length and any alignment. The allocator
  derives the chunk count from the length and aligns the first chunk itself,
  so the requirements on both are gone, and with them the
  `ChunkAllocatorError` variants that reported them. The remaining failure is
  `OutOfMemory`, now a struct with `Display` and `core::error::Error`.
- **Breaking:** the allocator takes a single memory region and places its
  bitmap at the end of it. The separate bitmap parameter is gone, and the
  region may hold uninitialized content.
- Added `required_region_size`, which reports the region size that holds a
  given chunk count at any alignment.
- **Breaking:** removed the `heap!` and `heap_bitmap!` macros, `PageAligned`,
  and `DEFAULT_CHUNK_AMOUNT`. Use `required_region_size` to size a region, and
  a `#[repr(align(...))]` wrapper if a specific alignment is wanted.
- **Breaking:** `usage` returns a ratio between `0.0` and `1.0` instead of a
  percentage rounded to two decimals. This drops the `libm` dependency.
- `GlobalChunkAllocator` forwards `capacity` and `chunk_count`.
- `AllocatorApiGlue::grow` reports an alignment it cannot serve as `AllocError`
  instead of panicking.
- `AllocatorApiGlue` implements `shrink`, so an allocation that shrinks within
  the chunks it already owns keeps its place instead of being copied.
- `GlobalChunkAllocator` also forwards `chunk_size` and `min_alignment`.
- **Breaking:** the allocator takes a single memory region and places its
  bitmap at the end of it. The separate bitmap parameter is gone, and the
  region may hold uninitialized content.
- Added `required_region_size`, which reports the region size that holds a
  given chunk count at any alignment.
- **Breaking:** removed the `heap!` and `heap_bitmap!` macros, `PageAligned`,
  and `DEFAULT_CHUNK_AMOUNT`. Use `required_region_size` to size a region, and
  a `#[repr(align(...))]` wrapper if a specific alignment is wanted.
- **Breaking:** `usage` returns a ratio between `0.0` and `1.0` instead of a
  percentage rounded to two decimals. This drops the `libm` dependency.
- `GlobalChunkAllocator` forwards `capacity`, `chunk_count`, `chunk_size` and
  `min_alignment`.
- `AllocatorApiGlue::grow` reports an alignment it cannot serve as `AllocError`
  instead of panicking.
- `AllocatorApiGlue` implements `shrink`, so an allocation that shrinks within
  the chunks it already owns keeps its place instead of being copied.
- Deallocation points the next search at the freed region. The rule that was
  there before could never fire, so a buffer that is allocated and freed in a
  loop was searched for across the whole heap every time.

## v0.1.6 (2024-09-29)
I discourage the use of this library, please look for an alternative. Use it
only as learning resource or so. The library contains a few cases that produce
UB.

There are better alternatives, such as <https://crates.io/crates/talc>.

- Fixed one case of UB (but not the one of the realloc implementation)

## v0.1.5 (2022-03-17)
- optimized "realloc" method if existing memory chunk(s)
  allocation is already big enough

## v0.1.4 (2022-03-17)
- improved API for macros: they now include named parameters (that are optional; defaults are used then)
  - `heap!(chunks=16, chunksize=256)`
  - `heap_bitmap!(chunks=16)`
- check heap alignment on first allocation (because alignment of the backing memory can not be guaranteed during
  const time) (see <https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248>)
