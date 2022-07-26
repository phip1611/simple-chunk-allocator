# Changelog

## v0.1.5 (2022-03-17)
- optimized "realloc" method if existing memory chunk(s)
  allocation is already big enough

## v0.1.4 (2022-03-17)
- improved API for macros: they now include named parameters (that are optional; defaults are used then)
  - `heap!(chunks=16, chunksize=256)`
  - `heap_bitmap!(chunks=16)`
- check heap alignment on first allocation (because alignment of the backing memory can not be guaranteed during
  const time) (see <https://github.com/rust-lang/rust/issues/90962#issuecomment-1064148248>)
