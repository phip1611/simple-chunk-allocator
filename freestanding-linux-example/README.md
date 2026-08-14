# Freestanding x86_64 Linux Example

This `no_std` example uses `simple-chunk-allocator` for a static heap and
performs Linux syscalls directly. It requires x86_64 Linux.

Run it with `cargo run` from this directory. It illustrates allocator setup;
normal Linux programs should use the system allocator instead.
