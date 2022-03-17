# Freestanding x86_64 Linux Example that uses simple-chunk-allocator

## TL;DR:
Just type `cargo run`.

## Details
This is a freestanding Linux binary. This means, technically it is a "no_std" binary that can run under Linux
and shut down gracefully. It only works on x86_64 and uses my simple chunk allocator to manage its heap.
The architectural limitation comes from the build process and is not a limitation of my allocator.

Using this allocator approach is in contrast to libc that uses `brk` and `mmap` syscalls for that. However, this
is just an example. Under Linux, you always want to use `mmap` etc. However, if you have a freestanding binary
without available memory management, simple chunk allocator is a nice and convenient way to go with a nice
API.
