# Simple Chunk Allocator

A simple allocator written in Rust that manages memory in fixed-size chunks. Can be used as
`#[global_allocator]` in Rust as well as with the `allocator_api` feature to use this allocator
in types of the Rust standard library, such as `Vec` or `BTreeMap`. This project originates from
my [Diplom thesis project](https://github.com/phip1611/diplomarbeit-impl).

Because I had lots of struggles to create this (my first ever allocator), I outsourced it for better
testability and to share my knowledge and findings with others in the hope that someone can learn from it
in any way.

TL;DR: First-fit allocators are a stupid approach because if the heap is big and usage increases, the steps required
to look into the bookkeeping data structure grows linear. This was a problem in my Diplom thesis project that
gave me terrible evaluation benchmark measurements. However, I could fix this with this optimized version of
a next fit allocator.
