/*
MIT License

Copyright (c) 2022 Philipp Schuster

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
//! Module for macros [`heap`] and [`heap_bitmap`].

/// Helper macro that initializes a page-aligned static memory area with a correct size to
/// get used as heap in [`crate::GlobalChunkAllocator`].
///
/// # Example
/// ```rust
/// use simple_chunk_allocator::heap;
///
/// // chunk size: 256; chunk amount: 16
/// let heap = heap!(chunks = 256, chunksize = 16);
///
/// const CHUNK_SIZE: usize = 256;
/// const CHUNK_AMOUNT: usize = 24;
/// let heap = heap!(chunks = CHUNK_AMOUNT, chunksize = CHUNK_SIZE);
/// let heap = heap!(chunks = 24, chunksize = CHUNK_SIZE);
/// let heap = heap!(chunks = CHUNK_AMOUNT, chunksize = 256);
/// ```
#[macro_export]
macro_rules! heap {
    (chunks=$chunk_amount:literal, chunksize=$chunk_size:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    (chunks=$chunk_amount:literal, chunksize=$chunk_size:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    (chunks=$chunk_amount:path, chunksize=$chunk_size:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    (chunks=$chunk_amount:path, chunksize=$chunk_size:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    (chunks=$chunk_amount:literal) => {
        heap!(chunks=$chunk_amount, chunksize=$crate::DEFAULT_CHUNK_SIZE)
    };
    (chunks=$chunk_amount:path) => {
        heap!(chunks=$chunk_amount, chunksize=$crate::DEFAULT_CHUNK_SIZE)
    };
    () => {
        heap!(chunks=$crate::DEFAULT_CHUNK_AMOUNT, chunksize=$crate::DEFAULT_CHUNK_SIZE)
    };
}

/// Helper macro that initializes a page-aligned static memory area with a correct size to
/// get used as heap bookkeeping bitmap in [`crate::GlobalChunkAllocator`].
///
/// # Example
/// ```rust
/// use simple_chunk_allocator::heap_bitmap;
///
/// // chunk size: 256; chunk amount: 16
/// let heap_bitmap = heap_bitmap!(chunks = 16);
///
/// const CHUNK_AMOUNT: usize = 24;
/// let heap_bitmap = heap_bitmap!(chunks = CHUNK_AMOUNT);
/// ```
#[macro_export]
macro_rules! heap_bitmap {
    (chunks=$chunk_amount:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount / 8])
    };
    (chunks=$chunk_amount:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount / 8])
    };
    () => {
        heap_bitmap!(chunks=$crate::DEFAULT_CHUNK_AMOUNT)
    };
}

#[cfg(test)]
mod tests {

    // Tests that the macro `heap!` compiles with all supported input types.
    #[test]
    fn test_macro_heap_compiles() {
        const A: usize = 8;
        const B: usize = 16;
        let _heap_implicit = heap!(chunks = 8);
        let _heap_explicit = heap!(chunks = 8, chunksize = 16);

        let _heap_implicit = heap!(chunks = A);
        let _heap_explicit = heap!(chunks = 8, chunksize = B);
        let _heap_explicit = heap!(chunks = A, chunksize = B);
        let _heap_explicit = heap!(chunks = A, chunksize = 16);

        // overflows stack in tests :(
        /*let heap_default = heap!();
        assert_eq!(
            heap_default.len(),
            DEFAULT_CHUNK_AMOUNT * DEFAULT_CHUNK_SIZE,
            "default heap size is expected to equal to 1MiB"
        );*/
    }

    // Tests that the macro `heap_bitmap!` compiles with all supported input types.
    #[test]
    fn test_macro_heap_bitmap_compiles() {
        const A: usize = 8;
        let _heap_implicit = heap_bitmap!();
        let _heap_explicit_lit = heap_bitmap!(chunks = 8);
        let _heap_explicit_path = heap_bitmap!(chunks = A);
    }
}
