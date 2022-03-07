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
/// let heap = heap!(256, 16);
///
/// const CHUNK_SIZE: usize = 256;
/// const CHUNK_AMOUNT: usize = 24;
/// let heap = heap!(CHUNK_SIZE, CHUNK_AMOUNT);
/// let heap = heap!(24, CHUNK_AMOUNT);
/// ```
#[macro_export]
macro_rules! heap {
    ($chunk_amount:literal, $chunk_size:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    ($chunk_amount:literal, $chunk_size:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    ($chunk_amount:path, $chunk_size:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    ($chunk_amount:path, $chunk_size:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount * $chunk_size])
    };
    ($chunk_amount:literal) => {
        heap!($chunk_amount, $crate::DEFAULT_CHUNK_SIZE)
    };
    ($chunk_amount:path) => {
        heap!($chunk_amount, $crate::DEFAULT_CHUNK_SIZE)
    };
    () => {
        heap!($crate::DEFAULT_CHUNK_AMOUNT, $crate::DEFAULT_CHUNK_SIZE)
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
/// let heap_bitmap = heap_bitmap!(16);
///
/// const CHUNK_AMOUNT: usize = 24;
/// let heap_bitmap = heap_bitmap!(CHUNK_AMOUNT);
/// ```
#[macro_export]
macro_rules! heap_bitmap {
    ($chunk_amount:path) => {
        $crate::PageAligned::new([0_u8; $chunk_amount / 8])
    };
    ($chunk_amount:literal) => {
        $crate::PageAligned::new([0_u8; $chunk_amount / 8])
    };
    () => {
        heap_bitmap!($crate::DEFAULT_CHUNK_AMOUNT)
    };
}

#[cfg(test)]
mod tests {

    // Tests that the macro `heap!` compiles with all supported input types.
    #[test]
    fn test_macro_heap_compiles() {
        const A: usize = 8;
        const B: usize = 16;
        let _heap_implicit = heap!(8);
        let _heap_explicit = heap!(8, 16);

        let _heap_implicit = heap!(A);
        let _heap_explicit = heap!(8, B);
        let _heap_explicit = heap!(A, B);
        let _heap_explicit = heap!(A, 16);

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
        let _heap_explicit_lit = heap_bitmap!(8);
        let _heap_explicit_path = heap_bitmap!(A);
    }
}
