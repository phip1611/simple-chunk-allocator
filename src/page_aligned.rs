/*
MIT License

Copyright (c) 2026 Philipp Schuster

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
//! Module for [`PageAligned`].

use core::ops::Deref;
use core::ops::DerefMut;

/// Wrapper around a `T` that gets page-aligned.
/// All important methods allow the usage in const contexts.
#[derive(Debug)]
#[repr(align(4096))]
pub struct PageAligned<T>(T);

impl<T> PageAligned<T> {
    /// Constructor.
    pub const fn new(t: T) -> Self {
        Self(t)
    }

    /// Like [`Deref::deref`] but const.
    pub const fn deref_const(&self) -> &T {
        &self.0
    }

    /// Like [`DerefMut::deref_mut`] but const.
    pub const fn deref_mut_const(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> From<T> for PageAligned<T> {
    fn from(t: T) -> Self {
        Self(t)
    }
}

impl<T> Deref for PageAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for PageAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
