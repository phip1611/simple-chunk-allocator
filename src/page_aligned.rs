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
