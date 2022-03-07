/// Compiler hint wrapper around a bool for static branch prediction optimization.
/// I use this in favor of the intrinsic because otherwise the Rust plugin of Clion complains
/// that unlikely needs an unsafe block which is not true.
#[allow(non_snake_case)]
#[inline(always)]
pub(crate) fn UNLIKELY(bool: bool) -> bool {
    #[allow(unused_unsafe)]
    unsafe {
        core::intrinsics::unlikely(bool)
    }
}
