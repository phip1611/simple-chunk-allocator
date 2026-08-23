/*
 * This freestanding Linux binary uses no libc but can start and exit gracefully.
 * It uses the "simple-chunk-allocator" crate for heap management.
 */

#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

extern crate alloc;

use alloc::vec::Vec;
use core::alloc::Layout;
use core::arch::asm;
use core::fmt::Write;
use core::panic::PanicInfo;
use simple_chunk_allocator::GlobalChunkAllocator;

type Allocator = GlobalChunkAllocator<16>;

const REGION_SIZE: usize = Allocator::required_region_size(16);
static mut REGION: [u8; REGION_SIZE] = [0; REGION_SIZE];

#[global_allocator]
// SAFETY: `ALLOCATOR` is the only user of `REGION` for the whole program.
static ALLOCATOR: Allocator =
    unsafe { Allocator::new((&raw mut REGION).cast(), REGION_SIZE) };

/// Referenced as entry by linker argument. Entry into the code.
#[no_mangle]
fn start() -> ! {
    writeln!(StdoutWriter, "Hello :)").unwrap();
    // allocations work; uses the global allocator
    let mut vec = Vec::new();
    (0..10).for_each(|x| vec.push(x));
    writeln!(StdoutWriter, "vec: {:#?}", vec).unwrap();
    exit();
}

// --------------------------------------------------------------

const LINUX_SYS_WRITE: u64 = 1;
const LINUX_SYS_EXIT_GROUP: u64 = 231;

struct StdoutWriter;

impl core::fmt::Write for StdoutWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        const STDOUT: u64 = 1;
        // SAFETY: the syscall arguments describe the valid bytes of `s`.
        let written_bytes = unsafe {
            linux_syscall_3(LINUX_SYS_WRITE, STDOUT, s.as_ptr() as _, s.len() as _)
        };
        assert_eq!(
            written_bytes as usize,
            s.len(),
            "currently only supports to write all bytes in once!"
        );
        Ok(())
    }
}

fn exit() -> ! {
    // SAFETY: `exit_group` takes no pointer arguments and never returns.
    unsafe {
        linux_syscall_3(LINUX_SYS_EXIT_GROUP, 0, 0, 0);
    }
    unreachable!();
}

/// Wrapper around a Linux syscall with three arguments. It returns
/// the syscall result (or error code) that gets stored in rax.
unsafe fn linux_syscall_3(num: u64, arg1: u64, arg2: u64, arg3: u64) -> i64 {
    let res;
    asm!(
        // there is no need to write "mov"-instructions, see below
        "syscall",
        // from 'in("rax")' the compiler will
        // generate corresponding 'mov'-instructions
        in("rax") num,
        in("rdi") arg1,
        in("rsi") arg2,
        in("rdx") arg3,
        lateout("rax") res,
    );
    res
}

#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    let _ = writeln!(StdoutWriter, "PANIC: {:?}", info);
    exit();
}

#[alloc_error_handler]
fn alloc_error_handler(layout: Layout) -> ! {
    panic!("Can't handle allocation: layout = {:?}", layout);
}
