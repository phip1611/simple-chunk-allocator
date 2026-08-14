#![feature(slice_ptr_get)]

use core::alloc::{GlobalAlloc, Layout};
use core::hint::black_box;
use core::ptr::NonNull;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use simple_chunk_allocator::{
    ChunkAllocator, DEFAULT_CHUNK_SIZE, GlobalChunkAllocator, PageAligned,
};
use std::boxed::Box;

const CHUNK_COUNT: usize = 1024;
const HEAP_SIZE: usize = CHUNK_COUNT * DEFAULT_CHUNK_SIZE;
const BITMAP_SIZE: usize = CHUNK_COUNT / 8;

struct Fixture {
    // These fields keep the raw backing storage alive for `allocator`.
    _heap: Box<PageAligned<[u8; HEAP_SIZE]>>,
    _bitmap: Box<PageAligned<[u8; BITMAP_SIZE]>>,
    allocator: ChunkAllocator<'static>,
}

impl Fixture {
    fn empty() -> Self {
        let mut heap_memory = Box::new(PageAligned::new([0; HEAP_SIZE]));
        let mut bitmap_memory = Box::new(PageAligned::new([0; BITMAP_SIZE]));
        let heap = core::ptr::slice_from_raw_parts_mut(
            heap_memory.deref_mut_const().as_mut_ptr(),
            HEAP_SIZE,
        );
        let bitmap = core::ptr::slice_from_raw_parts_mut(
            bitmap_memory.deref_mut_const().as_mut_ptr(),
            BITMAP_SIZE,
        );

        // SAFETY: the boxed regions are valid, disjoint, and exclusively
        // owned by this fixture until its allocator is dropped.
        let allocator = unsafe { ChunkAllocator::new_raw(heap, bitmap) };
        let mut fixture = Self {
            _heap: heap_memory,
            _bitmap: bitmap_memory,
            allocator,
        };
        fixture.initialize();
        fixture
    }

    fn initialize(&mut self) {
        let layout = Layout::from_size_align(1, 1).unwrap();
        let allocation = self.allocator.allocate(layout).unwrap();

        // SAFETY: `allocation` is the live allocation returned for `layout`.
        unsafe {
            self.allocator
                .deallocate(allocation.as_non_null_ptr(), layout)
        };
    }

    fn allocate(&mut self, layout: Layout) -> NonNull<u8> {
        self.allocator.allocate(layout).unwrap().as_non_null_ptr()
    }

    fn fill(&mut self, layout: Layout) {
        while self.allocator.allocate(layout).is_ok() {}
    }

    fn fragment(&mut self, layout: Layout) {
        let mut allocations = Vec::new();
        while let Ok(allocation) = self.allocator.allocate(layout) {
            allocations.push(allocation.as_non_null_ptr());
        }
        for (index, allocation) in allocations.into_iter().enumerate() {
            if index % 2 == 0 {
                // SAFETY: every pointer was allocated with this layout.
                unsafe { self.allocator.deallocate(allocation, layout) };
            }
        }
    }
}

struct GlobalFixture {
    // These fields keep the raw backing storage alive for `allocator`.
    _heap: Box<PageAligned<[u8; HEAP_SIZE]>>,
    _bitmap: Box<PageAligned<[u8; BITMAP_SIZE]>>,
    allocator: GlobalChunkAllocator<'static>,
}

impl GlobalFixture {
    fn empty() -> Self {
        let mut heap_memory = Box::new(PageAligned::new([0; HEAP_SIZE]));
        let mut bitmap_memory = Box::new(PageAligned::new([0; BITMAP_SIZE]));
        let heap = core::ptr::slice_from_raw_parts_mut(
            heap_memory.deref_mut_const().as_mut_ptr(),
            HEAP_SIZE,
        );
        let bitmap = core::ptr::slice_from_raw_parts_mut(
            bitmap_memory.deref_mut_const().as_mut_ptr(),
            BITMAP_SIZE,
        );

        // SAFETY: the boxed regions are valid, disjoint, and exclusively
        // owned by this fixture until its allocator is dropped.
        let allocator = unsafe { GlobalChunkAllocator::new_raw(heap, bitmap) };
        let fixture = Self {
            _heap: heap_memory,
            _bitmap: bitmap_memory,
            allocator,
        };
        fixture.initialize();
        fixture
    }

    fn initialize(&self) {
        let layout = Layout::from_size_align(1, 1).unwrap();
        let ptr = self.allocate(layout);
        self.deallocate(ptr, layout);
    }

    fn allocate(&self, layout: Layout) -> NonNull<u8> {
        // SAFETY: `layout` is valid and this fixture exclusively owns the
        // allocator.
        let ptr = unsafe { self.allocator.alloc(layout) };
        NonNull::new(ptr).expect("benchmark allocation must fit")
    }

    fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: `ptr` is a live allocation returned by this allocator for
        // `layout`.
        unsafe { self.allocator.dealloc(ptr.as_ptr(), layout) };
    }

    fn fragment(&self, layout: Layout) {
        let mut allocations = Vec::new();
        loop {
            // SAFETY: `layout` is valid and this fixture exclusively owns the
            // allocator.
            let ptr = unsafe { self.allocator.alloc(layout) };
            let Some(ptr) = NonNull::new(ptr) else {
                break;
            };
            allocations.push(ptr);
        }
        for (index, allocation) in allocations.into_iter().enumerate() {
            if index % 2 == 0 {
                self.deallocate(allocation, layout);
            }
        }
    }
}

fn benchmark_phases(criterion: &mut Criterion) {
    let one_chunk = Layout::from_size_align(DEFAULT_CHUNK_SIZE, 1).unwrap();
    let mut group = criterion.benchmark_group("phases/search");

    group.bench_function("empty", |bencher| {
        bencher.iter_batched(
            Fixture::empty,
            |mut fixture| {
                black_box(
                    fixture
                        .allocator
                        .benchmark_find_free_region(4, DEFAULT_CHUNK_SIZE),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("sparse", |bencher| {
        bencher.iter_batched(
            || {
                let mut fixture = Fixture::empty();
                for _ in 0..CHUNK_COUNT / 4 {
                    fixture.allocate(one_chunk);
                }
                fixture
            },
            |mut fixture| {
                black_box(
                    fixture
                        .allocator
                        .benchmark_find_free_region(4, DEFAULT_CHUNK_SIZE),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("fragmented", |bencher| {
        bencher.iter_batched(
            || {
                let mut fixture = Fixture::empty();
                fixture.fragment(one_chunk);
                fixture
            },
            |mut fixture| {
                black_box(
                    fixture
                        .allocator
                        .benchmark_find_free_region(2, DEFAULT_CHUNK_SIZE),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("full", |bencher| {
        bencher.iter_batched(
            || {
                let mut fixture = Fixture::empty();
                fixture.fill(one_chunk);
                fixture
            },
            |mut fixture| {
                black_box(
                    fixture
                        .allocator
                        .benchmark_find_free_region(1, DEFAULT_CHUNK_SIZE),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();

    let mut group = criterion.benchmark_group("phases/bitmap");
    for chunk_count in [1, 4, 16, 64] {
        group.bench_function(format!("mark_used/{chunk_count}"), |bencher| {
            bencher.iter_batched(
                Fixture::empty,
                |mut fixture| {
                    fixture
                        .allocator
                        .benchmark_mark_range_as_used(0, chunk_count);
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("mark_free/{chunk_count}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut fixture = Fixture::empty();
                    fixture
                        .allocator
                        .benchmark_mark_range_as_used(0, chunk_count);
                    fixture
                },
                |mut fixture| {
                    fixture
                        .allocator
                        .benchmark_mark_range_as_free(0, chunk_count);
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_direct_operations(criterion: &mut Criterion) {
    let layouts = [
        (
            "one_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE, 1).unwrap(),
        ),
        (
            "multi_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE * 16, 1).unwrap(),
        ),
        ("page_aligned", Layout::from_size_align(4096, 4096).unwrap()),
    ];
    let mut group = criterion.benchmark_group("direct");

    for (name, layout) in layouts {
        group.bench_function(format!("allocate/{name}"), |bencher| {
            bencher.iter_batched(
                Fixture::empty,
                |mut fixture| black_box(fixture.allocate(layout)),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("deallocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut fixture = Fixture::empty();
                    let ptr = fixture.allocate(layout);
                    (fixture, ptr)
                },
                |(mut fixture, ptr)| {
                    // SAFETY: setup allocated `ptr` with `layout`.
                    unsafe { fixture.allocator.deallocate(ptr, layout) };
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("cycle/{name}"), |bencher| {
            let mut fixture = Fixture::empty();
            bencher.iter(|| {
                let ptr = fixture.allocate(layout);
                // SAFETY: this iteration allocated `ptr` with `layout`.
                unsafe { fixture.allocator.deallocate(ptr, layout) };
                black_box(ptr)
            });
        });
    }
    group.finish();
}

fn benchmark_global_operations(criterion: &mut Criterion) {
    let layouts = [
        (
            "one_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE, 1).unwrap(),
        ),
        (
            "multi_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE * 16, 1).unwrap(),
        ),
        ("page_aligned", Layout::from_size_align(4096, 4096).unwrap()),
    ];
    let mut group = criterion.benchmark_group("global");

    for (name, layout) in layouts {
        group.bench_function(format!("allocate/{name}"), |bencher| {
            bencher.iter_batched(
                GlobalFixture::empty,
                |fixture| black_box(fixture.allocate(layout)),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("deallocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let fixture = GlobalFixture::empty();
                    let ptr = fixture.allocate(layout);
                    (fixture, ptr)
                },
                |(fixture, ptr)| {
                    fixture.deallocate(ptr, layout);
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("cycle/{name}"), |bencher| {
            let fixture = GlobalFixture::empty();
            bencher.iter(|| {
                let ptr = fixture.allocate(layout);
                fixture.deallocate(ptr, layout);
                black_box(ptr)
            });
        });
    }
    group.finish();
}

fn benchmark_fragmented_operations(criterion: &mut Criterion) {
    let layouts = [
        (
            "one_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE, 1).unwrap(),
        ),
        (
            "multi_chunk",
            Layout::from_size_align(DEFAULT_CHUNK_SIZE * 16, 1).unwrap(),
        ),
        ("page_aligned", Layout::from_size_align(4096, 4096).unwrap()),
    ];
    let mut direct = criterion.benchmark_group("direct_fragmented");

    for (name, layout) in layouts {
        direct.bench_function(format!("allocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut fixture = Fixture::empty();
                    fixture.fragment(layout);
                    fixture
                },
                |mut fixture| black_box(fixture.allocate(layout)),
                BatchSize::SmallInput,
            );
        });
        direct.bench_function(format!("deallocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut fixture = Fixture::empty();
                    fixture.fragment(layout);
                    let ptr = fixture.allocate(layout);
                    (fixture, ptr)
                },
                |(mut fixture, ptr)| {
                    // SAFETY: setup allocated `ptr` with `layout`.
                    unsafe { fixture.allocator.deallocate(ptr, layout) };
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
        direct.bench_function(format!("cycle/{name}"), |bencher| {
            let mut fixture = Fixture::empty();
            fixture.fragment(layout);
            bencher.iter(|| {
                let ptr = fixture.allocate(layout);
                // SAFETY: this iteration allocated `ptr` with `layout`.
                unsafe { fixture.allocator.deallocate(ptr, layout) };
                black_box(ptr)
            });
        });
    }
    direct.finish();

    let mut global = criterion.benchmark_group("global_fragmented");
    for (name, layout) in layouts {
        global.bench_function(format!("allocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let fixture = GlobalFixture::empty();
                    fixture.fragment(layout);
                    fixture
                },
                |fixture| black_box(fixture.allocate(layout)),
                BatchSize::SmallInput,
            );
        });
        global.bench_function(format!("deallocate/{name}"), |bencher| {
            bencher.iter_batched(
                || {
                    let fixture = GlobalFixture::empty();
                    fixture.fragment(layout);
                    let ptr = fixture.allocate(layout);
                    (fixture, ptr)
                },
                |(fixture, ptr)| {
                    fixture.deallocate(ptr, layout);
                    black_box(fixture.allocator.usage())
                },
                BatchSize::SmallInput,
            );
        });
        global.bench_function(format!("cycle/{name}"), |bencher| {
            let fixture = GlobalFixture::empty();
            fixture.fragment(layout);
            bencher.iter(|| {
                let ptr = fixture.allocate(layout);
                fixture.deallocate(ptr, layout);
                black_box(ptr)
            });
        });
    }
    global.finish();
}

criterion_group!(
    benches,
    benchmark_phases,
    benchmark_direct_operations,
    benchmark_global_operations,
    benchmark_fragmented_operations,
);
criterion_main!(benches);
