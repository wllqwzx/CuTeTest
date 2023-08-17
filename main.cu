#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/container/tuple.hpp>
#include <cute/int_tuple.hpp>
#include <iostream>

#include <cuda_fp16.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/dispatch_policy.hpp>

#define p(v) std::cout << #v << " : " << v << std::endl
#define kp(v) cute::print(#v " : "); cute::print(v); cute::print("\n")
#define pp(v) if (cute::thread0()) { printf("L%u : ", __LINE__); cute::print(#v " : "); cute::print(v); cute::print("\n"); }

// #define p(v)
// #define kp(v)
// #define pp(v)


void test_int_tuple() {
    // auto tup = cute::IntTuple<int, uint32_t, size_t>({1, 2U, 3U});
    auto tup = cute::make_tuple(1, 2U, 3U);
    p(tup);               // (1,2,3)
    p(cute::get<2>(tup)); // 3
    p(cute::rank(tup));   // _3
    p(cute::depth(tup));  // _1
    p(cute::size(tup));   // 6

    // auto tup_nest = cute::IntTuple<int, cute::IntTuple<int, int>>({2, cute::IntTuple<int, int>({3, 4})});
    auto tup_nest = cute::make_tuple(2, cute::make_tuple(3, cute::Int<4>{}));
    p(tup_nest);               // (2,(3,4))
    p(cute::get<1>(tup_nest)); // (3,4)
    p(cute::rank(tup_nest));   // _2
    p(cute::depth(tup_nest));  // _2
    p(cute::size(tup_nest));   // 24
}

void test_layout() {
    // auto ly = cute::Layout<cute::IntTuple<int, int, int>, cute::IntTuple<int, int, int>>({3,4,5}, {1,3,12});
    // auto ly = cute::make_layout(cute::make_shape(3,4,5), cute::make_stride(1,3,12));
    auto ly = cute::make_layout(cute::make_shape(3, 4, 5)); // use default stride
    p(ly);                // (3,4,5):(_1,3,12)
    p(cute::size<0>(ly)); // 3
    p(cute::size<1>(ly)); // 4
    p(cute::size<2>(ly)); // 5
    p(cute::size(ly));    // 60
    p(cute::cosize(ly));  // 60 = ly(cute::size(ly) - 1) + 1, last elem address
    p(ly.shape());        // (3,4,5)
    p(ly.stride());       // (_1,3,12)
    p(ly(0, 0, 0));       // 0
    p(ly(1, 0, 0));       // 1
    p(ly(0, 1, 0));       // 3
    p(ly(0, 0, 1));       // 12
    p(ly(2, 3, 4));       // 59
    p(ly(0));             // 0
    p(ly(1));             // 1
    p(ly(59));            // 59

    // assume ly = (a1, a2):(s1, s2), then ly(x1, x2) = x1 * s1 + x2 * s2
    // therefore, transpose only need to change the order of shape and stride, and
    // the memory can keep unchanged
    // ((a1, a2):(s1, s2))(x1, x2) = ((a2, a1):(s2, s1))(x2, x1)

    // show 2d coord to offset ascii view, only support 2d layout
    cute::print_layout(cute::make_layout(cute::make_shape(8, 4), cute::make_shape(4, 2)));
    // or print_latex for latex source
    // cute::print_latex(cute::make_layout(cute::make_shape(8, 4), cute::make_shape(4, 2)));
}

void test_layout_opeartion() {
    auto nest_ly = cute::make_layout(cute::make_shape(4, cute::make_shape(4, 2)),
                                     cute::make_shape(4, cute::make_shape(1, 16)));
    p(nest_ly);                 // (4,(4,2)):(4,(1,16))
    p(cute::flatten(nest_ly));  // (4,4,2):(4,1,16)

    // The coalesce operation first flattens the layout, then combines all the modes that are possible to
    // combine, starting with mode 0 (the leftmost mode) and moving right.
    p(cute::coalesce(nest_ly));     // (4,4,2):(4,1,16)
    auto nest_ly2 = cute::make_layout(cute::make_shape(cute::_8{}, cute::_4{}));
    p(cute::coalesce(nest_ly2));    // _32:_1

    // composition
    //   - composition(ly_a, ly_b)(idx) == ly_a(ly_b(idx))
    //   - layout maps coord (or idx) to offset
    //   - compose ly_a with ly_b requires ly_a's max input idx cover ly_b's max output offset
    //   - not any two layouts can be composed, because the result mapping may not be able to be
    //     represented with shape + stride
    using namespace cute;
    auto ly_a = make_layout(make_shape(Int<20>{}, _2{}), make_stride(_16{}, _4{}));
    auto ly_b = make_layout(make_shape(     _4{}, _5{}), make_stride( _1{}, _4{}));
    auto ly_ab = composition(ly_a, ly_b);
    p(ly_ab);           // (_4,_5):(_16,_64)
    p(ly_ab(5));        // 80
    p(ly_a(ly_b(5)));   // 80

    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_24>{});
    p(composition(a, b));

    // layout products: reproduce one layout over another
    Layout tile            = Layout<Shape <_2,_2>,
                                    Stride<_1,_2>>{};
    Layout matrix_of_tiles = Layout<Shape <_3,_4>,
                                    Stride<_4,_1>>{};

    p(tile);                                    // (_2,_2):(_1,_2)
    p(matrix_of_tiles);                         // (_3,_4):(_4,_1)
    p(logical_product(tile, matrix_of_tiles));  // ((_2,_2),(_3,_4)):((_1,_2),(_16,_4))
    p(blocked_product(tile, matrix_of_tiles));  // ((_2,_3),_8):((_1,_16),_2)
    p(raked_product(tile, matrix_of_tiles));    // ((_3,_2),(_4,_2)):((_16,_1),(_4,_2))
    p(tiled_product(tile, matrix_of_tiles));    // ((_2,_2),_3,_4):((_1,_2),_16,_4)

    // layout division: divide a layout into components, are useful as a basis for tiling and partitioning layouts.
    Layout full_layout = Layout<Shape<_16>, Stride<_3>>{};
    Layout tile_shape = Layout<Shape<_4>, Stride<_1>>{};
    p(full_layout);                                 // (_16):(_3)
    p(tile_shape);                                  // (_4):(_1)
    p(logical_divide(full_layout, tile_shape));     // ((_4),_4):((_3),_12)
    p(zipped_divide(full_layout, tile_shape));
    // p(tiled_divide(full_layout, tile_shape));
}

void test_swizzle() {
    using namespace cute;
    auto a = Layout<Shape<_8,_8>, Stride<_8,_1>>{};
    cute::print_layout(a);
    cute::print_layout(composition(Swizzle<0,0,0>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{})); // identity
    cute::print_layout(composition(Swizzle<0,0,7>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{})); // identity
    cute::print_layout(composition(Swizzle<0,1,0>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{})); // identity
    cute::print_layout(composition(Swizzle<0,7,0>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{})); // identity
    cute::print_layout(composition(Swizzle<1,0,1>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<1,0,2>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<1,0,3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<1,1,1>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<1,2,1>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<2,0,3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<2,0,-3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<2,1,-3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
    cute::print_layout(composition(Swizzle<3,3,3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{}));
}

void __global__ test_debug_kernel() {
    if (cute::thread0()) {
        cute::print("thread=%d, block=%d\n", threadIdx.x, blockIdx.x); // thread=0, block=0
    }
    __syncthreads();
    if (cute::thread(1)) {
        cute::print("thread=%d, block=%d\n", threadIdx.x, blockIdx.x); // thread=1, block=0
    }
    __syncthreads();
    if (cute::thread(5, 1)) {
        cute::print("thread=%d, block=%d\n", threadIdx.x, blockIdx.x); // thread=5, block=1
    }
}

void test_debug() {
    test_debug_kernel<<<2, 32>>>();
    cudaDeviceSynchronize();
}

void __global__ test_tensor_kernel(void* ptr) {
    // global memory
    if (cute::thread0()) {
        cute::print("global memory\n");
        // tag memory
        auto gmem_ptr = cute::make_gmem_ptr(reinterpret_cast<float*>(ptr));
        cute::print(gmem_ptr);      // gmem_ptr_32b((nil))
        cute::print("\n");

        // static layout
        cute::Tensor gmem_8s = cute::make_tensor(gmem_ptr, cute::Int<8>{});
        cute::print(gmem_8s);       // _8:_1
        cute::print("\n");

        // dynamic layout
        cute::Tensor gmem_8d = cute::make_tensor(gmem_ptr, 8);
        cute::print(gmem_8d);       // 8:_1
        cute::print("\n");

        // mixed static & dynamic
        cute::Tensor gmem_8dx16s = make_tensor(gmem_ptr, cute::make_shape (      8    , cute::_16{}),
                                                         cute::make_stride(cute::_16{}, cute::_1{}));
        cute::print(gmem_8dx16s);           // (8,_16):(_16,_1) with tensor value
        cute::print(gmem_8dx16s.layout());  // (8,_16):(_16,_1)
        cute::print("\n");
    }
    __syncthreads();

    // shared memory
    if (cute::thread0()) {
        cute::print("shared memory\n");
        auto smem_shape = cute::make_shape(cute::_4{}, cute::_8{});
        __shared__ float smem[decltype(cute::size(smem_shape))::value];   // (static-only allocation)
        cute::Tensor smem_4x8_col = cute::make_tensor(cute::make_smem_ptr(&smem[0]), smem_shape);
        cute::Tensor smem_4x8_row = cute::make_tensor(cute::make_smem_ptr(&smem[0]), smem_shape, cute::GenRowMajor{});
        cute::print(smem_4x8_col);  // (_4,_8):(_1,_4)
        cute::print("\n");
        cute::print(smem_4x8_row);  // (_4,_8):(_8,_1)
        cute::print("\n");
    }
    __syncthreads();

    // register memory (static layouts only)
    if (cute::thread0()) {
        cute::print("register memory\n");
        cute::Tensor rmem_4x8_col = cute::make_tensor<float>(cute::make_shape(cute::_4{}, cute::_8{}));
        cute::Tensor rmem_4x8_row = cute::make_tensor<float>(cute::make_shape(cute::_4{}, cute::_8{}), cute::GenRowMajor{});
        cute::Tensor rmem_4x8_mix = cute::make_tensor<float>(cute::make_shape(cute::_4{}, cute::_8{}), cute::make_stride(cute::_2{}, cute::_32{}));
        cute::print(rmem_4x8_col);  // (_4,_8):(_1,_4)
        cute::print("\n");
        cute::print(rmem_4x8_row);  // (_4,_8):(_8,_1)
        cute::print("\n");
        cute::print(rmem_4x8_mix);  // (_4,_8):(_2,_32)
        cute::print("\n");

        // slice with _ (only create a view)
        auto gmem_ptr = cute::make_gmem_ptr(reinterpret_cast<float*>(ptr));
        cute::Tensor gmem_8dx16s = make_tensor(gmem_ptr, cute::make_shape (      8    , cute::_16{}),
                                                         cute::make_stride(cute::_16{}, cute::_1{}));
        cute::Tensor gmem_16s = gmem_8dx16s(1, cute::_); // slice
        cute::print(gmem_16s);      // (_16):(_1)
        cute::print("\n");

        // make_fragment_like (static src layouts only), inherit layout and dtype
        cute::Tensor rmem_16s = cute::make_fragment_like(gmem_16s);
        cute::print(rmem_16s);      // (_16):(_1)
        cute::print("\n");
        // if we only want reuse layout but not dtype, use make_tensor:
        cute::Tensor rmem_16h = cute::make_tensor<cutlass::half_t>(make_layout_like(gmem_16s.layout()));
        cute::print(rmem_16h);      // (_16):(_1)
        cute::print("\n");
    }

    // global => register copy
    cute::Tensor gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<float*>(ptr)),
                                                              cute::make_shape(cute::_8{}, 16));
    CUTE_STATIC_ASSERT_V(cute::rank(gmem) == cute::_2{});
    CUTE_STATIC_ASSERT_V(cute::is_static<decltype(cute::shape<0>(gmem))>{});
    cute::Tensor rmem_8 = cute::make_fragment_like(gmem(cute::_, 0)); // (_8):(_1)
    for (int t = 0; t < cute::size<1>(gmem); ++t) {
        // global => regster
        cute::copy(gmem(cute::_, t), rmem_8);
        // do_something(rmem_8)
    }
}

void test_tensor() {
    // CuTe's Tensor class represents a multidimensional array. 
    // The array's elements can live in any kind of memory, 
    // including global memory, shared memory, and register memory.

    //  access a Tensor's elements
    // operator(a, b, c);
    // operator(Coord({a, b, c}));
    // operator[Coord({a, b, c})];

    // One could summarize almost all CuTe use cases as follows:
    // - create Layouts,
    // - create Tensors with those Layouts, and
    // - invoke (either CuTe's, or custom) algorithms on those Tensors.

    // Users can "tag" the memory with its space
    // e.g., global or shared by calling make_gmem_ptr(g) when g is a pointer to 
    // global memory, or make_smem_ptr(s) when s is a pointer to shared memory.

    // Tagging memory makes it possible for CuTe's Tensor algorithms to use 
    // the fastest implementation for the specific kind of memory. It also avoids 
    // incorrect memory access.

    float* ptr;
    cudaMalloc(&ptr, sizeof(float) * 128);
    test_tensor_kernel<<<1,32>>>(ptr);
    cudaDeviceSynchronize();
}

void test_algorithm() {
    // common numerical algorithms performed on Tensors

    // 1. copy
    // The copy algorithm has two main overloads.
    // - copy(src, dst) : use default implementation 
    // - copy(copy_atom, src, dst) : use user provided copy impl: copy_atom

    // Either the default implementation or the implementation selected by a Copy_Atom
    // overload may use none or all available parallelism, and may have a variety of
    // synchronization semantics. The behavior depends on copy's parameter types.

    // users will need to perform the additional synchronization appropriate to that underlying
    // implementation before they may use the results of the copy algorithm

    // 2. copy_if
    // 3. gemm
    // 4. axpby
    // 5. fill
    // 6. clear
}

void __global__ test_mma_atom_kernel() {
    // template <class MMA_Atom,
    //         class AtomLayoutMNK   = Layout<Shape<_1,_1,_1>>,
    //         class ValLayoutMNK    = Layout<Shape<_1,_1,_1>>,
    //         class PermutationsMNK = Tile<Underscore,Underscore,Underscore>>
    // struct TiledMMA : MMA_Atom;

    // Here, the AtomLayoutMNK is the "thread" tiling of the atom -- how many replicates of this MMA atom do you want to tile across the logical MNK modes by distinct threads.
    // The ValLayoutMNK similarly specific the tiling across replicate values instead -- how many atoms is each thread going to issue as a part of this tiled MMA.


    using namespace cute;

    constexpr int kNWarps = 2;
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_S32S8S8S32_TN>;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,                      // MMA_atom:                                    16x8x16
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // MMA_atom * AtomLayoutMNK:                    
        Layout<Shape<_1,_2,_2>>>;           // MMA_atom * AtomLayoutMNK * ValLayoutMNK:     

    // dummy smem tensor
    __shared__ int8_t smem_buf[128*32];
    using BLK_M = _32;
    using BLK_N = _16;
    using BLK_K = _32;

    Tensor sA = make_tensor(make_smem_ptr(smem_buf), Layout<Shape<BLK_M, BLK_K>>{}); // (BLK_M, BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smem_buf), Layout<Shape<BLK_N, BLK_K>>{}); // (BLK_N, BLK_K)
    pp(sA.layout());
    pp(sB.layout());

    const int thread_idx = 0;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);                     // (MMA, N_MMA_M, N_MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);                     // (MMA, N_MMA_N, N_MMA_K)
    pp(tCrA.layout());
    pp(tCrB.layout());
}


void test_mma_atom() {
    // MMAs are architecture-specific. Different generations of GPU architectures introduce different sets of 
    // MMA instructions. However, CuTe features such as Layout makes it possible to expose MMAs for use in generic 
    // CUDA C++ code. We do this in two steps:

    // - We wrap each MMA's PTX instruction in an "Operation" struct.
    // - For each Operation struct, we define a "Traits" struct that defines all of the meta-information needed to use 
    //   the Operation.

    // CuTe supports MMA atoms that operate at a variety of hardware levels, including
    // - a single thread (e.g., fused multiply-add (FMA) instruction);
    // - a quadpair (Volta);
    // - a single warp (Ampere); and
    // - a warpgroup (Hopper).

    // example:
    test_mma_atom_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}

template<class Mshape, class NShape, class KShape,
         class TA, class AStride, class ABlockLayout, class AThreadLayout,
         class TB, class BStride, class BBlockLayout, class BThreadLayout,
         class TC, class CStride, class CBlockLayout, class CThreadLayout>
__global__ void __launch_bounds__(decltype(cute::size(CThreadLayout{}))::value)
gemm_kernel(const Mshape M, const NShape N, const KShape K,
            const TA* __restrict__ A, const AStride dA, const ABlockLayout sA, const AThreadLayout tA,
            const TB* __restrict__ B, const BStride dB, const BBlockLayout sB, const BThreadLayout tB,
                  TC* __restrict__ C, const CStride dC, const CBlockLayout sC, const CThreadLayout tC) {
    // check
    CUTE_STATIC_ASSERT(cute::is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(cute::is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(cute::is_static<CBlockLayout>::value);
    CUTE_STATIC_ASSERT(cute::is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(cute::is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(cute::is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT(cute::size(tA) == cute::size(tC));
    CUTE_STATIC_ASSERT(cute::size(tB) == cute::size(tC));

    CUTE_STATIC_ASSERT(cute::shape<0>(sA) == cute::shape<0>(sC));   // BLK_M
    CUTE_STATIC_ASSERT(cute::shape<0>(sB) == cute::shape<1>(sC));   // BLK_N
    CUTE_STATIC_ASSERT(cute::shape<1>(sA) == cute::shape<1>(sB));   // BLK_K

    if (cute::thread0()) {
        kp(sA); // (_128,_8):(_1,_128)
        kp(sB); // (_128,_8):(_1,_128)
        kp(sC); // (_128,_128):(_1,_128)
        kp(tA); // (_32,_8):(_1,_32)
        kp(tB); // (_32,_8):(_1,_32)
        kp(tC); // (_16,_16):(_1,_16)
    }

    // alloc shared memory buffers
    __shared__ TA smemA_ptr[decltype(cute::cosize(sA))::value]; // BLK_M * BLK_K
    __shared__ TB smemB_ptr[decltype(cute::cosize(sB))::value]; // BLK_N * BLK_K

    auto smemA = cute::make_tensor(cute::make_smem_ptr(smemA_ptr), sA);
    auto smemB = cute::make_tensor(cute::make_smem_ptr(smemB_ptr), sB);

    // Represent the full tensors
    auto gmemA = cute::make_tensor(cute::make_gmem_ptr(A), cute::make_layout(cute::make_shape(M, K)));
    auto gmemB = cute::make_tensor(cute::make_gmem_ptr(B), cute::make_layout(cute::make_shape(N, K)));
    auto gmemC = cute::make_tensor(cute::make_gmem_ptr(C), cute::make_layout(cute::make_shape(M, N)));

    if (cute::thread0()) {
        kp(smemA.layout()); // (_128,_8):(_1,_128)
        kp(smemB.layout()); // (_128,_8):(_1,_128)

        kp(gmemA.layout()); // (1024,1024):(_1,1024)
        kp(gmemB.layout()); // (1024,1024):(_1,1024)
        kp(gmemC.layout()); // (1024,1024):(_1,1024)
    }

    //
    // Get the corresponding tiles for this thread block
    //

    // auto blk_shape = cute::make_shape(cute::size<0>(smemA), cute::size<0>(smemB), cute::size<1>(smemB));    // (BLK_M, BLM_N, BLK_K)
    // auto blk_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);

    // (M, K) => (BLK_M, BLK_K, ceil_div(K, BLK_K))
    // cute::Tensor gA = cute::local_tile(gmemA, blk_shape, blk_coord, cute::Step<cute::_1, cute::Underscore, cute::_1>());
    cute::Tensor gA = cute::local_tile(gmemA, smemA.layout().shape(), cute::make_coord(blockIdx.x, cute::_));

    // (N, K) => (BLK_N, BLK_K, ceil_div(K, BLK_K))
    // cute::Tensor gB = cute::local_tile(gmemB, blk_shape, blk_coord, cute::Step<cute::Underscore, cute::_1, cute::_1>());
    cute::Tensor gB = cute::local_tile(gmemB, smemB.layout().shape(), cute::make_coord(blockIdx.y, cute::_));

    // (M, N) => (BLK_M, BLK_N)
    // cute::Tensor gC = cute::local_tile(gmemC, blk_shape, blk_coord, cute::Step<cute::_1, cute::_1, cute::Underscore>());
    cute::Tensor gC = cute::local_tile(gmemC, 
                               cute::make_shape(cute::size<0>(smemA), cute::size<0>(smemB)),
                               cute::make_coord(blockIdx.x, blockIdx.y));


    if (cute::thread0()) {
        kp(gA.layout());    // (_128,_8,128):(_1,1024,8192)
        kp(gB.layout());    // (_128,_8,128):(_1,1024,8192)
        kp(gC.layout());    // (_128,_128):(_1,1024)
    }

    // 
    // thread binding for copying: 
    //      partition A,B gmem and smem tile for each thread in thread layout A and B
    // 

    // global tile: (BLK_M, BLK_K, ceil_div(K, BLK_K)) => (THR_M, THR_K, ceil_div(K, BLK_K))
    //              (128, 8, 128) / (32, 8) => (4, 1, 128)
    cute::Tensor tAgA = cute::local_partition(gA, tA, threadIdx.x);
    // shm tile: (BLK_M, BLK_K) => (THR_M, THR_K)
    //           (128, 8) / (32, 8) => (4, 1)
    cute::Tensor tAsA = cute::local_partition(smemA, tA, threadIdx.x);

    // ditto for B
    cute::Tensor tBgB = cute::local_partition(gB, tB, threadIdx.x);
    cute::Tensor tBsB = cute::local_partition(smemB, tB, threadIdx.x);

    if (cute::thread0()) {
        kp(tAgA.layout());  // (_4,_1,128):(_32,_0,8192)
        kp(tAsA.layout());  // (_4,_1):(_32,_0)
        kp(tBgB.layout());  // (_4,_1,128):(_32,_0,8192)
        kp(tBsB.layout());  // (_4,_1):(_32,_0)
    }

    // 
    // thread binding for computing:
    //      partition A,B,C smem tile for each thread in thread layout C
    // 

    // Partition smemA (BLK_M, BLK_K) by the rows of tC:
    // - (BLK_M, BLK_K) => (THR_M, BLK_K)
    // - (128, 8) / (16, _) => (8, 8)
    cute::Tensor tCsA = cute::local_partition(smemA, tC, threadIdx.x, cute::Step<cute::_1, cute::Underscore>{});

    // Partition smemA (BLK_N, BLK_K) by the cols of tC:
    // - (BLK_N, BLK_K) => (THR_N,BLK_K)
    // - (128, 8) / (_, 16) => (8, 8)
    cute::Tensor tCsB = cute::local_partition(smemB, tC, threadIdx.x, cute::Step<cute::Underscore, cute::_1>{});

    // Partition gmemC (BLK_M, BLK_N) by the tile of tC:
    // - (BLK_M, BLK_N) => (THR_M, THR_N)
    // - (128, 128) / (16, 16) => (8, 8)
    // This is an exception, tCgC is for copying from regC to globalC
    cute::Tensor tCgC = cute::local_partition(gC, tC, threadIdx.x, cute::Step<cute::_1, cute::_1>{});

    // alloc register C for each thread for computing
    cute::Tensor tCrC = cute::make_fragment_like(tCgC);
    // set the accumulators to 0
    cute::clear(tCrC);

    if (cute::thread0()) {
        kp(tCsA.layout());  // (_8,_8):(_16,_128)
        kp(tCsB.layout());  // (_8,_8):(_16,_128)
        kp(tCgC.layout());  // (_8,_8):(_16,16384)
        kp(tCrC.layout());  // (_8,_8):(_1,_8)
    }

    // 
    // GEMM main loop
    // 

    auto k_iter = cute::size<2>(tAgA); // ceil_div(K, BLK_K)
    auto _ = cute::Underscore{};
    for (int k=0; k<k_iter; ++k) {
        // Copy gmem to smem
        cute::copy(tAgA(_,_,k), tAsA);
        cute::copy(tBgB(_,_,k), tBsB);

        // In case copy uses cp.async
        cute::cp_async_fence();     // cp.async.commit_group
        cute::cp_async_wait<0>();   // cp.async_wait_group 0

        __syncthreads();

        // Compute gemm on smem A & B and reg C
        cute::gemm(tCsA, tCsB, tCrC);
    }

    // epilogue: tCgC = alpha * tCrC + beta * tCgC => tCgC = tCrC
    cute::axpby((TC)1.0, tCrC, (TC)0.0, tCgC);

    // or just copy tCrC back to tCgC directly
    // cute::copy(tCrC, tCgC);
}

template<typename TA, typename TB, typename TC>
void gemm(TA* a_ptr, TB* b_ptr, TC* c_ptr, const int M, const int N, const int K) {
    // device tensor layout
    // - A: MxK (M-major)
    // - B: NxK (N-major)
    // - C: M*N (M-major)

    // Define global strides (mixed)
    auto dA = cute::make_stride(cute::_1{}, M);
    auto dB = cute::make_stride(cute::_1{}, N);
    auto dC = cute::make_stride(cute::_1{}, M);

    // Define block sizes (static)
    auto bM = cute::_128{};
    auto bN = cute::_128{};
    auto bK = cute::_8{};

    // Define the block layouts (static), use default stride
    auto sA = cute::make_layout(cute::make_shape(bM, bK));
    auto sB = cute::make_layout(cute::make_shape(bN, bK));
    auto sC = cute::make_layout(cute::make_shape(bM, bN));

    // Define the thread layouts (static), use 8 warp
    auto tA = cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{}));
    auto tB = cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{}));
    auto tC = cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}));

    dim3 dimGrid(cute::ceil_div(M, bM), cute::ceil_div(N, bN));
    dim3 dimBlock(cute::size(tC)); // 32 * 8 = 256

    cudaStream_t stream = 0;
    gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        M, N, K,
        a_ptr, dA, sA, tA,
        b_ptr, dB, sB, tB,
        c_ptr, dC, sC, tC
    );
}

void test_gemm_kernel() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    using TA = cutlass::half_t;
    using TB = cutlass::half_t;
    using TC = cutlass::half_t;

    size_t size_a = M * K * sizeof(TA);
    size_t size_b = K * N * sizeof(TB);
    size_t size_c = M * N * sizeof(TC);

    TA* h_a = (TA*) malloc(size_a);
    TB* h_b = (TB*) malloc(size_b);
    TC* h_c = (TC*) malloc(size_c);
    TC* h_c_ref = (TC*) malloc(size_c);

    TA* d_a;
    TB* d_b;
    TC* d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    srand(time(0));
    for (int i=0; i<M*K; ++i) {
        h_a[i] = (TA)(rand() / float(RAND_MAX));
    }
    for (int i=0; i<K*N; ++i) {
        h_b[i] = (TB)(rand() / float(RAND_MAX));
    }
    for (int i=0; i<M*N; ++i) {
        h_c[i] = (TC)(0);
    }
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CUTE_CHECK_LAST();

    // check error
    // for (int im=0; im<M; ++im) {
    //     for (int in=0; in<N; ++in) {
    //         TC acc = (TC)0.0;
    //         for (int ik=0; ik<K; ++ik) {
    //             acc += h_a[im + ik * M] * h_b[in + ik * K];
    //         }
    //         double rel_err = cute::abs((double)h_c[im + in * M] - (double)acc) / (double)acc;
    //         assert (rel_err < 0.03);
    //     }
    // }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template<typename TileShape, class TA, class TB, class TC,
        class TiledMma,
        class GmemTiledCopyA, class SmemLayoutAtomA, class SmemCopyAtomA,
        class GmemTiledCopyB, class SmemLayoutAtomB, class SmemCopyAtomB>
__global__ void gemm_kernel_with_atom(int M, int N, int K,
            const TA* __restrict__ A, const TB* __restrict__ B, TC* __restrict__ C) {
    using namespace cute;

    pp(TileShape{});        // (_128,_128,_64)
    pp(SmemLayoutAtomA{});  // S<2,4,3> o _0 o (_16,_64):(_64,_1)
    pp(SmemLayoutAtomB{});  // S<2,4,3> o _0 o (_16,_64):(_64,_1)

    // =====================
    // multi stage example
    // using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<3>;
    // using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
    //                                            make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}),
    //                                            Int<DispatchPolicy::Stages>{})));
    // pp(SmemLayoutA{});  // S<2,4,3> o _0 o (_128,_64,_3):(_64,_1,_8192)

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

    pp(SmemLayoutA{});  // S<2,4,3> o _0 o (_128,_64):(_64,_1) : (BLK_M, BLK_K)
    pp(SmemLayoutB{});  // S<2,4,3> o _0 o (_128,_64):(_64,_1) : (BLK_N, BLK_K)

    static_assert(rank(SmemLayoutA{}) == 2);
    static_assert(rank(SmemLayoutB{}) == 2);

    // alloc shared memory buffers
    __shared__ TA smemA_ptr[decltype(cute::cosize(SmemLayoutA{}))::value]; // BLK_M * BLK_K
    __shared__ TB smemB_ptr[decltype(cute::cosize(SmemLayoutB{}))::value]; // BLK_N * BLK_K

    Tensor sA = make_tensor(make_smem_ptr(smemA_ptr), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smemB_ptr), SmemLayoutB{});
    pp(sA.layout());    // S<2,4,3> o _0 o (_128,_64):(_64,_1) : (BLK_M, BLK_K)
    pp(sB.layout());    // S<2,4,3> o _0 o (_128,_64):(_64,_1) : (BLK_N, BLK_K)

    // Represent the full tensors
    auto gmemA = cute::make_tensor(cute::make_gmem_ptr(A), cute::make_layout(cute::make_shape(M, K), cute::GenRowMajor{}));
    auto gmemB = cute::make_tensor(cute::make_gmem_ptr(B), cute::make_layout(cute::make_shape(N, K), cute::GenRowMajor{}));
    auto gmemC = cute::make_tensor(cute::make_gmem_ptr(C), cute::make_layout(cute::make_shape(M, N), cute::GenRowMajor{}));

    pp(gmemA.layout()); // (1024,1024):(1024,_1)
    pp(gmemB.layout()); // (1024,1024):(1024,_1)
    pp(gmemC.layout()); // (1024,1024):(1024,_1)

    // Get the corresponding tiles for this thread block
    // (M, K) => (BLK_M, BLK_K, ceil_div(K, BLK_K))
    cute::Tensor gA = cute::local_tile(gmemA, sA.layout().shape(), cute::make_coord(blockIdx.x, cute::_));

    // (N, K) => (BLK_N, BLK_K, ceil_div(K, BLK_K))
    cute::Tensor gB = cute::local_tile(gmemB, sB.layout().shape(), cute::make_coord(blockIdx.y, cute::_));

    // (M, N) => (BLK_M, BLK_N)
    cute::Tensor gC = cute::local_tile(gmemC, 
                                       cute::make_shape(cute::size<0>(sA), cute::size<0>(sB)),
                                       cute::make_coord(blockIdx.x, blockIdx.y));

    pp(gA.layout());    // (_128,_64,16):(1024,_1,_64)
    pp(gB.layout());    // (_128,_64,16):(1024,_1,_64)
    pp(gC.layout());    // (_128,_128):(1024,_1)


    // threading binding for gmem=>smem copy using GmemTiledCopyA/B
    const int thread_idx = threadIdx.x;
    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto copy_a_thr = gmem_tiled_copy_a.get_slice(thread_idx);
    auto copy_b_thr = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = copy_a_thr.partition_S(gA);
    Tensor tAsA = copy_a_thr.partition_D(sA);
    Tensor tBgB = copy_b_thr.partition_S(gB);
    Tensor tBsB = copy_b_thr.partition_D(sB);
    pp(tAsA.layout());  // ((_16,_1),_4,_1):((_1,_0),_2048,_0)         : (ACPY, ACPY_M, ACPY_K)
    pp(tBsB.layout());  // ((_16,_1),_4,_1):((_1,_0),_2048,_0)         : (BCPY, BCPY_N, BCPY_K)
    pp(tAgA.layout());  // ((_16,_1),_4,_1,16):((_1,_0),32768,_0,_64)  : (ACPY, ACPY_M, ACPY_K, k_loop)
    pp(tBgB.layout());  // ((_16,_1),_4,_1,16):((_1,_0),32768,_0,_64)  : (BCPY, BCPY_N, BCPY_K, k_loop)

    // 
    // MMA compute
    // 
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    // allocate regA and regB for each thread
    Tensor tCrA  = thr_mma.partition_fragment_A(sA);
    Tensor tCrB  = thr_mma.partition_fragment_B(sB);
    pp(tCrA.layout());  // ((_4,_2,_2),_4,_2):((_1,_4,_8),_16,_64)  : (MMA_elem_per_thr, BLK_M/WP_M/MMA_M, BLK_K/WP_K/MMA_K)
    pp(tCrB.layout());  // ((_4,_2),_8,_2):((_1,_4),_8,_64)         : (MMA_elem_per_thr, BLK_N/WP_N/MMA_N, BLK_K/WP_K/MMA_K)

    // thread binding for gC
    Tensor tCgC = thr_mma.partition_C(gC);
    // allocate regC for each thread
    Tensor tCrC = thr_mma.partition_fragment_C(gC);
    pp(tCgC.layout());  // ((_2,_2),_4,_8):((_1,8192),32768,_16)
    pp(tCrC.layout());  // ((_2,_2),_4,_8):((_1,_2),_4,_16)
    cute::clear(tCrC);

    // threading binding for smem
    auto smem_tiled_copy_a = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto thr_copy_A        = smem_tiled_copy_a.get_thread_slice(thread_idx);
    Tensor tCsA            = thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = thr_copy_A.retile_D(tCrA);  // tCrA and tCrA_copy_view share same storage, 
                                                         // tCrA_copy_view is used for efficient smem=>rmem copy
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    pp(tCsA.layout());              // ((_16,_1),_4,_2):((_1,_0),_2048,32)  : (MMA_elem_per_thr, BLK_M/WP_M/MMA_M, BLK_K/WP_K/MMA_K)
    pp(tCrA_copy_view.layout());    // ((_16,_1),_4,_2):((_1,_0),_16,_64)   : (MMA_elem_per_thr, BLK_M/WP_M/MMA_M, BLK_K/WP_K/MMA_K)
    // tCrA and tCrA_copy_view share same storage
    // pp(tCrA(0));            // 0
    // pp(tCrA_copy_view(0));  // 0
    // tCrA(0) = 123;
    // pp(tCrA_copy_view(0));  // 123

    auto smem_tiled_copy_b = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto thr_copy_B        = smem_tiled_copy_b.get_thread_slice(thread_idx);
    Tensor tCsB            = thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = thr_copy_B.retile_D(tCrB);  // tCrB and tCrB_copy_view share same storage
                                                         // tCrB_copy_view is used for efficient smem=>rmem copy
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    pp(tCsB.layout());              // ((_16,_1),_4,_2):((_1,_0),_2048,32)  : (MMA_elem_per_thr, BLK_N/WP_N/MMA_N, BLK_K/WP_K/MMA_K)
    pp(tCrB_copy_view.layout());    // ((_16,_1),_4,_2):((_1,_0),_16,_64)   : (MMA_elem_per_thr, BLK_N/WP_N/MMA_N, BLK_K/WP_K/MMA_K)

    // 
    // GEMM main loop (outer k)
    // 
    auto k_loop = size<3>(tAgA); // ceil_div(K, BLK_K)
    for (int k=0; k<k_loop; ++k) {
        // copy gmem => smem (use LDGSTS(cp.async))
        copy(gmem_tiled_copy_a, tAgA(_,_,_,k), tAsA);
        cute::cp_async_fence();     // cp.async.commit_group
        copy(gmem_tiled_copy_b, tBgB(_,_,_,k), tBsB);
        cute::cp_async_fence();     // cp.async.commit_group
        cute::cp_async_wait<0>();   // cp.async_wait_group 0
        __syncthreads();

        // copy smem => rmem (use LDSM)
        copy(smem_tiled_copy_a, tCsA, tCrA_copy_view);
        copy(smem_tiled_copy_b, tCsB, tCrB_copy_view);

        __syncthreads();

        // tensor core mma
        cute::gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        // or use explicit GEMM inner loop (inner k)
        // auto k_inner_loop = size<2>(tCrA);  // BLK_K/WP_K/MMA_K
        // for (int inn_k=0; inn_k<k_inner_loop; ++inn_k) {
        //     // tensor core mma
        //     cute::gemm(tiled_mma, tCrC, tCrA(_,_,inn_k), tCrB(_,_,inn_k), tCrC);
        // }
    }

    // copy rmem=>gmem
    copy(tCrC, tCgC);
}

template<typename TA, typename TB, typename TC>
void gemm_with_atom(TA* a_ptr, TB* b_ptr, TC* c_ptr, const int M, const int N, const int K) {
    using namespace cute;
    // device tensor layout
    // - A: MxK (K-major)
    // - B: NxK (K-major)
    // - C: M*N (N-major)

    // Define block sizes (static)
    using bM = cute::_128;
    using bN = cute::_128;
    using bK = cute::_64;

    using TileShape = Shape<bM, bN, bK>;
    static constexpr int ThreadCount = 128;

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
        Layout<Shape<_2,_2,_1>>,   // 2x2x1 thread group (equals to #warp)
        Layout<Shape<_1,_2,_1>>>;  // 1x2x1 value group for 16x16x32 and LDSM (seems equals to 16x16x256bit?)

    // ===== for A (M,K)  K-major =====
    using SmemLayoutAtomA = decltype(
        composition(
        Swizzle<2,4,3>{},
        Layout<Shape <_16,_64>,
               Stride<_64, _1>>{}));

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    static constexpr int kAlignmentA = 16;
    // for gmemA=>smemA
    using GmemTiledCopyA = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                        Layout<Shape<_32,_4>, Stride<_4,_1>>{},     // thread layout, s.t. BLK_K % kAlignmentA * THR_K == 0
                        Layout<Shape<_1, Int<kAlignmentA>>>{}));    // value layout, s.t. kAlignmentA = sizeof(uint128_t) / sizeof(int8_t)

    // LDS.32- or LDSM-based copy atom
    // using SmemCopyAtomA = Copy_Atom<DefaultCopy, uint8_t>;
    // for smemA=>rmemA
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;   // LDSM works

    // ===== for B (N,K)  K-major =====
    using SmemLayoutAtomB = decltype(
    composition(
        Swizzle<2,4,3>{},
        Layout<Shape <_16,_64>,
                Stride<_64, _1>>{}));

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    static constexpr int kAlignmentB = 16;
    using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                    Layout<Shape<_32,_4>, Stride< _4,_1>>{},    // thread layout, s.t. BLK_K % kAlignmentB * THR_K == 0
                    Layout<Shape<_1,Int<kAlignmentB>>>{}));     // value layout, s.t. kAlignmentB = sizeof(uint128_t) / sizeof(int8_t)

    // LDS.32- or LDSM-based copy atom
    // using SmemCopyAtomB = Copy_Atom<DefaultCopy, uint32_t>;
    using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;  // LDSM works

    dim3 dimGrid(cute::ceil_div(M, bM{}), cute::ceil_div(N, bN{}));
    dim3 dimBlock(ThreadCount);

    cudaStream_t stream = 0;
    gemm_kernel_with_atom<
        TileShape, TA, TB, TC,
        TiledMma,
        GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA,
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB
    ><<<dimGrid, dimBlock, 0, stream>>>(
        M, N, K,
        a_ptr, b_ptr, c_ptr
    );
}

void test_gemm_with_atom_kernel() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    using TA = int8_t;
    using TB = int8_t;
    using TC = int32_t;

    size_t size_a = M * K * sizeof(TA);
    size_t size_b = K * N * sizeof(TB);
    size_t size_c = M * N * sizeof(TC);

    TA* h_a = (TA*) malloc(size_a);
    TB* h_b = (TB*) malloc(size_b);
    TC* h_c = (TC*) malloc(size_c);
    TC* h_c_ref = (TC*) malloc(size_c);

    TA* d_a;
    TB* d_b;
    TC* d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    srand(time(0));
    for (int i=0; i<M*K; ++i) {
        h_a[i] = (TA)(rand() / 256 - 128);
    }
    for (int i=0; i<K*N; ++i) {
        h_b[i] = (TB)(rand() / 256 - 128);
    }
    for (int i=0; i<M*N; ++i) {
        h_c[i] = (TC)(0);
    }
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gemm_with_atom(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CUTE_CHECK_LAST();

    // check error
    // for (int im=0; im<M; ++im) {
    //     for (int in=0; in<N; ++in) {
    //         TC acc = (TC)0.0;
    //         for (int ik=0; ik<K; ++ik) {
    //             acc += h_a[im * K + ik] * h_b[in * K + ik];
    //         }
    //         double rel_err = cute::abs((double)h_c[im * N + in] - (double)acc) / (double)acc;
    //         if ((im * N + in) % (M * N / 10) == 0) {
    //             printf("gt=%d, out=%d, at (%d,%d)\n", acc, h_c[im * N + in], im, in);
    //         }
    //         if (rel_err > 0.01) {
    //             printf("error occured: with gt=%d, out=%d, at (%d,%d)\n", acc, h_c[im * N + in], im, in);
    //             exit(1);
    //         }
    //     }
    // }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template<class BlockLayout, class ThreadLayout>
__global__ void __launch_bounds__(decltype(cute::size(ThreadLayout{}))::value)
predication_kernel(const BlockLayout sA, const ThreadLayout tA) {
    // 1. make_identity_tensor:
    // - make_identity_tensor returns a "fake" tensor that when query with coord (m, n) it returns (m, n)
    cute::Tensor cA = cute::make_identity_tensor(cute::make_shape(cute::size<0>(sA), cute::size<1>(sA)));
    if (cute::thread0()) {
        kp(cA.layout());    // (_128,_8):(0:_1,1:_1)
        kp(cA(0,1));        // (0,1)
        kp(cA(2,6));        // (2,6)
        kp(cA(125,0));      // (125,0)
        kp(cA(12,7));       // (12,7)
    }
    // - this coord tensor can keep its original coord after tiling, thus can be used to trace the
    //   original coord of a tensor.
    cute::Tensor tAcA = cute::local_partition(cA, tA, threadIdx.x);
    if (cute::thread0()) {
        kp(tAcA.layout());  // (_4,_1):(0:_32,_0)
        kp(tAcA(0, 0));     // (0,0)
        kp(tAcA(1, 0));     // (32,0)
        kp(tAcA(3, 0));     // (96,0)
    }

    // - and we can use it to create a predicate tensor by comparing the coord tensor's value with
    //   the bounds of the original layout. for example:

    // predicate tensor:
    // cuye::Tensor tApA = cute::make_tensor<bool>(cute::make_shape(...), cute::make_stride(...);

    // Populate:
    // CUTE_UNROLL
    // for (int m = 0; m < cute::size<0>(tApA); ++m) {
    //     tApA(m,0) = cute::get<0>(tAcA(m,0)) < m_max_coord;
    // }

    // - We can then use the predicate tensors in copy_if to copy only the elements for which the
    //   corresponding predicate tensor elements are nonzero:

    // copy_if(tApA, tAgA(...), tAsA(...));

}

void test_predication() {
    // predication is used when tiling isn't perfect
    // The general procedure is that we:
    // - create an "identity" layout with the same shape as our original data;
    // - repeat the same tiling/partitioning/slicing (possibly rounding up) on that identity layout;
    // - create a "predicate tensor" by comparing the coordinates of that reference layout with the bounds of the original layout;
    // - use the predicate tensor to mask off accesses to out-of-bounds elements.

    auto bM = cute::_128{};
    auto bK = cute::_8{};
    auto sA = cute::make_layout(cute::make_shape(bM, bK));
    auto tA = cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{}));
    dim3 dimBlock(cute::size(tA)); // 32 * 8 = 256
    predication_kernel<<<1, dimBlock>>>(sA, tA);
    cudaDeviceSynchronize();
    CUTE_CHECK_LAST();
}

int main() {
    test_int_tuple();
    test_layout();
    test_layout_opeartion();
    test_swizzle();
    test_debug();
    test_tensor();
    test_algorithm();
    test_mma_atom();
    test_gemm_kernel();
    test_gemm_with_atom_kernel();
    test_predication();
}
