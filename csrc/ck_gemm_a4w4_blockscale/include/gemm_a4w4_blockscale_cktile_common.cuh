// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_mx.hpp"

using TILE_FP32   = float;
using TILE_FP16   = ck_tile::half_t;
using TILE_BF16   = ck_tile::bf16_t;
using TILE_F4PK   = ck_tile::pk_fp4_t;
using TILE_E8M0PK = int32_t;

using ABDataType      = TILE_F4PK;
using XPackedDataType = TILE_E8M0PK;

using ALayout  = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout  = ck_tile::tensor_layout::gemm::ColumnMajor;
using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout  = ck_tile::tensor_layout::gemm::RowMajor;
using DLayout  = ck_tile::tensor_layout::gemm::RowMajor;

template <ck_tile::index_t M_Tile,
          ck_tile::index_t N_Tile,
          ck_tile::index_t K_Tile,
          ck_tile::index_t M_Warp,
          ck_tile::index_t N_Warp,
          ck_tile::index_t K_Warp,
          ck_tile::index_t M_Warp_Tile,
          ck_tile::index_t N_Warp_Tile,
          ck_tile::index_t K_Warp_Tile,
          bool TiledMMAPermuteN                    = false,
          bool TransposeC                          = false,
          bool UsePersistentKernel                 = false,
          ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave,
          int BlockPerCu                           = 1,
          bool AQRowMajor                          = false>
struct CreateTileGemmConfig
{
    static constexpr ck_tile::index_t M_Tile_v                  = M_Tile;
    static constexpr ck_tile::index_t N_Tile_v                  = N_Tile;
    static constexpr ck_tile::index_t K_Tile_v                  = K_Tile;
    static constexpr ck_tile::index_t M_Warp_v                  = M_Warp;
    static constexpr ck_tile::index_t N_Warp_v                  = N_Warp;
    static constexpr ck_tile::index_t K_Warp_v                  = K_Warp;
    static constexpr ck_tile::index_t M_Warp_Tile_v             = M_Warp_Tile;
    static constexpr ck_tile::index_t N_Warp_Tile_v             = N_Warp_Tile;
    static constexpr ck_tile::index_t K_Warp_Tile_v             = K_Warp_Tile;
    static constexpr bool TiledMMAPermuteN_v                    = TiledMMAPermuteN;
    static constexpr bool TransposeC_v                          = TransposeC;
    static constexpr bool UsePersistentKernel_v                 = UsePersistentKernel;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler_v = Scheduler;
    static constexpr int BlockPerCu_v                           = BlockPerCu;
    static constexpr bool AQRowMajor_v                          = AQRowMajor;
};

template <ck_tile::index_t M_Tile,
          ck_tile::index_t N_Tile,
          ck_tile::index_t K_Tile,
          ck_tile::index_t M_Warp,
          ck_tile::index_t N_Warp,
          ck_tile::index_t K_Warp,
          ck_tile::index_t M_Warp_Tile,
          ck_tile::index_t N_Warp_Tile,
          ck_tile::index_t K_Warp_Tile,
          bool TiledMMAPermuteN                    = false,
          bool TransposeC                          = false,
          bool UsePersistentKernel                 = false,
          ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave,
          int BlockPerCu                           = 1,
          bool AQRowMajor                          = false>
using TileGemmConfig = CreateTileGemmConfig<M_Tile,
                                            N_Tile,
                                            K_Tile,
                                            M_Warp,
                                            N_Warp,
                                            K_Warp,
                                            M_Warp_Tile,
                                            N_Warp_Tile,
                                            K_Warp_Tile,
                                            TiledMMAPermuteN,
                                            TransposeC,
                                            UsePersistentKernel,
                                            Scheduler,
                                            BlockPerCu,
                                            AQRowMajor>;

template <typename EDataType, bool HasBias>
struct EpilogueTraits;

template <typename EDataType>
struct EpilogueTraits<EDataType, true>
{
    using ElementwiseOp    = ck_tile::element_wise::MultiDAdd;
    using DsLayout         = ck_tile::tuple<DLayout>;
    using DsDataType       = ck_tile::tuple<EDataType>;
    using CShuffleDataType = TILE_FP32;
};

template <typename EDataType>
struct EpilogueTraits<EDataType, false>
{
    using ElementwiseOp    = ck_tile::element_wise::PassThrough;
    using DsLayout         = ck_tile::tuple<>;
    using DsDataType       = ck_tile::tuple<>;
    using CShuffleDataType = EDataType;
};

// Needed only for passing AQDataType and BQDataType to QuantGemmKernel
template <typename Problem, typename Policy = ck_tile::UniversalGemmPipelineAgBgCrPolicy>
struct QuantGemmPipelineAgBgCrCompV3 : public ck_tile::GemmPipelineAgBgCrCompV3<Problem, Policy>
{
    using AQDataType = ck_tile::remove_cvref_t<typename Problem::AQDataType>;
    using BQDataType = ck_tile::remove_cvref_t<typename Problem::BQDataType>;
};

template <typename CDataType,
          typename GemmConfig,
          typename HostArguments,
          bool HasBias,
          bool PreshuffleB,
          bool UseDoubleSmemBuffer = PreshuffleB>
void TileGemmComputeImpl(const HostArguments& args)
{
    using ComputeDataType = ABDataType;
    using AccDataType     = TILE_FP32;

    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr bool TransposeC            = false;
    constexpr bool UseStructuredSparsity = false;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile_v, GemmConfig::N_Tile_v, GemmConfig::K_Tile_v>,
        ck_tile::sequence<GemmConfig::M_Warp_v, GemmConfig::N_Warp_v, GemmConfig::K_Warp_v>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile_v,
                          GemmConfig::N_Warp_Tile_v,
                          GemmConfig::K_Warp_Tile_v>>;

    // using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                        kPadN,
                                                        kPadK,
                                                        UseDoubleSmemBuffer,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        TransposeC,
                                                        UseStructuredSparsity,
                                                        false, // persistent,
                                                        GemmConfig::NumWaveGroups,
                                                        PreshuffleB>;

    using PipelineProblem = ck_tile::UniversalGemmPipelineProblem<ABDataType,
                                                                  ABDataType,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  GemmTraits,
                                                                  GemmConfig::Scheduler_v>;

    using GemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<PipelineProblem>;

    static_assert(!GemmConfig::TiledMMAPermuteN_v,
                    "TiledMMAPermuteN=true requires PermuteNEpilogue, not CShuffleEpilogue");
    using EpTraits     = EpilogueTraits<CDataType, HasBias>;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ABDataType,
                                            ABDataType,
                                            typename EpTraits::DsDataType,
                                            AccDataType,
                                            CDataType,
                                            typename EpTraits::DsLayout,
                                            CLayout,
                                            typename EpTraits::ElementwiseOp,
                                            TilePartitioner::MPerBlock,
                                            TilePartitioner::NPerBlock,
                                            GemmConfig::M_Warp_v,
                                            GemmConfig::N_Warp_v * GemmConfig::K_Warp_v,
                                            GemmConfig::M_Warp_Tile_v,
                                            GemmConfig::N_Warp_Tile_v,
                                            GemmConfig::K_Warp_Tile_v,
                                            PipelineProblem::TransposeC,
                                            1,
                                            false,
                                            1,
                                            1,
                                            false,
                                            void,
                                            void,
                                            false,
                                            TILE_FP32,
                                            typename EpTraits::CShuffleDataType>>;

    using Kernel = ck_tile::MXGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(args);

    const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    ck_tile::launch_kernel(
        ck_tile::stream_config{at::hip::getCurrentHIPStream() /*stream_id*/,
                                false /*time_kernel*/,
                                1 /*log_level*/},
        ck_tile::make_kernel<GemmConfig::BlockPerCu_v>(Kernel{}, grids, blocks, 0, kargs));

}

template <typename CDataType,
          bool HasBias,
          typename GemmInstance>
__forceinline__ torch::Tensor gemm_a4w4_blockscale_cktile_impl(torch::Tensor& XQ,
                                                               torch::Tensor& WQ,
                                                               torch::Tensor& x_scale,
                                                               torch::Tensor& w_scale,
                                                               torch::Tensor& Y,
                                                               std::optional<torch::Tensor> bias,
                                                               int k_batch = 1)
{
    // check
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    TORCH_CHECK(XQ.stride(-1) == 1,
                "CKTile blockscale GEMM: XQ inner dim must be contiguous, "
                "got strides=[",
                XQ.stride(0),
                ",",
                XQ.stride(1),
                "]");
    TORCH_CHECK(WQ.stride(-1) == 1,
                "CKTile blockscale GEMM: WQ inner dim must be contiguous, "
                "got strides=[",
                WQ.stride(0),
                ",",
                WQ.stride(1),
                "]");
    TORCH_CHECK(Y.stride(-1) == 1,
                "CKTile blockscale GEMM: Y inner dim must be contiguous, "
                "got strides=[",
                Y.stride(0),
                ",",
                Y.stride(1),
                "]");

    // Split-K uses atomic_add into C; zero the output buffer first.
    // Use zero_() so all rows are cleared regardless of the leading-dimension
    // stride (e.g. padded tensors produced by vLLM's _maybe_pad_fp8_weight).
    if(k_batch > 1)
    {
        Y.zero_();
    }

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1) * 2; // always fp4_x2

    int StrideA = XQ.stride(-2) * 2; // always fp4_x2
    int StrideB = WQ.stride(-2) * 2; // always fp4_x2
    int StrideC = Y.stride(-2);
    int Scale_Stride_A = x_scale.stride(-2);
    int Scale_Stride_B = w_scale.stride(-2);

    using ScaleM = ck_tile::MXScalePointer<TILE_E8M0PK, 1, 32>;
    using ScaleN = ck_tile::MXScalePointer<TILE_E8M0PK, 1, 32>;

    auto runWithBias = [&]() {
        using HostArgs = ck_tile::MxGemmHostArgs<1, 1, 1>;

        HostArgs args(
            std::array<const void*, 1>{XQ.data_ptr()},
            std::array<const void*, 1>{x_scale.data_ptr()},
            std::array<const void*, 1>{WQ.data_ptr()},
            std::array<const void*, 1>{w_scale.data_ptr()},
            std::array<const void*, 1>{bias.value().data_ptr()},
            Y.data_ptr(),
            k_batch,
            M,
            N,
            K,
            std::array<int, 1>{StrideA},
            std::array<int, 1>{StrideB},
            std::array<int, 1>{static_cast<int>(bias.value().stride(-2))},
            StrideC
        );

        TileGemmComputeImpl<CDataType,
                            GemmInstance,
                            HostArgs,
                            true,
                            false,
                            false>(args);
    };

    auto runWithoutBias = [&]() {
        using HostArgs = ck_tile::MxGemmHostArgs<1, 1, 0>;

        HostArgs args(
            std::array<const void*, 1>{XQ.data_ptr()},
            std::array<const void*, 1>{x_scale.data_ptr()},
            std::array<const void*, 1>{WQ.data_ptr()},
            std::array<const void*, 1>{w_scale.data_ptr()},
            std::array<const void*, 0>{},
            Y.data_ptr(),
            k_batch,
            M,
            N,
            K,
            std::array<int, 1>{StrideA},
            std::array<int, 1>{StrideB},
            std::array<int, 0>{},
            StrideC
        );

        TileGemmComputeImpl<CDataType,
                            GemmInstance,
                            HostArgs,
                            true,
                            false,
                            false>(args);
    };

    if constexpr(HasBias)
    {
        if(bias != std::nullopt)
        {
            runWithBias();
        }
        else
        {
            runWithoutBias();
        }
    }
    else
    {
        runWithoutBias();
    }

    return Y;
}

#endif // USE_ROCM
