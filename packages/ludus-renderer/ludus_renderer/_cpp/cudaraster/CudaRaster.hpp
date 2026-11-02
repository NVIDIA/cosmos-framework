// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// This is a slimmed-down and modernized version of the original
// CudaRaster codebase that accompanied the HPG 2011 paper
// "High-Performance Software Rasterization on GPUs" by Laine and Karras.
// Modifications have been made to accommodate post-Volta execution model
// with warp divergence. Support for shading, blending, quad rendering,
// and supersampling have been removed as unnecessary for nvdiffrast.
//------------------------------------------------------------------------

// TODO: @ashwinvk Move this such that is publicly linked with
// to https://github.com/NVlabs/nvdiffrast and license should be Apache

namespace CR
{

class RasterImpl;

//------------------------------------------------------------------------
// Interface class to isolate user from implementation details.
//------------------------------------------------------------------------

class CudaRaster
{
public:
    enum
    {
        RenderModeFlag_EnableBackfaceCulling = 1 << 0,   // Enable backface culling.
    };

public:
					        CudaRaster				(void);
					        ~CudaRaster				(void);

    void                    setViewportSize         (int width, int height, int numImages);              // Width and height must be multiples of tile size (8x8).
    void                    setRenderModeFlags      (unsigned int renderModeFlags);                      // Affects all subsequent calls to drawTriangles(). Defaults to zero.
    void                    deferredClear           (unsigned int clearColor);                           // Clears color and depth buffers during next call to drawTriangles().
    void                    setVertexBuffer         (void* vertices, int numVertices);                   // GPU pointer managed by caller. Vertex positions in clip space as float4 (x, y, z, w).
    void                    setIndexBuffer          (void* indices, int numTriangles);                   // GPU pointer managed by caller. Triangle index+color quadruplets as uint4 (idx0, idx1, idx2, color).
    void                    setTiebreakerColorBuffer(void* colors);                                      // GPU pointer managed by caller. Per-vertex packed RGBA8 colors for deterministic z-fight tiebreaker.
    void                    setDeterministicTiebreaker(bool enable);                                     // When true, uses vertex colors as deterministic z-fight tiebreaker instead of triIdx.
    bool                    drawTriangles           (const int* ranges, bool peel, cudaStream_t stream); // Ranges (offsets and counts) as #triangles entries, not as bytes. If NULL, draw all triangles. Returns false in case of internal overflow.
    void*                   getColorBuffer          (void);                                              // GPU pointer managed by CudaRaster.
    void*                   getDepthBuffer          (void);                                              // GPU pointer managed by CudaRaster.                                           // Swap depth and peeling buffers.

private:
					        CudaRaster           	(const CudaRaster&); // forbidden
	CudaRaster&             operator=           	(const CudaRaster&); // forbidden

private:
    RasterImpl*             m_impl;                 // Opaque pointer to implementation.
};

//------------------------------------------------------------------------
} // namespace CR
