// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"

//------------------------------------------------------------------------
// Forward declarations.

class LudusGLState;
class LudusTimestampedState;
struct LudusCudaState;

//------------------------------------------------------------------------
// Python Ludus GL state wrapper (mesh shader-based f-theta rendering).

class LudusGLStateWrapper
{
public:
    LudusGLStateWrapper         (int cudaDeviceIdx);
    ~LudusGLStateWrapper        (void);

    void setContext             (void);
    void releaseContext         (void);
    void setMsaaSamples         (int samples);

    LudusGLState*               pState;
    bool                        automatic;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
// Python Ludus Timestamped state wrapper (GPU-native temporal rendering).

class LudusTimestampedStateWrapper
{
public:
    LudusTimestampedStateWrapper    (int cudaDeviceIdx);
    ~LudusTimestampedStateWrapper   (void);

    void setContext             (void);
    void releaseContext         (void);

    // Scene management
    int uploadScene             (torch::Tensor scene_desc,
                                 torch::Tensor polyline_pools,
                                 torch::Tensor polygon_pools,
                                 torch::Tensor obstacle_pools,
                                 int max_obstacles_in_pool,
                                 torch::Tensor timestamps,
                                 torch::Tensor int32_data,
                                 torch::Tensor vertices,
                                 torch::Tensor triangles,
                                 torch::Tensor poses,
                                 torch::Tensor float_data);
    void uploadCameras          (torch::Tensor intrinsics);
    void uploadColorPalette     (torch::Tensor colors);  // [num_prim_types, 4] RGBA colors
    void removeScene            (int sceneId);
    void clearScenes            (void);
    void preallocateBuffers     (int maxScenes, int bytesPerScene);
    int  uploadScenesBatch      (torch::Tensor scene_descs,
                                 torch::Tensor polyline_pools,
                                 torch::Tensor polygon_pools,
                                 torch::Tensor obstacle_pools,
                                 torch::Tensor bounds,
                                 torch::Tensor timestamps,
                                 torch::Tensor int32_data,
                                 torch::Tensor vertices,
                                 torch::Tensor triangles,
                                 torch::Tensor poses,
                                 torch::Tensor float_data);
    void setTessellationThreshold(float threshold);
    void setMaxTessellationLevels(int polyline, int polygon, int cube);
    void setLineWidths          (float polyline_regular, float polyline_bev,
                                 float ego_traj_regular, float ego_traj_bev,
                                 float wireframe);
    void setResolutionScale     (float scale);
    void setDepthScaling        (float enabled);
    void setCullRadius          (float radius);
    void setMsaaSamples         (int samples);
    int  getMaxBatchSize        (void);
    void swapBufferSets         (void);

    LudusTimestampedState*      pState;
    bool                        automatic;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
// Python Ludus CUDA state wrapper (fully CUDA-based rendering, no OpenGL).

class LudusCudaStateWrapper
{
public:
    LudusCudaStateWrapper       (int cudaDeviceIdx);
    ~LudusCudaStateWrapper      (void);

    void setLineWidths          (float polyline_regular, float polyline_bev,
                                 float ego_traj_regular, float ego_traj_bev,
                                 float wireframe);
    void setResolutionScale     (float scale);
    void setDepthScaling        (float enabled);
    void setCullRadius          (float radius);
    void setMaxTessellationLevels(int polyline, int polygon, int cube);
    void uploadColorPalette     (torch::Tensor colors);
    void setMsaaSamples         (int samples);

    LudusCudaState*             pState;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
