// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_SLOW_GROUPS = 1536;
// SMEM = descriptors only: 1536 * 5 * 4 = 30 KB → 7 blocks/SM on H100, 3 on Ada

__device__ __forceinline__ int bit_length_dev(int v) {
    int n = 0;
    while (v > 0) { n++; v >>= 1; }
    return n;
}

__device__ __forceinline__ int read_varint_dev(
    const uint8_t* buf, int pos, int end, int& out_pos
) {
    int result = 0, shift = 0;
    while (pos < end) {
        uint8_t b = buf[pos++];
        result |= (int)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    out_pos = pos;
    return result;
}

__device__ __forceinline__ int unpack_one(
    const uint8_t* src, int packed_start, int local_idx, int bit_width
) {
    int bit_pos = local_idx * bit_width;
    int byte_idx = packed_start + (bit_pos >> 3);
    int bit_off = bit_pos & 7;
    uint32_t val = (uint32_t)src[byte_idx]
                 | ((uint32_t)src[byte_idx + 1] << 8)
                 | ((uint32_t)src[byte_idx + 2] << 16)
                 | ((uint32_t)src[byte_idx + 3] << 24);
    return (int)((val >> bit_off) & ((1u << bit_width) - 1u));
}

// One block per RLE stream (= n_pages * 3 blocks).
// All reads from global memory (L1/L2 cached). SMEM used only for
// slow-path group descriptors (~30 KB) → high occupancy.
__global__ void decode_rle_streams_kernel(
    const uint8_t* __restrict__ data,      // concat decompressed pages + 4B pad
    const int* __restrict__ page_offset,   // byte offset per page in data
    const int* __restrict__ page_length,   // byte length per page
    const int* __restrict__ page_max_rep,
    const int* __restrict__ page_max_def,
    const int* __restrict__ page_num_values,
    const int* __restrict__ page_out_start, // output pos prefix-sum per page
    int total_values,
    int* __restrict__ output,              // [3 * total_values]
    int n_pages
) {
    int block_id = blockIdx.x;
    int page_id = block_id / 3;
    int stream_type = block_id - page_id * 3; // 0=rep, 1=def, 2=idx
    if (page_id >= n_pages) return;

    int tid = threadIdx.x;
    int mr = page_max_rep[page_id];
    int md = page_max_def[page_id];
    int nv = page_num_values[page_id];

    if (nv == 0) return;
    if (stream_type == 0 && mr == 0) return;
    if (stream_type == 1 && md == 0) return;

    int p_off = page_offset[page_id];
    int p_len = page_length[page_id];

    // --- Parse page header to find this stream's byte range ---
    int cursor = p_off;
    int stream_start = 0, stream_len = 0, bw = 0;

    if (mr > 0) {
        int rep_len = (int)data[cursor]
                    | ((int)data[cursor + 1] << 8)
                    | ((int)data[cursor + 2] << 16)
                    | ((int)data[cursor + 3] << 24);
        cursor += 4;
        if (stream_type == 0) {
            stream_start = cursor;
            stream_len = rep_len;
            bw = bit_length_dev(mr);
        }
        cursor += rep_len;
    }

    if (md > 0) {
        int def_len = (int)data[cursor]
                    | ((int)data[cursor + 1] << 8)
                    | ((int)data[cursor + 2] << 16)
                    | ((int)data[cursor + 3] << 24);
        cursor += 4;
        if (stream_type == 1) {
            stream_start = cursor;
            stream_len = def_len;
            bw = bit_length_dev(md);
        }
        cursor += def_len;
    }

    if (stream_type == 2) {
        bw = (int)data[cursor];
        cursor += 1;
        stream_start = cursor;
        stream_len = (p_off + p_len) - cursor;
    }

    if (stream_len == 0 || bw == 0) return;

    int out_base = stream_type * total_values + page_out_start[page_id];

    // --- Speculative all-BP validation (reads from L1/L2 cached gmem) ---
    int bp_data_bytes = (8 * bw + 7) / 8;
    int bp_group_bytes = 1 + bp_data_bytes;
    int expected_groups = stream_len / bp_group_bytes;
    if (expected_groups * 8 > nv) expected_groups = (nv + 7) / 8;

    int my_valid = 1;
    for (int g = tid; g < expected_groups; g += BLOCK_SIZE) {
        int hdr_pos = stream_start + g * bp_group_bytes;
        if (g * bp_group_bytes >= stream_len || data[hdr_pos] != 0x03) {
            my_valid = 0; break;
        }
    }

    unsigned warp_vote = __ballot_sync(0xFFFFFFFF, my_valid);
    __shared__ int block_all_valid;
    if (tid == 0) block_all_valid = 1;
    __syncthreads();
    if (warp_vote != 0xFFFFFFFF) atomicAnd(&block_all_valid, 0);
    __syncthreads();

    if (block_all_valid) {
        // ======== FAST PATH: all bit-packed, read from gmem (L1 cached) ========
        for (int g = tid; g < expected_groups; g += BLOCK_SIZE) {
            int packed = stream_start + g * bp_group_bytes + 1;
            int o = out_base + g * 8;
            int cnt = min(8, nv - g * 8);
            for (int v = 0; v < cnt; v++)
                output[o + v] = unpack_one(data, packed, v, bw);
        }
        return;
    }

    // ======== SLOW PATH: has RLE groups (batched) ========
    // SMEM used for group descriptors (thread 0 writes, all threads read).
    // Processes groups in batches of MAX_SLOW_GROUPS to handle arbitrarily
    // large streams without truncation.
    extern __shared__ uint8_t smem_raw[];
    int* desc_byte_off  = (int*)smem_raw;
    int* desc_out_start = desc_byte_off + MAX_SLOW_GROUPS;
    int* desc_count     = desc_out_start + MAX_SLOW_GROUPS;
    int* desc_type      = desc_count + MAX_SLOW_GROUPS;
    int* desc_rle_val   = desc_type + MAX_SLOW_GROUPS;

    __shared__ int n_groups_sh;
    __shared__ int s_rpos;
    __shared__ int s_idx;

    if (tid == 0) {
        s_rpos = 0;
        s_idx = 0;
    }
    __syncthreads();

    while (true) {
        if (tid == 0) {
            const uint8_t* src = data + stream_start;
            int slen = stream_len;
            int rpos = s_rpos, idx = s_idx, ng = 0;

            while (rpos < slen && idx < nv && ng < MAX_SLOW_GROUPS) {
                int next_pos;
                int header = read_varint_dev(src, rpos, slen, next_pos);
                rpos = next_pos;

                if (header & 1) {
                    int count = (header >> 1) * 8;
                    int byte_count = (count * bw + 7) / 8;
                    int actual = min(count, nv - idx);

                    desc_byte_off[ng] = stream_start + rpos;
                    desc_out_start[ng] = out_base + idx;
                    desc_count[ng] = actual;
                    desc_type[ng] = 1;

                    rpos += byte_count;
                    idx += actual;
                    ng++;
                } else {
                    int count = header >> 1;
                    int val_bytes = (bw + 7) / 8;
                    int val = 0;
                    for (int vi = 0; vi < val_bytes; vi++)
                        val |= (int)src[rpos + vi] << (vi * 8);
                    rpos += val_bytes;

                    int actual = min(count, nv - idx);
                    desc_byte_off[ng] = 0;
                    desc_out_start[ng] = out_base + idx;
                    desc_count[ng] = actual;
                    desc_type[ng] = 0;
                    desc_rle_val[ng] = val;

                    idx += actual;
                    ng++;
                }
            }
            n_groups_sh = ng;
            s_rpos = rpos;
            s_idx = idx;
        }
        __syncthreads();

        int ng = n_groups_sh;
        if (ng == 0) break;

        for (int g = tid; g < ng; g += BLOCK_SIZE) {
            int o = desc_out_start[g];
            int cnt = desc_count[g];

            if (desc_type[g] == 1) {
                int ps = desc_byte_off[g];
                for (int v = 0; v < cnt; v++)
                    output[o + v] = unpack_one(data, ps, v, bw);
            } else {
                int val = desc_rle_val[g];
                for (int v = 0; v < cnt; v++)
                    output[o + v] = val;
            }
        }
        __syncthreads();
    }
}

// ---- Python entry point ----

torch::Tensor decode_rle_streams(
    torch::Tensor data,             // uint8, concat decompressed pages + 4B pad
    torch::Tensor page_offset,      // int32, [n_pages]
    torch::Tensor page_length,      // int32, [n_pages]
    torch::Tensor page_max_rep,     // int32, [n_pages]
    torch::Tensor page_max_def,     // int32, [n_pages]
    torch::Tensor page_num_values,  // int32, [n_pages]
    torch::Tensor page_out_start,   // int32, [n_pages] prefix-sum of num_values
    int total_values                // sum(num_values)
) {
    TORCH_CHECK(data.is_cuda(), "data must be on CUDA");
    int n_pages = page_offset.size(0);
    int n_streams = n_pages * 3;
    int output_size = 3 * total_values;

    auto output = torch::zeros({output_size},
                               torch::dtype(torch::kInt32).device(data.device()));

    if (n_pages == 0 || total_values == 0) return output;

    // Descriptors only: 1536 * 5 * 4 = 30720 bytes
    int smem_size = MAX_SLOW_GROUPS * 5 * (int)sizeof(int);

    decode_rle_streams_kernel<<<n_streams, BLOCK_SIZE, smem_size,
                                at::cuda::getCurrentCUDAStream()>>>(
        data.data_ptr<uint8_t>(),
        page_offset.data_ptr<int>(),
        page_length.data_ptr<int>(),
        page_max_rep.data_ptr<int>(),
        page_max_def.data_ptr<int>(),
        page_num_values.data_ptr<int>(),
        page_out_start.data_ptr<int>(),
        total_values,
        output.data_ptr<int>(),
        n_pages
    );

    return output;
}
