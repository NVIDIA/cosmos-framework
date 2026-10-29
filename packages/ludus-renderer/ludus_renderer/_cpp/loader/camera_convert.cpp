// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <cmath>
#include <vector>

static double eval_poly(const double* coeffs, int n_coeffs, double x) {
    double result = 0.0, xi = 1.0;
    for (int i = 0; i < n_coeffs; i++) {
        result += coeffs[i] * xi;
        xi *= x;
    }
    return result;
}

// Compute fw_poly and max_ray_angle for all cameras.
//
// Inputs (all CPU tensors):
//   poly_coeffs: [n_cameras, max_poly_len] float64, padded polynomial coefficients
//   poly_lengths: [n_cameras] int32, actual length of each polynomial
//   is_bw_poly: [n_cameras] bool
//   cx_raw, cy_raw: [n_cameras] float64, original (unscaled) principal points
//   img_w_raw, img_h_raw: [n_cameras] float64, original image dimensions
//   cx_scaled, cy_scaled: [n_cameras] float64, scaled principal points
//   img_w_scaled, img_h_scaled: [n_cameras] float64, scaled image dimensions
//   poly_scale: [n_cameras] float64, scaling factor for polynomial coefficients
//
// Returns:
//   fw_poly: [n_cameras, 6] float32
//   max_ray_angle: [n_cameras] float32
std::vector<torch::Tensor> compute_camera_params(
    torch::Tensor poly_coeffs,
    torch::Tensor poly_lengths,
    torch::Tensor is_bw_poly,
    torch::Tensor cx_raw,
    torch::Tensor cy_raw,
    torch::Tensor img_w_raw,
    torch::Tensor img_h_raw,
    torch::Tensor cx_scaled,
    torch::Tensor cy_scaled,
    torch::Tensor img_w_scaled,
    torch::Tensor img_h_scaled,
    torch::Tensor poly_scale_t
) {
    const int n = poly_coeffs.size(0);
    const int max_poly = poly_coeffs.size(1);

    auto fw_out = torch::zeros({n, 6}, torch::kFloat32);
    auto mra_out = torch::empty({n}, torch::kFloat32);

    auto pc_a = poly_coeffs.accessor<double, 2>();
    auto pl_a = poly_lengths.accessor<int32_t, 1>();
    auto bw_a = is_bw_poly.accessor<bool, 1>();
    auto cx_r = cx_raw.accessor<double, 1>();
    auto cy_r = cy_raw.accessor<double, 1>();
    auto w_r = img_w_raw.accessor<double, 1>();
    auto h_r = img_h_raw.accessor<double, 1>();
    auto cx_s = cx_scaled.accessor<double, 1>();
    auto cy_s = cy_scaled.accessor<double, 1>();
    auto w_s = img_w_scaled.accessor<double, 1>();
    auto h_s = img_h_scaled.accessor<double, 1>();
    auto ps_a = poly_scale_t.accessor<double, 1>();
    auto fw_a = fw_out.accessor<float, 2>();
    auto mra_a = mra_out.accessor<float, 1>();

    const int N_SAMPLES = 200;

    for (int cam = 0; cam < n; cam++) {
        const double* poly = &pc_a[cam][0];
        int plen = pl_a[cam];
        double pscale = ps_a[cam];
        bool is_bw = bw_a[cam];
        double cxs = cx_s[cam], cys = cy_s[cam];
        double ws = w_s[cam], hs = h_s[cam];

        // fw_poly computation
        if (is_bw && plen > 1 && poly[1] != 0.0) {
            // Backward poly: need polyfit inversion
            double r_max = 0.0;
            double corners[4] = {
                std::sqrt((0 - cx_r[cam]) * (0 - cx_r[cam]) + (0 - cy_r[cam]) * (0 - cy_r[cam])),
                std::sqrt((w_r[cam] - cx_r[cam]) * (w_r[cam] - cx_r[cam]) + (0 - cy_r[cam]) * (0 - cy_r[cam])),
                std::sqrt((0 - cx_r[cam]) * (0 - cx_r[cam]) + (h_r[cam] - cy_r[cam]) * (h_r[cam] - cy_r[cam])),
                std::sqrt((w_r[cam] - cx_r[cam]) * (w_r[cam] - cx_r[cam]) + (h_r[cam] - cy_r[cam]) * (h_r[cam] - cy_r[cam])),
            };
            for (int c = 0; c < 4; c++) r_max = std::max(r_max, corners[c]);

            // Generate samples and evaluate polynomial
            std::vector<double> rs(N_SAMPLES), thetas(N_SAMPLES);
            int n_valid = 0;
            for (int s = 0; s < N_SAMPLES; s++) {
                double t = (double)s / (N_SAMPLES - 1);
                rs[s] = t * r_max;
                thetas[s] = eval_poly(poly, plen, rs[s]);
                if (thetas[s] > 1e-6) n_valid++;
            }

            if (n_valid > 0) {
                // Build Vandermonde system for polyfit: theta -> r, degree 5
                // Solve via normal equations: (V^T V) c = V^T r
                // V[i,j] = theta[i]^j, j=0..5
                double VtV[6][6] = {};
                double Vtr[6] = {};

                for (int s = 0; s < N_SAMPLES; s++) {
                    if (thetas[s] <= 1e-6) continue;
                    double th = thetas[s], r = rs[s];
                    double th_pow[6];
                    th_pow[0] = 1.0;
                    for (int j = 1; j < 6; j++) th_pow[j] = th_pow[j-1] * th;

                    for (int j = 0; j < 6; j++) {
                        Vtr[j] += th_pow[j] * r;
                        for (int k = 0; k < 6; k++) {
                            VtV[j][k] += th_pow[j] * th_pow[k];
                        }
                    }
                }

                // Solve 6x6 system via Cholesky-like pivoted Gaussian elimination
                double A[6][7];
                for (int j = 0; j < 6; j++) {
                    for (int k = 0; k < 6; k++) A[j][k] = VtV[j][k];
                    A[j][6] = Vtr[j];
                }
                for (int j = 0; j < 6; j++) {
                    // Partial pivoting
                    int pivot = j;
                    for (int k = j + 1; k < 6; k++) {
                        if (std::abs(A[k][j]) > std::abs(A[pivot][j])) pivot = k;
                    }
                    if (pivot != j) {
                        for (int k = 0; k < 7; k++) std::swap(A[j][k], A[pivot][k]);
                    }
                    if (std::abs(A[j][j]) < 1e-15) continue;
                    double inv = 1.0 / A[j][j];
                    for (int k = j; k < 7; k++) A[j][k] *= inv;
                    for (int i = 0; i < 6; i++) {
                        if (i == j) continue;
                        double f = A[i][j];
                        for (int k = j; k < 7; k++) A[i][k] -= f * A[j][k];
                    }
                }

                // coeffs are A[j][6], in order c0..c5 (theta^0..theta^5)
                // Apply poly_scale and zero out c0
                fw_a[cam][0] = 0.0f;
                for (int j = 1; j < 6; j++) {
                    fw_a[cam][j] = (float)(A[j][6] * pscale);
                }
            } else {
                double fw_k1 = (1.0 / poly[1]) * pscale;
                fw_a[cam][0] = 0.0f;
                fw_a[cam][1] = (float)fw_k1;
                for (int j = 2; j < 6; j++) fw_a[cam][j] = 0.0f;
            }
        } else {
            // Forward poly: just copy with scale
            int copy_len = std::min(plen, 6);
            for (int j = 0; j < copy_len; j++) {
                fw_a[cam][j] = (float)(poly[j] * pscale);
            }
        }

        // Corner distance for max_ray_angle
        double dx[4] = {0 - cxs, ws - cxs, 0 - cxs, ws - cxs};
        double dy[4] = {0 - cys, 0 - cys, hs - cys, hs - cys};
        double r_max_corner = 0.0;
        for (int c = 0; c < 4; c++) {
            double d = std::sqrt(dx[c] * dx[c] + dy[c] * dy[c]);
            r_max_corner = std::max(r_max_corner, d);
        }

        double theta_corner;
        if (!is_bw) {
            // Bisection: find theta where poly(theta) = r_max_corner
            double lo = 0.0, hi = M_PI;
            for (int iter = 0; iter < 50; iter++) {
                double mid = (lo + hi) * 0.5;
                double r_mid = eval_poly(poly, plen, mid);
                if (r_mid < r_max_corner) lo = mid; else hi = mid;
            }
            theta_corner = (lo + hi) * 0.5;
        } else {
            theta_corner = eval_poly(poly, plen, r_max_corner);
        }

        mra_a[cam] = (float)(std::abs(theta_corner) * 1.05);
    }

    return {fw_out, mra_out};
}

#ifndef SCENE_LOADER_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_camera_params", &compute_camera_params,
          "Compute fw_poly and max_ray_angle for cameras (GIL-free)",
          py::call_guard<py::gil_scoped_release>());
}
#endif
