#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline inverse-dynamics (ID) eval on the held-out LIBERO episodes.

For each episode and each full 17-frame window k (frames 16k..16k+16 at 20 fps = the FD eval's chunk grid) the 17 REAL
frames are fed with mode=inverse_dynamics (all 5 vision latents conditioning, no action conditioning) and the model
denoises the 16 frame-wise actions from seeded noise; the prediction is compared with the real actions act10[16k:16k+16]
(raw [dpos(3), rot6d(6), gripper(1)], unit-less OSC commands). The real actions never enter the batch (placeholder =
zeros); a same-seed zeros-vs-real leak check runs on the first windows of the first episode. Model loading and the batch
mirror scripts/fd_rollout_eval.py. The model returns NORMALIZED actions (quantile, stats block global_raw; the
ActionProcessingRecord carries no normalizer), denormalized here with the dataset's own stats.
Outputs: run_config.json, summary.json, per_action.csv, per_window.csv, episodes/epNNN/{id_pred.npz,traces.png}, RECORD.md.
"""
from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VIEW_DESC = "The left half shows the third-person view; the right half shows the wrist-mounted camera."
CANVAS_W, CANVAS_H = 320, 192
CHUNK = 16
RAW_DIM = 10
# Unit conversions reported next to the command-unit numbers: least-squares fits through the origin of the achieved
# end-effector displacement (observation.state, world frame) on the command at the same frame, 360 training episodes,
# 96,119 frame pairs (runs/id_eval/RECORD.md): 10.79 / 12.12 / 11.62 mm per unit (x/y/z, R^2 0.861 / 0.931 / 0.912),
# pooled 11.62; achieved relative rotation 6.40 deg per unit of command rotation-vector norm (R^2 0.896).
ALPHA_MM_PER_UNIT = 11.62
ALPHA_XYZ = [10.79, 12.12, 11.62]
BETA_DEG_PER_UNIT = 6.40
ZERO_MOTION10 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]  # dpos 0, R = I, gripper open (train majority)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--root", required=True, help="libero_10 LeRobot dir")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="", help="comma-separated subset of held-out episode ids")
    ap.add_argument("--max-chunks", type=int, default=0, help="debug: windows per episode (0 = all full windows)")
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=3.0, help="UniPC shift (training shift of the 256 tier)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--leak-check-windows", type=int, default=2, help="first N windows of the first episode: zeros vs real batch action")
    ap.add_argument("--video-control", default="none", choices=["none", "shuffled", "static"],
                    help="shuffled: video of another window of the same episode; static: frame 16k repeated 17x")
    ap.add_argument("--title", default="")
    return ap.parse_args()


def load_model(ckpt: str, config_file: str, output_dir: Path, use_ema: bool):
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.scripts.action_policy_server_utils import disable_runtime_ema_for_frozen_config, maybe_init_distributed

    maybe_init_distributed()
    ov = OmniSetupOverrides.model_validate({"checkpoint_path": str(ckpt), "config_file": str(config_file), "use_ema_weights": bool(use_ema)})
    ov.output_dir = output_dir
    ov.sampler = "unipc"
    setup_args = ov.build_setup()
    init_output_dir(setup_args.output_dir)
    setup_args = disable_runtime_ema_for_frozen_config(setup_args)
    pipe = OmniInference.create(setup_args)
    model = pipe.model
    model.eval()
    return pipe, model


def check_id_plan(sp) -> dict:
    """The ID plan for (17 frames, 16 actions): vision conditioning on all 5 latents, no action conditioning, offset 1."""
    info = {k: (list(v) if hasattr(v, "__iter__") else v) for k, v in
            ((k, getattr(sp, k)) for k in ("condition_frame_indexes_vision", "condition_frame_indexes_action", "action_start_frame_offset", "has_action", "has_vision") if hasattr(sp, k))}
    cv = info.get("condition_frame_indexes_vision"); ca = info.get("condition_frame_indexes_action")
    assert cv is None or list(cv) == [0, 1, 2, 3, 4], info
    assert ca is None or list(ca) == [], info
    assert info.get("action_start_frame_offset", 1) == 1, info
    return info


def build_batch(model, device, item: dict, cap: str, action_t_d: torch.Tensor) -> dict:
    """Fresh dict per call: generate_samples_from_batch mutates the batch."""
    from cosmos_framework.data.generator.action.utils.action_processing import make_batched_action_processing_fields

    return {
        model.input_video_key: [[item["video"]]],  # uint8 [3,17,192,320]; 17 real frames, all conditioning
        "action": [[action_t_d]],  # [16,64]; gives the noise its shape; the ID plan has no action conditioning
        **make_batched_action_processing_fields(item["action_processing_record"], 1),  # raw_action_dim=10, normalizer None
        "mode": ["inverse_dynamics"],
        "ai_caption": [cap],
        "prompt": [cap],
        "conditioning_fps": [item["conditioning_fps"]],
        "conditioning_fps_action": [torch.tensor(20, dtype=torch.long)],
        "image_size": item["image_size"].unsqueeze(0).to(device),
        "domain_id": [item["domain_id"]],
        "sequence_plan": [item["sequence_plan"]],
    }


def generate(model, batch: dict, seed_k: int, args) -> torch.Tensor:
    """One UniPC pass; returns the NORMALIZED [16,10] fp32 action prediction on the CPU."""
    with torch.inference_mode():
        samples = model.generate_samples_from_batch(
            batch, guidance=args.guidance, seed=[seed_k], num_steps=args.num_steps, shift=float(args.shift), has_negative_prompt=False,
        )
    assert "action" in samples, "no action output: the plan did not mark the action rows as generated"
    pred = samples["action"][0].float().cpu()
    pred = pred.squeeze(0) if pred.ndim == 3 else pred
    assert tuple(pred.shape) == (CHUNK, RAW_DIM), tuple(pred.shape)
    assert torch.isfinite(pred).all()
    return pred


# ---------------------------------------------------------------- metrics
def gram_schmidt(r6: np.ndarray) -> np.ndarray:
    """rot6d = first two COLUMNS of R ([R00,R10,R20,R01,R11,R21]) -> rotation matrices [N,3,3]."""
    c0, c1 = r6[:, :3].astype(np.float64), r6[:, 3:].astype(np.float64)
    b1 = c0 / np.maximum(np.linalg.norm(c0, axis=1, keepdims=True), 1e-12)
    c1p = c1 - (b1 * c1).sum(1, keepdims=True) * b1
    b2 = c1p / np.maximum(np.linalg.norm(c1p, axis=1, keepdims=True), 1e-12)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def geodesic_deg(Rp: np.ndarray, Rr: np.ndarray) -> np.ndarray:
    tr = np.einsum("nii->n", np.einsum("nij,nkj->nik", Rp, Rr))
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def transitions(s: np.ndarray):
    return [(int(t), "open" if s[t] == 1 else "close") for t in np.flatnonzero(np.diff(s) != 0) + 1]


def match_events(real_ev, gen_ev, tol: int):  # same greedy rule as gripper_classifier.py
    used = set(); matched = []
    for t_r, kind in real_ev:
        best = None
        for j, (t_g, kg) in enumerate(gen_ev):
            if j in used or kg != kind:
                continue
            if abs(t_g - t_r) <= tol and (best is None or abs(t_g - t_r) < abs(gen_ev[best][0] - t_r)):
                best = j
        if best is not None:
            used.add(best); matched.append((t_r, gen_ev[best][0], kind))
    return matched, [e for j, e in enumerate(gen_ev) if j not in used]


def r2(P, Y):
    ss_res = ((P - Y) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def metrics(P, Y, wins, per_ep, q01, q99, tol=10, lag_excl=10) -> dict:
    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

    norm_q = lambda a: 2.0 * (a - q01) / (q99 - q01) - 1.0
    err = P[:, :3] - Y[:, :3]
    out = dict(dpos_mae_u=np.abs(err).mean(0), dpos_rmse_u=np.sqrt((err ** 2).mean(0)), dpos_mean_norm_err_u=float(np.linalg.norm(err, axis=1).mean()))
    out["dpos_mae_mm"] = out["dpos_mae_u"] * np.array(ALPHA_XYZ)
    out["dpos_mean_norm_err_mm"] = ALPHA_MM_PER_UNIT * out["dpos_mean_norm_err_u"]
    Rr = gram_schmidt(Y[:, 3:9]); ang = geodesic_deg(gram_schmidt(P[:, 3:9]), Rr)
    try:
        Rsvd = convert_rotation(torch.tensor(P[:, 3:9], dtype=torch.float32), "rot6d", "matrix", normalize_matrix=True)
        Rsvd = Rsvd.numpy() if hasattr(Rsvd, "numpy") else np.asarray(Rsvd)
        ang_svd = float(geodesic_deg(Rsvd.astype(np.float64), Rr).mean())
    except Exception as e:  # the framework decode is a secondary column; never fail the eval on it
        ang_svd = None; out["rot_svd_decode_error"] = repr(e)[:200]
    out.update(rot_geodesic_deg_mean=float(ang.mean()), rot_geodesic_deg_median=float(np.median(ang)), rot_geodesic_deg_p95=float(np.percentile(ang, 95)),
               rot_geodesic_deg_mean_svd_decode=ang_svd, rot_real_vs_identity_deg_mean=float(geodesic_deg(Rr, np.broadcast_to(np.eye(3), Rr.shape)).mean()))
    po, ro = P[:, 9] > 0.5, Y[:, 9] > 0.5
    def f1(pos):
        tp = ((po == pos) & (ro == pos)).sum(); fp = ((po == pos) & (ro != pos)).sum(); fn = ((po != pos) & (ro == pos)).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    out.update(gripper_acc=float((po == ro).mean()), gripper_f1_open=f1(True), gripper_f1_closed=f1(False))
    out["gripper_macro_f1"] = (out["gripper_f1_open"] + out["gripper_f1_closed"]) / 2
    out["nrmse_per_dim"] = np.sqrt(((norm_q(P) - norm_q(Y)) ** 2).mean(0)); out["nrmse_mean"] = float(out["nrmse_per_dim"].mean())
    out["r2_per_dim"] = r2(P, Y); out["heldout_mean_action"] = Y.mean(0)
    n_real = n_pred = n_match = 0; dts = []; amb = np.zeros(len(Y), bool); ep_end = []
    for e, m in per_ep.items():
        sl = slice(m["off"], m["off"] + CHUNK * m["n_full"])
        ev_r, ev_p = transitions(ro[sl].astype(int)), transitions(po[sl].astype(int))
        mt, _ = match_events(ev_r, ev_p, tol)
        n_real += len(ev_r); n_pred += len(ev_p); n_match += len(mt); dts += [abs(a - b) for a, b, _ in mt]
        for t, _ in ev_r:
            amb[m["off"] + t : m["off"] + t + lag_excl] = True
        ep_end.append(np.linalg.norm((P[sl, :3] - Y[sl, :3]).sum(0)))
    ch_end = [np.linalg.norm((P[s, :3] - Y[s, :3]).sum(0)) for (_, _, s) in wins]
    out.update(gripper_acc_unambiguous=float((po == ro)[~amb].mean()) if (~amb).any() else None, n_unambiguous=int((~amb).sum()),
               transitions=dict(n_real_in_span=n_real, n_pred=n_pred, n_matched=n_match, recall=n_match / max(n_real, 1),
                                precision=(n_match / n_pred) if n_pred else None, median_abs_dt_frames=(float(np.median(dts)) if dts else None),
                                mean_abs_dt_frames=(float(np.mean(dts)) if dts else None)),
               integrated=dict(chunk_end_err_u_mean=float(np.mean(ch_end)), chunk_end_err_mm_mean=ALPHA_MM_PER_UNIT * float(np.mean(ch_end)),
                               chunk_end_err_mm_median=ALPHA_MM_PER_UNIT * float(np.median(ch_end)), episode_end_err_u_mean=float(np.mean(ep_end)),
                               episode_end_err_mm_mean=ALPHA_MM_PER_UNIT * float(np.mean(ep_end)), episode_end_err_mm_median=ALPHA_MM_PER_UNIT * float(np.median(ep_end)),
                               episode_end_err_mm_max=ALPHA_MM_PER_UNIT * float(np.max(ep_end))))
    return out


def tolist(o):
    if isinstance(o, dict): return {k: tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [tolist(v) for v in o]
    if isinstance(o, np.ndarray): return [float(x) for x in o.ravel()] if o.ndim else float(o)
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


def train_mean_action(root: str, holdout: str) -> tuple[np.ndarray, int]:
    """Constant baseline: mean raw action over all frames of the TRAIN split (per-episode frame-wise actions)."""
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
    ds = LIBEROLeRobotDataset(root=root, fps=20, chunk_length=CHUNK, mode="inverse_dynamics", camera_mode="concat_view", image_size=256,
                              action_space="frame_wise_relative", rotation_space="6d", pose_coordinate_frame="native",
                              action_normalization="quantile_rot", split="train", holdout_episodes_path=holdout)
    acc = []; n = 0
    for ep in [int(e) for e in ds._ep_vals.tolist()]:
        rows = np.flatnonzero(ds._row_episode == ep)
        a = ds._build_frame_wise_action(ds._row_action[rows]).numpy(); acc.append(a.sum(0)); n += len(a)
    return np.sum(acc, 0) / n, n


def plot_traces(ed: Path, ep: int, real10: np.ndarray, pred10: np.ndarray, task: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    t = np.arange(len(real10))
    for i, name in enumerate(["dx", "dy", "dz"]):
        ax[i].plot(t, real10[:, i], color="#1f6feb", lw=1.4, label="real"); ax[i].plot(t, pred10[:, i], color="#d1242f", lw=1.0, label="predicted (ID)")
        ax[i].set_ylabel(name + " (cmd units)"); ax[i].grid(alpha=.3)
    ax[3].step(t, real10[:, 9], color="#1f6feb", lw=1.4, where="post", label="real"); ax[3].step(t, pred10[:, 9], color="#d1242f", lw=1.0, where="post", label="predicted")
    ax[3].set_ylabel("gripper (1 = open)"); ax[3].set_xlabel("action index (20 Hz)"); ax[3].grid(alpha=.3); ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"episode {ep}: {task}"); fig.tight_layout(); fig.savefig(ed / "traces.png", dpi=110); plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / "episodes").mkdir(exist_ok=True)
    t_start = time.time()
    pipe, model = load_model(args.checkpoint_path, args.config_file, out / "_omni_inference", use_ema=not args.no_ema)
    t_load = time.time() - t_start
    device = torch.device("cuda")

    from cosmos_framework.data.generator.action.action_normalization import denormalize_action, normalize_action
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    ds = LIBEROLeRobotDataset(
        root=args.root, fps=20, chunk_length=CHUNK, mode="inverse_dynamics", camera_mode="concat_view",
        image_size=256, action_space="frame_wise_relative", rotation_space="6d", pose_coordinate_frame="native",
        action_normalization="quantile_rot", split="val", holdout_episodes_path=args.holdout,
    )
    max_action_dim = int(model.config.max_action_dim)
    transform = ActionTransformPipeline(tokenizer_config=None, cfg_dropout_rate=0.0, max_action_dim=max_action_dim,
                                        append_idle_frames=True, idle_frames_dropout=0.0, format_prompt_as_json=True)
    norm_method = ds.action_normalization  # the method _build_result applies (and the inverse we apply to predictions)
    stats = ds._load_norm_stats()
    assert norm_method == "quantile", norm_method
    assert stats["q01"].numel() == RAW_DIM
    q01, q99 = stats["q01"].numpy().astype(np.float64), stats["q99"].numpy().astype(np.float64)
    zeros_action = torch.zeros(CHUNK, max_action_dim, dtype=torch.float32)
    mean_train, n_train = train_mean_action(args.root, args.holdout)

    ep_ids = [int(e) for e in ds._ep_vals.tolist()]
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",") if x.strip()}
        ep_ids = [e for e in ep_ids if e in want]

    run_cfg = dict(
        checkpoint_path=str(args.checkpoint_path), config_file=str(args.config_file), use_ema_weights=not args.no_ema, mode="inverse_dynamics",
        num_steps=args.num_steps, guidance=args.guidance, shift=float(args.shift), seed=args.seed, seed_rule="seed*100000 + ep*100 + k",
        chunk=CHUNK, window_frames=CHUNK + 1, fps=20, conditioning_fps=20, conditioning_fps_action=20, partial_windows="excluded",
        batch_action="zeros", video_control=args.video_control, canvas=[CANVAS_H, CANVAS_W], content=[160, 320],
        stats_file=str(getattr(ds, "_stats_file", "")), stats_key=str(getattr(ds, "_stats_key", "")), normalization=norm_method,
        denormalize="a = 0.5*(a_n+1)*(q99-q01)+q01", rot6d_decode="gram_schmidt (headline); framework convert_rotation svd logged",
        alpha_mm_per_unit=ALPHA_MM_PER_UNIT, alpha_mm_per_unit_xyz=ALPHA_XYZ, beta_deg_per_unit=BETA_DEG_PER_UNIT,
        gripper_open_value=1, gripper_threshold=0.5, transition_tol_frames=10, cmd_lag_exclude_frames=10,
        train_mean_action=[float(x) for x in mean_train], train_mean_n_frames=int(n_train), model_load_s=round(t_load, 1), episodes=ep_ids,
    )
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    leak, action_rows, window_rows, per_ep_meta, plan_info = [], [], [], {}, None
    P_norm, P_raw, Y_raw, wins = [], [], [], []
    off = 0
    for ep_i, ep in enumerate(ep_ids):
        rows = np.flatnonzero(ds._row_episode == ep)
        assert len(rows) > 1 and np.all(np.diff(rows) == 1)
        T = int(len(rows)); ts = [float(ds._row_timestamp[j]) for j in rows]
        task = ds._tasks[int(ds._row_task[rows[0]])].split(" | ")[0].strip()
        t0 = time.time()
        real01 = ds._load_video(ds._episodes[ep], ts)  # [T,3,256,512] float [0,1]
        act10 = ds._build_frame_wise_action(ds._row_action[rows])  # [T,10] raw
        t_decode = time.time() - t0
        n_full = (T - 1) // CHUNK
        if args.max_chunks > 0:
            n_full = min(n_full, args.max_chunks)
        times = []; ep_P, ep_Pn = [], []
        for k in range(n_full):
            f0 = CHUNK * k
            gt10 = act10[f0 : f0 + CHUNK]
            if args.video_control == "static":
                video_in = real01[f0 : f0 + 1].repeat(CHUNK + 1, 1, 1, 1)
            elif args.video_control == "shuffled":
                fs = CHUNK * ((k + max(n_full // 2, 1)) % n_full); video_in = real01[fs : fs + CHUNK + 1]
            else:
                video_in = real01[f0 : f0 + CHUNK + 1]
            item = transform(ds._build_result(mode="inverse_dynamics", video=video_in, action=torch.zeros(CHUNK, RAW_DIM), ai_caption=task,
                                              additional_view_description=VIEW_DESC), resolution=None)
            if plan_info is None:
                plan_info = check_id_plan(item["sequence_plan"]); print(f"[id_eval] sequence plan: {plan_info}", flush=True)
            assert tuple(item["video"].shape) == (3, CHUNK + 1, CANVAS_H, CANVAS_W), tuple(item["video"].shape)
            cap = item["ai_caption"]; cap = json.dumps(cap) if isinstance(cap, dict) else str(cap)
            assert "idle_frame" not in cap, cap
            seed_k = args.seed * 100000 + ep * 100 + k
            tc0 = time.time()
            pred_norm = generate(model, build_batch(model, device, item, cap, zeros_action), seed_k, args)
            times.append(time.time() - tc0)
            if ep_i == 0 and k < args.leak_check_windows:
                item_real = transform(ds._build_result(mode="inverse_dynamics", video=video_in, action=gt10.clone(), ai_caption=task,
                                                       additional_view_description=VIEW_DESC), resolution=None)
                pred_real = generate(model, build_batch(model, device, item_real, cap, item_real["action"]), seed_k, args)
                pred_rep = generate(model, build_batch(model, device, item, cap, zeros_action), seed_k, args)
                pred_other_seed = generate(model, build_batch(model, device, item, cap, zeros_action), seed_k + 1, args)
                leak.append(dict(episode=ep, k=k, bit_identical_real_vs_zeros=bool(torch.equal(pred_real, pred_norm)),
                                 max_abs_diff_real_vs_zeros=float((pred_real - pred_norm).abs().max()),
                                 max_abs_diff_zeros_repeat=float((pred_rep - pred_norm).abs().max()),
                                 max_abs_diff_other_seed=float((pred_other_seed - pred_norm).abs().max())))
                print(f"[id_eval] leak check {leak[-1]}", flush=True)
            pred10 = denormalize_action(pred_norm, norm_method, stats)
            gt_norm = normalize_action(gt10, norm_method, stats)
            P_norm.append(pred_norm.numpy()); P_raw.append(pred10.numpy()); Y_raw.append(gt10.numpy()); ep_P.append(pred10.numpy()); ep_Pn.append(pred_norm.numpy())
            wins.append((ep, k, slice(off + CHUNK * k, off + CHUNK * k + CHUNK)))
            Rr = gram_schmidt(gt10.numpy()[:, 3:9]); angs = geodesic_deg(gram_schmidt(pred10.numpy()[:, 3:9]), Rr)
            for j in range(CHUNK):
                action_rows.append(dict(episode=ep, chunk_index=k, pos_in_window=j + 1, action_index=f0 + j, seed_k=seed_k,
                                        **{f"real_d{d}": float(gt10[j, d]) for d in range(RAW_DIM)}, **{f"pred_d{d}": float(pred10[j, d]) for d in range(RAW_DIM)},
                                        **{f"pred_norm_d{d}": float(pred_norm[j, d]) for d in range(RAW_DIM)}, **{f"real_norm_d{d}": float(gt_norm[j, d]) for d in range(RAW_DIM)},
                                        err_norm_u=float(np.linalg.norm(pred10[j, :3].numpy() - gt10[j, :3].numpy())), rot_geodesic_deg=float(angs[j]),
                                        real_open=int(gt10[j, 9] > 0.5), pred_open=int(pred10[j, 9] > 0.5)))
            e3 = pred10.numpy()[:, :3] - gt10.numpy()[:, :3]
            window_rows.append(dict(episode=ep, chunk_index=k, frame_start=f0, frame_end=f0 + CHUNK, seed_k=seed_k, gen_s=round(times[-1], 3),
                                    dpos_mae_x_u=float(np.abs(e3[:, 0]).mean()), dpos_mae_y_u=float(np.abs(e3[:, 1]).mean()), dpos_mae_z_u=float(np.abs(e3[:, 2]).mean()),
                                    dpos_mean_norm_err_mm=float(ALPHA_MM_PER_UNIT * np.linalg.norm(e3, axis=1).mean()), rot_geodesic_deg_mean=float(angs.mean()), rot_geodesic_deg_max=float(angs.max()),
                                    gripper_acc=float(((pred10[:, 9] > 0.5) == (gt10[:, 9] > 0.5)).float().mean()), chunk_end_err_u=float(np.linalg.norm(e3.sum(0))),
                                    chunk_end_err_mm=float(ALPHA_MM_PER_UNIT * np.linalg.norm(e3.sum(0))), pred_norm_absmax=float(pred_norm.abs().max())))
            if ep_i == 0 and k == 0:
                print(f"[id_eval] prompt example: {cap}", flush=True)
            print(f"[id_eval] ep {ep} window {k + 1}/{n_full} {times[-1]:.2f}s  pred_norm|max| {pred_norm.abs().max():.2f}", flush=True)
        per_ep_meta[ep] = dict(T=T, n_full=n_full, off=off, task=task, gen_s_per_window=float(np.mean(times)) if times else None, decode_s=round(t_decode, 2))
        off += CHUNK * n_full
        ed = out / "episodes" / f"ep{ep:03d}"; ed.mkdir(exist_ok=True)
        real_ep = act10[: CHUNK * n_full].numpy(); pred_ep = np.concatenate(ep_P) if ep_P else np.zeros((0, RAW_DIM), np.float32)
        np.savez_compressed(ed / "id_pred.npz", real10=real_ep, pred10=pred_ep, pred_norm=(np.concatenate(ep_Pn) if ep_Pn else pred_ep), window_starts=np.arange(n_full) * CHUNK)
        plot_traces(ed, ep, real_ep, pred_ep, task)

    # ---------------------------------------------------------------- scoring
    P, Pn, Y = np.concatenate(P_raw).astype(np.float64), np.concatenate(P_norm).astype(np.float64), np.concatenate(Y_raw).astype(np.float64)
    agg = metrics(P, Y, wins, per_ep_meta, q01, q99)
    baselines = {
        "train_mean_constant": {"vector": [float(x) for x in mean_train], "n_train_frames": int(n_train),
                                 "aggregate": metrics(np.broadcast_to(mean_train.astype(np.float64), Y.shape).copy(), Y, wins, per_ep_meta, q01, q99)},
        "zero_motion_majority_open": {"vector": ZERO_MOTION10, "aggregate": metrics(np.broadcast_to(np.array(ZERO_MOTION10), Y.shape).copy(), Y, wins, per_ep_meta, q01, q99)},
    }
    per_episode = []
    for ep, m in per_ep_meta.items():
        sl = slice(m["off"], m["off"] + CHUNK * m["n_full"])
        if m["n_full"] == 0:
            continue
        w = [(e, k, s) for (e, k, s) in wins if e == ep]
        sub = metrics(P[sl], Y[sl], [(e, k, slice(s.start - m["off"], s.stop - m["off"])) for (e, k, s) in w], {ep: dict(off=0, n_full=m["n_full"])}, q01, q99)
        per_episode.append(dict(episode=ep, task=m["task"], frames_total=m["T"], windows=m["n_full"], actions_scored=CHUNK * m["n_full"],
                                actions_excluded_tail=(m["T"] - 1) - CHUNK * m["n_full"], dpos_mae_u=sub["dpos_mae_u"], dpos_mean_norm_err_mm=sub["dpos_mean_norm_err_mm"],
                                rot_geodesic_deg_mean=sub["rot_geodesic_deg_mean"], gripper_acc=sub["gripper_acc"], n_real_transitions=sub["transitions"]["n_real_in_span"],
                                n_pred_transitions=sub["transitions"]["n_pred"], n_matched=sub["transitions"]["n_matched"], r2_per_dim=sub["r2_per_dim"],
                                chunk_end_err_mm_mean=sub["integrated"]["chunk_end_err_mm_mean"], episode_end_err_mm=sub["integrated"]["episode_end_err_mm_mean"],
                                gen_s_per_window=m["gen_s_per_window"], decode_s=m["decode_s"]))
    per_chunk = {}
    for k in sorted({k for (_, k, _) in wins}):
        idx = np.concatenate([np.arange(s.start, s.stop) for (_, kk, s) in wins if kk == k])
        e3 = P[idx, :3] - Y[idx, :3]; ang = geodesic_deg(gram_schmidt(P[idx, 3:9]), gram_schmidt(Y[idx, 3:9]))
        ce = [np.linalg.norm((P[s, :3] - Y[s, :3]).sum(0)) for (_, kk, s) in wins if kk == k]
        per_chunk[str(k)] = dict(n_windows=len(ce), dpos_mean_norm_err_mm=float(ALPHA_MM_PER_UNIT * np.linalg.norm(e3, axis=1).mean()), rot_geodesic_deg_mean=float(ang.mean()),
                                 gripper_acc=float(((P[idx, 9] > 0.5) == (Y[idx, 9] > 0.5)).mean()), chunk_end_err_mm_mean=float(ALPHA_MM_PER_UNIT * np.mean(ce)))
    by_pos = {}
    for j in range(CHUNK):
        idx = np.array([s.start + j for (_, _, s) in wins])
        e3 = P[idx, :3] - Y[idx, :3]; ang = geodesic_deg(gram_schmidt(P[idx, 3:9]), gram_schmidt(Y[idx, 3:9]))
        by_pos[str(j + 1)] = dict(dpos_mean_norm_err_u=float(np.linalg.norm(e3, axis=1).mean()), rot_geodesic_deg_mean=float(ang.mean()), gripper_acc=float(((P[idx, 9] > 0.5) == (Y[idx, 9] > 0.5)).mean()))
    n_actions = int(len(Y)); n_excl = int(sum((m["T"] - 1) - CHUNK * m["n_full"] for m in per_ep_meta.values()))
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = dict(run_config=run_cfg, sequence_plan=plan_info, n_episodes=len(per_ep_meta), n_windows=len(wins), n_actions_scored=n_actions,
                   n_partial_chunks_excluded=int(sum(1 for m in per_ep_meta.values() if (m["T"] - 1) % CHUNK)), n_actions_excluded_tail=n_excl, leak_check=leak,
                   pred_norm_absmax=float(np.abs(Pn).max()), aggregate=agg, per_episode=per_episode, per_chunk_index=per_chunk, by_position_in_window=by_pos,
                   baselines=baselines, wall_s_total=round(time.time() - t_start, 1), gen_s_per_window_mean=float(np.mean([w["gen_s"] for w in window_rows])), peak_gpu_mem_gb=round(peak, 2))
    (out / "summary.json").write_text(json.dumps(tolist(summary), indent=2))
    with open(out / "per_action.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(action_rows[0].keys())); w.writeheader(); w.writerows(action_rows)
    with open(out / "per_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys())); w.writeheader(); w.writerows(window_rows)
    a = agg; b1 = baselines["train_mean_constant"]["aggregate"]; b2 = baselines["zero_motion_majority_open"]["aggregate"]
    fmt = lambda x, p=3: ("n/a" if x is None else f"{x:.{p}f}")
    rec = [f"# RECORD — inverse-dynamics eval{(': ' + args.title) if args.title else ''}", "",
           f"Checkpoint `{args.checkpoint_path}` (EMA {not args.no_ema}), config `{args.config_file}`; mode inverse_dynamics; {len(per_ep_meta)} episodes, {len(wins)} windows, {n_actions} actions scored ({n_excl} tail actions excluded); UniPC {args.num_steps} steps, guidance {args.guidance}, shift {args.shift}, seed {args.seed} (per window seed*100000+ep*100+k); video control: {args.video_control}.",
           f"Model load {t_load:.1f} s, {summary['gen_s_per_window_mean']:.2f} s per window, wall {summary['wall_s_total']:.0f} s, peak GPU {peak:.2f} GB. Leak check (zeros vs real action in the batch, same seed): {leak}.", "",
           "| metric | this run | train-mean constant | zero motion + open |", "| --- | --- | --- | --- |",
           f"| dpos MAE x/y/z (units) | {' / '.join(fmt(v) for v in a['dpos_mae_u'])} | {' / '.join(fmt(v) for v in b1['dpos_mae_u'])} | {' / '.join(fmt(v) for v in b2['dpos_mae_u'])} |",
           f"| mean ‖dpos error‖ (units; mm at {ALPHA_MM_PER_UNIT}/unit) | {fmt(a['dpos_mean_norm_err_u'])}; {fmt(a['dpos_mean_norm_err_mm'],2)} | {fmt(b1['dpos_mean_norm_err_u'])}; {fmt(b1['dpos_mean_norm_err_mm'],2)} | {fmt(b2['dpos_mean_norm_err_u'])}; {fmt(b2['dpos_mean_norm_err_mm'],2)} |",
           f"| rotation geodesic mean / median / p95 (deg) | {fmt(a['rot_geodesic_deg_mean'],2)} / {fmt(a['rot_geodesic_deg_median'],2)} / {fmt(a['rot_geodesic_deg_p95'],2)} | {fmt(b1['rot_geodesic_deg_mean'],2)} / {fmt(b1['rot_geodesic_deg_median'],2)} / {fmt(b1['rot_geodesic_deg_p95'],2)} | {fmt(b2['rot_geodesic_deg_mean'],2)} / {fmt(b2['rot_geodesic_deg_median'],2)} / {fmt(b2['rot_geodesic_deg_p95'],2)} |",
           f"| gripper sign accuracy (all / unambiguous) | {fmt(a['gripper_acc'])} / {fmt(a['gripper_acc_unambiguous'])} | {fmt(b1['gripper_acc'])} / {fmt(b1['gripper_acc_unambiguous'])} | {fmt(b2['gripper_acc'])} / {fmt(b2['gripper_acc_unambiguous'])} |",
           f"| gripper macro-F1 | {fmt(a['gripper_macro_f1'])} | {fmt(b1['gripper_macro_f1'])} | {fmt(b2['gripper_macro_f1'])} |",
           f"| transitions recall / precision (±10 frames, {a['transitions']['n_real_in_span']} real) | {fmt(a['transitions']['recall'],2)} / {fmt(a['transitions']['precision'],2)} | {fmt(b1['transitions']['recall'],2)} / {fmt(b1['transitions']['precision'],2)} | {fmt(b2['transitions']['recall'],2)} / {fmt(b2['transitions']['precision'],2)} |",
           f"| median abs dt of matched transitions (frames) | {fmt(a['transitions']['median_abs_dt_frames'],1)} | n/a | n/a |",
           f"| nRMSE mean over 10 dims | {fmt(a['nrmse_mean'])} | {fmt(b1['nrmse_mean'])} | {fmt(b2['nrmse_mean'])} |",
           f"| R² dpos x/y/z | {' / '.join(fmt(v,2) for v in a['r2_per_dim'][:3])} | {' / '.join(fmt(v,2) for v in b1['r2_per_dim'][:3])} | {' / '.join(fmt(v,2) for v in b2['r2_per_dim'][:3])} |",
           f"| R² gripper | {fmt(a['r2_per_dim'][9],2)} | {fmt(b1['r2_per_dim'][9],2)} | {fmt(b2['r2_per_dim'][9],2)} |",
           f"| integrated error at chunk end (mm, mean / median) | {fmt(a['integrated']['chunk_end_err_mm_mean'],1)} / {fmt(a['integrated']['chunk_end_err_mm_median'],1)} | {fmt(b1['integrated']['chunk_end_err_mm_mean'],1)} / {fmt(b1['integrated']['chunk_end_err_mm_median'],1)} | {fmt(b2['integrated']['chunk_end_err_mm_mean'],1)} / {fmt(b2['integrated']['chunk_end_err_mm_median'],1)} |",
           f"| integrated error at episode end (mm, mean / median / max) | {fmt(a['integrated']['episode_end_err_mm_mean'],1)} / {fmt(a['integrated']['episode_end_err_mm_median'],1)} / {fmt(a['integrated']['episode_end_err_mm_max'],1)} | {fmt(b1['integrated']['episode_end_err_mm_mean'],1)} / {fmt(b1['integrated']['episode_end_err_mm_median'],1)} / {fmt(b1['integrated']['episode_end_err_mm_max'],1)} | {fmt(b2['integrated']['episode_end_err_mm_mean'],1)} / {fmt(b2['integrated']['episode_end_err_mm_median'],1)} / {fmt(b2['integrated']['episode_end_err_mm_max'],1)} |",
           "", f"Real frame-wise rotations average {a['rot_real_vs_identity_deg_mean']:.2f} deg from identity; rotation decoded by Gram-Schmidt (framework SVD decode: {fmt(a['rot_geodesic_deg_mean_svd_decode'],2)} deg mean). Max |normalized prediction| {summary['pred_norm_absmax']:.2f}.",
           f"Train-mean baseline vector (over {n_train:,} train frames): {[round(float(x), 5) for x in mean_train]}."]
    (out / "RECORD.md").write_text("\n".join(rec) + "\n")
    print(f"[id_eval] done: {n_actions} actions in {len(wins)} windows; dpos mean err {a['dpos_mean_norm_err_mm']:.2f} mm (baseline {b1['dpos_mean_norm_err_mm']:.2f}); "
          f"rot {a['rot_geodesic_deg_mean']:.2f} deg (baseline {b1['rot_geodesic_deg_mean']:.2f}); gripper acc {a['gripper_acc']:.3f} (baseline {b1['gripper_acc']:.3f}); "
          f"transitions recall {a['transitions']['recall']:.2f}; peak {peak:.1f} GB; {summary['gen_s_per_window_mean']:.2f} s/window", flush=True)


if __name__ == "__main__":
    main()
