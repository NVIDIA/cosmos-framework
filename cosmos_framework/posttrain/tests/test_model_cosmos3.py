# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Gate-2 — `Cosmos3FlowPolicy` adapter fidelity against the real OmniMoT.

These are L1 GPU integration tests: they instantiate the 8B OmniMoT (random VLM
backbone) and assert the adapter's `denoise_error` reproduces the model's OWN native
flow-matching per-instance loss on a shared `(x0, sigma, epsilon)`. This proves the
adapter wires the real OmniMoT primitives (`_pack_input_sequence`, `get_interpolation`,
`denoise`, fp32 mean-MSE reduction) correctly — covering both the T2V (unmasked) and
I2V (condition-masked) paths — so the model-agnostic DPO objective in
`algorithm/dpo.py` (gate-1 verified) drives this model faithfully.

The shared-epsilon trick: drive the model's own `_add_noise_to_input` (which draws
epsilon internally) to get the native loss, extract that exact epsilon, then feed it
to the adapter. The only thing under test is the adapter's pack→interpolate→denoise→
reduce wiring; the noise and the net are held identical.

Run (single GPU is sufficient; the model needs torch.distributed):
    PYTHONPATH=. torchrun --nproc_per_node=1 --standalone -m pytest -v -s \
        cosmos_framework/posttrain/tests/test_model_cosmos3.py --L1

Verified 2026-06-28 on GB200 (8B random-init): T2V adapter 2.7280 vs native 2.7285
(rel 1.5e-4); I2V (cond_mask [3,1,1]) adapter 2.7249 vs native 2.7247 (rel 8.5e-5);
full dpo_objective(winner==loser) = ln2, margin 0, acc 0.5.
"""

from __future__ import annotations

import math
import os
import socket

import pytest
import torch

from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.posttrain.algorithm.dpo import dpo_objective
from cosmos_framework.posttrain.model.cosmos3 import Cosmos3FlowPolicy, _require_cosmos3_condition

pytestmark = RunIf(supported_arch=["B200"])

VFM_8B_TEST_EXPERIMENT = "t2w_mot_exp302_000_qwen3_vl_8b_multires"


def _maybe_init_distributed() -> None:
    """Init a single-process NCCL group if not already up (OmniMoT queries world_size
    during setup). Mirrors omni_mot_model_test._maybe_init_distributed."""
    if not (torch.distributed.is_available() and not torch.distributed.is_initialized()):
        return
    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not has_env:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, port = s.getsockname()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if has_env:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)


@pytest.fixture(scope="module")
def policy() -> Cosmos3FlowPolicy:
    """Instantiate the 8B OmniMoT once and wrap it in the adapter (with a frozen ref).
    `net.eval()` makes the two forwards (native vs adapter) deterministic so the only
    residual difference is nondeterministic-kernel noise (asserted within tolerance)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _maybe_init_distributed()
    # The CI job selects public checkpoint resolution. Register the pinned Wan2.2
    # HF mirror in this pytest process before OmniMoT constructs its tokenizer.
    from cosmos3.common.checkpoints import register_checkpoints

    register_checkpoints()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    overrides = [
        "--",
        f"experiment={VFM_8B_TEST_EXPERIMENT}",
        "model.config.vlm_config.pretrained_weights.enabled=False",
        "++model.config.vlm_config.model_instance.config.include_visual=True",
        f"model.config.parallelism.data_parallel_shard_degree={world_size}",
        "callbacks=[basic,optimization,job_monitor]",
    ]
    # This contract needs one experiment, not the full training registry.
    config = load_config(
        "cosmos_framework/configs/base/config.py",
        overrides,
        experiment_module=(
            "cosmos_framework.configs.base.experiment.pretraining."
            "image_video_mixed_resolution.exp302_joint_img_vid_mixed_res_qwen8b"
        ),
    )
    model = instantiate(config.model).cuda()
    model.net.eval()
    return Cosmos3FlowPolicy(model, build_ref=True)


def _build_batch(
    device: torch.device, B: int = 1, T: int = 9, H: int = 256, W: int = 256, cond_frames=None, seed: int | None = None
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    batch = {
        "ai_caption": ["a test video"] * B,
        "video": [[torch.randint(0, 256, (3, T, H, W), generator=g, dtype=torch.uint8).to(device)] for _ in range(B)],
        "image_size": [torch.tensor([[H, W, H, W]], dtype=torch.float32).to(device) for _ in range(B)],
        # Identical prompt across the pair (DPO symmetry); deterministic so winner/loser match.
        "text_token_ids": [[torch.arange(64, dtype=torch.long).to(device)] for _ in range(B)],
        "conditioning_fps": [torch.tensor([24.0]).to(device).to(torch.bfloat16) for _ in range(B)],
    }
    if cond_frames is not None:
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        batch["sequence_plan"] = [
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(cond_frames))
            for _ in range(B)
        ]
    return batch


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize("cond_frames", [None, [0]], ids=["t2v_unmasked", "i2v_masked"])
def test_adapter_fidelity_vs_native(policy: Cosmos3FlowPolicy, cond_frames) -> None:
    """Adapter `denoise_error` == OmniMoT native per-instance FM loss on shared (x0,σ,ε)."""
    model = policy.omnimot
    dev = torch.device("cuda")
    batch = _build_batch(dev, cond_frames=cond_frames)

    x0, cond = policy.encode_and_condition(batch)
    cond = _require_cosmos3_condition(cond)
    cm = policy.condition_mask(cond)
    assert (cm is None) == (cond_frames is None), f"condition_mask presence mismatch for cond_frames={cond_frames}"

    is_image = cond.is_image_batch
    rf = model.rectified_flow_image if is_image else model.rectified_flow_video
    n_train = rf.noise_scheduler.config.num_train_timesteps
    torch.manual_seed(0)
    _ts, sigmas = model._get_train_noise_level_vision(
        batch_size=int(x0.shape[0]),
        is_image_batch=is_image,
        num_vision_latent_frames=[int(x0.shape[2])] * int(x0.shape[0]),
        resolutions=cond.resolutions,
    )
    timesteps = sigmas * n_train
    gdc = cond.gen_data_clean

    # Native reference via the model's own ops; capture the epsilon it draws.
    packed_native = model._pack_input_sequence(cond.sequence_plans, cond.text_tokens, gdc, timesteps.cpu())
    torch.manual_seed(12345)
    gen_noised = model._add_noise_to_input(gdc, packed_native, sigmas)
    eps_used = gen_noised.epsilon_vision
    model._replace_clean_with_noised(packed_native, gen_noised)
    packed_native.to_cuda()
    with torch.no_grad():
        out_native = model.denoise(net=model.net, data_batch_packed=packed_native)
    _, native = model._compute_flow_matching_loss(
        pred=out_native["preds_vision"],
        target=gen_noised.vt_target_vision,
        condition_mask=packed_native.vision.condition_mask,
        timesteps=timesteps,
        has_valid_tokens=True,
        rectified_flow=rf,
        normalize_by_active=True,  # matches the adapter's sum-over-active masked reduction
    )

    # Adapter under test, fed the model's own epsilon.
    epsilon = torch.cat(eps_used, dim=0).to(dev)
    with torch.no_grad():
        mine = policy.denoise_error(
            net=policy.net,
            x0=x0,
            cond=cond,
            sigma=sigmas,
            epsilon=epsilon,
            condition_mask=cm,
        )

    a, b = mine.detach().float(), native.detach().float()
    torch.testing.assert_close(a, b, rtol=2e-3, atol=1e-4)


@pytest.mark.L1
@pytest.mark.GPU
def test_dpo_loss_ln2_at_init(policy: Cosmos3FlowPolicy) -> None:
    """With winner==loser and net_ref a frozen copy of net, the implicit reward margin
    is exactly 0 → loss = -log σ(0) = ln2, accuracy 0.5 (tie-aware)."""
    dev = torch.device("cuda")
    batch = _build_batch(dev)
    loss, metrics = dpo_objective(policy, batch, batch, beta=500.0)
    loss_val = float(loss.detach())
    assert math.isfinite(loss_val)
    assert abs(loss_val - math.log(2)) < 0.05, f"loss={loss_val} != ln2"
    assert abs(float(metrics["dpo_reward_margin"])) < 1e-3
    assert abs(float(metrics["dpo_accuracy"]) - 0.5) < 1e-6


def _build_image_batch(device: torch.device, B: int = 1, H: int = 256, W: int = 256, seed: int | None = None) -> dict:
    # T2I batch: the `images` key (not `video`) routes is_image_batch=True; each image is
    # [C,H,W] in a one-item vision list (OmniMoT augments it to a T=1 latent frame).
    g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    return {
        "ai_caption": ["a test image"] * B,
        "images": [[torch.randint(0, 256, (3, H, W), generator=g, dtype=torch.uint8).to(device)] for _ in range(B)],
        "image_size": [torch.tensor([[H, W, H, W]], dtype=torch.float32).to(device) for _ in range(B)],
        "conditioning_fps": [torch.tensor([24.0]).to(device).to(torch.bfloat16) for _ in range(B)],
    }


@pytest.mark.L1
@pytest.mark.GPU
def test_dpo_loss_image_t2i_path(policy: Cosmos3FlowPolicy) -> None:
    """T=1 image (T2I) path: the `images` key routes `is_image_batch=True` → the OmniMoT
    encodes a single latent frame (T=1, vs video's T>1) and the adapter uses
    `rectified_flow_image`. dpo_loss must run finite; winner==loser ⇒ ln2 at init."""
    dev = torch.device("cuda")
    batch = _build_image_batch(dev)
    x0, cond = policy.encode_and_condition(batch)
    cond = _require_cosmos3_condition(cond)
    assert cond.is_image_batch is True
    assert x0.shape[2] == 1, f"image latent must be T=1, got {tuple(x0.shape)}"  # vs video T>1
    loss, metrics = dpo_objective(policy, batch, batch, beta=500.0)
    loss_val = float(loss.detach())
    assert math.isfinite(loss_val)
    assert abs(loss_val - math.log(2)) < 0.05, f"loss={loss_val} != ln2"


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize("media", ["video", "image"], ids=["t2v", "t2i"])
def test_dpo_cross_impl_parity_vs_cosmos_rl(policy: Cosmos3FlowPolicy, monkeypatch, media) -> None:
    """Gate-3 — end-to-end cross-impl parity: my `dpo_loss` + adapter == cosmos-rl's
    reference `Cosmos3Model.compute_dpo_loss` on the SAME model, batch, and injected
    (sigma, epsilon). cosmos-rl is imported in the TEST ONLY (the production code never
    imports it). A cosmos-rl `Cosmos3Model` is built as a shell grafted onto our
    already-instantiated model (shared net + net_ref, no second random init).

    Both implementations are fed identical inputs:
      - one shared encode (via our adapter; cosmos-rl's own get_data_and_condition
        expects cosmos-rl's dataloader format, so we wrap the shared gen_data_clean in
        a cosmos-rl Cosmos3Condition);
      - one injected (sigma, epsilon): mine returns RF fraction t; cosmos-rl returns
        EDM sigma = t/(1-t) (compute_dpo_loss maps t_B = sigma/(sigma+1) back to t).
    cosmos-rl's compute_dpo_loss calls `omnimot.denoise(..., fps_vision=...)`, but the
    current OmniMoT denoise() has no fps_vision arg (fps moved into packing), so we wrap
    denoise to swallow it — numerically a no-op. A warmup pass stabilizes the
    attention-kernel autotuning so the grad (policy) forward is run-to-run deterministic
    (otherwise the 0.5·beta=250 factor amplifies cold-start kernel noise).

    Verified 2026-06-29 on GB200 (8B random-init): all 11 metrics bit-exact
    (|d|=0), loss 0.588283, real margin 0.222 (winner≠loser)."""
    import torch.nn as nn

    crl_mod = pytest.importorskip("cosmos_rl.policy.model.wfm.models.cosmos3")
    Cosmos3Model = crl_mod.Cosmos3Model
    Cosmos3Condition = crl_mod.Cosmos3Condition

    model = policy.omnimot
    dev = torch.device("cuda")

    # cosmos-rl shell grafted onto our model (shared weights by construction).
    crl = Cosmos3Model.__new__(Cosmos3Model)
    nn.Module.__init__(crl)
    crl.omnimot = model
    crl.net = model.net
    crl.net_ref = policy.net_ref
    crl.rectified_flow = model.rectified_flow_image if media == "image" else model.rectified_flow_video
    crl.tokenizer_vision_gen = model.tokenizer_vision_gen

    # winner != loser (different media, identical prompt/plan/fps). Video (T>1) and image
    # (T=1) exercise both `rectified_flow_video` and `rectified_flow_image` — compute_dpo_loss
    # routes on is_image_batch from the cond, so this covers both modality paths.
    mk = (lambda s: _build_batch(dev, seed=s)) if media == "video" else (lambda s: _build_image_batch(dev, seed=s))
    batch_w = mk(111)
    batch_l = mk(222)

    # One shared encode via our adapter; wrap into cosmos-rl's Cosmos3Condition.
    with torch.no_grad():
        x0_w, mc_w = policy.encode_and_condition(batch_w)
        x0_l, mc_l = policy.encode_and_condition(batch_l)
        mc_w = _require_cosmos3_condition(mc_w)
        mc_l = _require_cosmos3_condition(mc_l)
    B = int(x0_w.shape[0])

    def to_crl_cond(mc):
        return Cosmos3Condition(
            sequence_plans=mc.sequence_plans, gen_data_clean=mc.gen_data_clean, text_tokens=mc.text_tokens
        )

    condc_w, condc_l = to_crl_cond(mc_w), to_crl_cond(mc_l)

    # Injected shared noise: mine = RF fraction t; cosmos-rl = EDM sigma = t/(1-t).
    torch.manual_seed(7)
    t_B = torch.rand(B, device=dev) * 0.6 + 0.2  # t ∈ [0.2, 0.8]
    epsilon = torch.randn(x0_w.shape, device=dev, dtype=torch.float32)
    sigma_rf = t_B.view(B, 1)
    sigma_edm = (t_B / (1.0 - t_B)).view(B, 1)

    crl_cache = {id(batch_w): (None, x0_w, condc_w), id(batch_l): (None, x0_l, condc_l)}
    my_cache = {id(batch_w): (x0_w, mc_w), id(batch_l): (x0_l, mc_l)}
    monkeypatch.setattr(crl, "get_data_and_condition", lambda batch: crl_cache[id(batch)])
    monkeypatch.setattr(crl, "draw_training_sigma_and_epsilon", lambda size, cond: (sigma_edm, epsilon))
    monkeypatch.setattr(policy, "encode_and_condition", lambda batch: my_cache[id(batch)])
    monkeypatch.setattr(policy, "draw_sigma_epsilon", lambda shape, cond: (sigma_rf, epsilon))
    orig_denoise = model.denoise
    monkeypatch.setattr(
        model,
        "denoise",
        lambda net=None, data_batch_packed=None, memory=None, fps_vision=None: orig_denoise(
            net=net, data_batch_packed=data_batch_packed, memory=memory
        ),
    )

    beta = 500.0
    # Warmup (stabilize attention-kernel autotuning), then the measured comparison.
    _ = crl.compute_dpo_objective(batch_w, batch_l, beta=beta)
    _ = dpo_objective(policy, batch_w, batch_l, beta=beta)
    loss_crl, m_crl = crl.compute_dpo_objective(batch_w, batch_l, beta=beta)
    loss_mine, m_mine = dpo_objective(policy, batch_w, batch_l, beta=beta)

    torch.testing.assert_close(loss_mine.detach().float(), loss_crl.detach().float(), rtol=1e-3, atol=2e-3)
    for k in [
        "dpo_loss",
        "dpo_reward_margin",
        "dpo_reward_winner",
        "dpo_reward_loser",
        "dpo_accuracy",
        "dpo_err_pol_w",
        "dpo_err_ref_w",
        "dpo_err_pol_l",
        "dpo_err_ref_l",
        "dpo_raw_model_loss",
        "dpo_raw_ref_loss",
    ]:
        a = torch.as_tensor(m_mine[k]).detach().float()
        b = torch.as_tensor(m_crl[k]).detach().float()
        torch.testing.assert_close(a, b, rtol=1e-3, atol=2e-3, msg=f"metric '{k}' mismatch: mine={a} crl={b}")
    # The gate is cross-impl PARITY (asserted above). We do NOT assert a specific
    # margin magnitude: at init net_ref ≈ net, so the margin is the amplified
    # (0.5·beta) policy-vs-ref grad/no_grad kernel gap — noise that varies with
    # warmup state (~0.005..0.2), not a content signal. Both impls just have to agree.
    assert math.isfinite(float(m_mine["dpo_reward_margin"]))


@pytest.mark.L1
@pytest.mark.GPU
def test_dpo_loss_with_preference_dataset(policy: Cosmos3FlowPolicy, tmp_path) -> None:
    """End-to-end real-data path: `DPOPreferenceDataset` → `DatasetPreferenceEnv` →
    `move_to_device` (the trainer's data-fetch) → `unpack_preferences` → `dpo_loss` on
    the real OmniMoT. Exercises the raw-`ai_caption` tokenization branch of
    `encode_and_condition` (the dataset yields captions, not pre-tokenized
    `text_token_ids`), the dataset batch format, and the production host→device mover."""
    import json

    from cosmos_framework.posttrain.data.datasets.dpo_preference import DPOPreferenceDataset
    from cosmos_framework.posttrain.data.packer.preference import unpack_preferences
    from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv
    from cosmos_framework.posttrain.roles.trainer.validation.preference import move_to_device

    def _save(path, seed):
        g = torch.Generator().manual_seed(seed)
        torch.save({"frames": torch.randint(0, 256, (3, 9, 256, 256), generator=g, dtype=torch.uint8)}, path)

    win, los = tmp_path / "w.pt", tmp_path / "l.pt"
    _save(win, 1)
    _save(los, 2)
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "a test video", "winner_video": str(win), "loser_video": str(los), "score_gap": 1.0})
        + "\n"
    )

    ds = DPOPreferenceDataset(str(jsonl), resolution="256", num_frames=9, cfg_dropout_rate=0.0)
    env = DatasetPreferenceEnv(ds)
    trajs = move_to_device(env.get_samples(1), torch.device("cuda"))  # trainer data-fetch
    winner_batch, loser_batch = unpack_preferences(trajs)
    # No pre-tokenized text → encode_and_condition must tokenize the raw caption.
    assert "text_token_ids" not in winner_batch and "ai_caption" in winner_batch
    assert winner_batch["video"][0][0].is_cuda  # mover landed tensors on the device

    loss, metrics = dpo_objective(policy, winner_batch, loser_batch, beta=500.0)
    assert math.isfinite(float(loss.detach()))
    for k in ("dpo_loss", "dpo_reward_margin", "dpo_err_pol_w", "dpo_err_ref_l"):
        assert k in metrics
