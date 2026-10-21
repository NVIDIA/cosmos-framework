# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VLLMRollout (DESIGN §1.2 body=(a); §8.2 #6 engine integration).

Needs a GPU and the `vllm` package (the vLLM venv: vllm==0.19.1 / torch==2.10.0), so it is
skip-guarded and marked L1/GPU (vLLM engine init > 30s). Verifies the Task #3 MVP contract
(RL_PLAN_AND_NOTES_zh.md §1.2): group integrity + per-token logprob (π_old)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")

from cosmos_framework.posttrain.roles.rollout import VLLMRollout
from cosmos_framework.posttrain.schema import ObsBundle, Status

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def llm():
    return vllm.LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enforce_eager=True,
    )


@pytest.mark.L1
@pytest.mark.GPU
def test_generate_returns_group_trajectories_with_logprob(llm):
    rollout = VLLMRollout(
        llm, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4}, packer=init_data_packer(MODEL)
    )
    prompts = [
        ObsBundle(representations={"prompt": "1 + 1 =", "reference": 2}, id="p0"),
        ObsBundle(representations={"prompt": "2 + 3 =", "reference": 5}, id="p1"),
    ]
    G = 3
    trajs = rollout.generate(prompts, group=G)

    # Group integrity: exactly P*G trajectories in P complete groups (§2.3). group_id is the
    # prompt's own id (obs.id) so groups never collide across replicas at fan-in.
    assert len(trajs) == 2 * G
    assert {t.group_id for t in trajs} == {"p0", "p1"}
    assert sum(t.group_id == "p0" for t in trajs) == G
    for t in trajs:
        assert t.status == Status.COMPLETED
        tr = t.transitions[0]
        assert tr.action is not None
        assert "completion_text" in tr.policy_info
        assert "prompt_ids" in tr.policy_info
        # π_old: one logprob per generated token (§2.4), non-empty, all finite floats.
        lp = tr.policy_info["rollout_logprobs"]
        assert isinstance(lp, list) and len(lp) == len(tr.action)
        assert len(lp) > 0
        assert all(isinstance(x, float) and x == x for x in lp)


@pytest.mark.L1
@pytest.mark.GPU
def test_apply_weights_bumps_version_noop(llm):
    # Packer is REQUIRED at construction now; this check is weight-only (never generates).
    rollout = VLLMRollout(llm, packer=init_data_packer(MODEL), sampling={"temperature": 1.0, "max_tokens": 2})
    assert rollout._weight_version == 0
    rollout.apply_weights()
    assert rollout._weight_version == 1  # static-weights seam: version bump only (§8.2 #4)


# --- L0 packer wiring tests (no vLLM engine) ---
_torch = pytest.importorskip("torch")
_transformers = pytest.importorskip("transformers")

from cosmos_framework.posttrain.data.packer import init_data_packer


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_accepts_packer_and_runs_guard():
    packer = init_data_packer("gpt2")
    r = VLLMRollout(model=object(), sampling={"max_tokens": 4}, packer=packer)  # fake engine; no generate
    assert r.packer is packer  # guard passed (gpt2 adds no special tokens)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_warns_but_accepts_bos_tokenizer(monkeypatch):
    """A BOS-adding tokenizer must WARN, not refuse.

    It used to raise, on the premise that vLLM tokenizes internally while the objective
    re-tokenizes. That premise died with the stored-ids preference: this rollout records
    `req.prompt_token_ids` — vLLM's OWN tokenization — and the objective scores exactly those, so
    the prefix cannot drift whatever the tokenizer adds. Refusing only blocked `llama`, which
    `data/packer/__init__.py` explicitly REGISTERS. The warning stays for the narrow residual risk:
    a future consumer that re-tokenizes `obs` instead of using the stored ids.

    Spies on the module's logger rather than using `caplog`: `utils.log` sets
    `logger.propagate = False`, so pytest's root handler never sees these records."""
    from cosmos_framework.posttrain.roles.rollout import vllm as vllm_mod

    warnings_seen: list[str] = []
    monkeypatch.setattr(vllm_mod.logger, "warning", lambda msg, *a, **k: warnings_seen.append(str(msg)))

    packer = init_data_packer("gpt2")

    class _BosTok:
        def __call__(self, s, add_special_tokens=True, **kw):
            class _R:
                pass

            r = _R()
            r.input_ids = ([1] if add_special_tokens else []) + [1, 2]
            return r

    packer.tokenizer = _BosTok()
    assert packer.adds_special_tokens() is True  # precondition: this tokenizer really does add one
    rollout = VLLMRollout(model=object(), sampling={"max_tokens": 4}, packer=packer)
    assert rollout.packer is packer  # constructed, not refused
    assert any("adds special tokens" in m for m in warnings_seen), f"expected a warning, got {warnings_seen}"


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_requires_packer_at_construction():
    # The contract lives in __init__, NOT generate(): a packer-less rollout used to construct, bind
    # and spawn every Ray actor fine and only die on the first generate() with the cluster fully up.
    # Omitted argument ⇒ TypeError; an explicit packer=None ⇒ our ValueError with the remedy.
    with pytest.raises(TypeError):
        VLLMRollout(model=object(), sampling={"max_tokens": 4})
    with pytest.raises(ValueError, match="requires a DataPacker at construction"):
        VLLMRollout(model=object(), packer=None, sampling={"max_tokens": 4})


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_params_after_model_are_keyword_only():
    # The 2nd positional slot changed meaning (sampling -> packer), so a stale positional call must
    # raise NOW rather than silently binding a sampling dict as the packer.
    with pytest.raises(TypeError):
        VLLMRollout(object(), {"max_tokens": 4})


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_input_accepts_both_str_and_dict():
    """The text packer returns a prompt string; the VLM packer returns
    {"prompt":…, "multi_modal_data":…}. vLLM takes both, but the rollout must not stringify a dict.

    `generate` runs the batch through `_training_request_identity` inside its rank-converged
    preflight, BEFORE `model.generate`. The contract accepts either a string request or a mapping
    carrying exactly one of `prompt` and `prompt_token_ids`, while preserving multimodal identity.
    """
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    identity = VLLMRollout._training_request_identity

    class _Obs:
        def __init__(self, sample_id: str) -> None:
            self.id = sample_id

    prompts = [_Obs("a"), _Obs("b")]
    sampling = {"max_tokens": 4}

    # A bare string and a multimodal mapping are both accepted. The mapping must carry ordered
    # content UUIDs for its images: `_CONTENT_IDENTITY_REQUIRED_MODALITIES` makes that unconditional
    # for the image modality, because the rank-convergence proof compares WHICH images each rank
    # sent, and a bare tensor list cannot answer that.
    mm = {
        "prompt": "hello",
        "multi_modal_data": {"image": ["IMG"]},
        "multi_modal_uuids": {"image": ["img-0"]},
    }
    proof = identity(prompts, ["hello", mm], sampling, 2)
    assert proof["group"] == 2 and proof["sampling"] == sampling
    assert [p["sample_id"] for p in proof["requests"]] == ["a", "b"]
    assert proof["requests"][0]["prompt"] == "hello"
    assert proof["requests"][1]["prompt"] == "hello"

    # Image media WITHOUT ordered UUIDs is refused. `HFVLMDataPacker.get_rollout_input` emits
    # content-addressed UUIDs, so this covers malformed custom packers and hand-built requests.
    with pytest.raises(ValueError, match="requires ordered content UUIDs"):
        identity(prompts[:1], [{"prompt": "hi", "multi_modal_data": {"image": ["IMG"]}}], sampling, 1)

    # A mapping with NEITHER prompt nor prompt_token_ids is refused -- the case the old validator
    # caught -- and so is one carrying both, which it could not express.
    with pytest.raises(RuntimeError, match="exactly one of prompt or prompt_token_ids"):
        identity(
            prompts[:1], [{"multi_modal_data": {"image": ["IMG"]}, "multi_modal_uuids": {"image": ["u"]}}], sampling, 1
        )
    with pytest.raises(RuntimeError, match="exactly one of prompt or prompt_token_ids"):
        identity(prompts[:1], [{"prompt": "hi", "prompt_token_ids": [1, 2]}], sampling, 1)

    # A non-str, non-mapping request is refused rather than stringified.
    with pytest.raises(TypeError, match="str or mapping"):
        identity(prompts[:1], [1234], sampling, 1)

    # A request count that disagrees with the prompt count is a shaping bug, not a silent zip.
    with pytest.raises(RuntimeError, match="request"):
        identity(prompts, ["only-one"], sampling, 1)


@pytest.mark.L1
@pytest.mark.GPU
def test_action_keeps_the_stop_token_that_completion_text_drops(llm):
    """`action` is the ENGINE's ids and KEEPS the stop token; re-tokenizing `completion_text` loses it.

    This pins the divergence measured against cosmos-rl (docs/design/grpo-cosmos-rl-parity.md). vLLM omits
    the stop token from `.text` while keeping it in `.token_ids` unless BOTH
    `skip_special_tokens=False` AND `include_stop_str_in_output=True` are set — neither of which is
    on by default. A consumer that re-tokenizes the text therefore trains one token short on every
    stopped completion, which was the ENTIRE residual loss gap between the two stacks. Scoring
    `action` is what makes this rollout immune, so the immunity is asserted rather than assumed.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    # An EXPLICIT stop token, not the model's EOS: this 0.5B never emits EOS inside the 512-token
    # context (measured — it runs to the cap at max_tokens=440), so keying on EOS makes the test
    # skip itself and guard nothing. `.` fires within ~24 tokens and exercises the identical vLLM
    # path: the stop token lands in `token_ids` and is withheld from `.text`.
    stop_id = tok(".", add_special_tokens=False).input_ids[0]
    rollout = VLLMRollout(
        llm,
        sampling={"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "stop_token_ids": [stop_id]},
        packer=init_data_packer(MODEL),
    )
    obs = ObsBundle(representations={"prompt": "Reply with exactly one word: hello", "reference": "hello"}, id="p0")
    trajs = rollout.generate([obs], group=1)
    assert trajs, "rollout produced no trajectories"

    for t in trajs:
        tr = t.transitions[0]
        ids = [int(x) for x in tr.action]
        assert len(ids) < 64, "completion hit the length cap, so no stop token was generated to test"
        retok = tok(tr.policy_info["completion_text"], add_special_tokens=False).input_ids
        assert ids[-1] == stop_id, "the scored action must retain the stop token the engine emitted"
        # The exact failure mode: text re-tokenization reproduces the completion MINUS its stop
        # token. If this ever stops holding, the parity conclusion needs re-measuring.
        assert retok == ids[:-1], "completion_text must not be used as a stand-in for action"
