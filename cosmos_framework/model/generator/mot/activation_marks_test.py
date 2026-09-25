# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Call-site marking for selective activation checkpointing.

Runs against ``cosmos_framework.model.attention`` with varlen arguments, which is what the
decomposed multiview attention calls and what selects NATTEN's varlen FMHA on
Blackwell -- a ``torch.library`` custom op inductor cannot decompose, rather than
an aten op it can. ``F.scaled_dot_product_attention`` would exercise neither.

The compiled cases carry the weight: three cheaper marking mechanisms work eager
and fail under ``torch.compile``, which is how a decoder layer runs, so an
eager-only suite would pass on all of them.
"""

import gc
import re

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from cosmos_framework.model.attention import attention
from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.model.generator.mot.activation_marks import (
    MARK_OP_QUALNAME,
    enable_marking,
    is_mark_op,
    mark_next_activation,
    marking_enabled,
    reset_marking_for_tests,
)
from cosmos_framework.model.generator.mot.parallelize_unified_mot import make_selective_ac_policy

HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
HIDDEN = HEADS * HEAD_DIM
TOKENS = 4096
SEGMENTS = 2
SEGMENT = TOKENS // SEGMENTS


def _policy(
    decisions: list[tuple[str, bool]], save_ops: tuple[str, ...] = ("fmha",), save_only_marked_ops: bool = True
):
    """The shipped policy, wrapped to record what it decided.

    Deliberately not a reimplementation. An earlier version of this file had one,
    it consulted the mark only on attention ops while the real policy consulted it
    on every op, and it hid a defect where the mark landed on the gather that
    precedes the kernel rather than on the kernel.
    """
    real = make_selective_ac_policy([re.compile(pattern) for pattern in save_ops], save_only_marked_ops)

    def policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        verdict = real(ctx, func, *args, **kwargs)
        name = getattr(func, "__name__", str(func))
        decisions.append((name, verdict == CheckpointPolicy.MUST_SAVE))
        return verdict

    return policy


@pytest.fixture(autouse=True)
def _marking_off_by_default():
    """Every test states its own precondition; the switch is process-level.

    Without this a test that enables marking would leave it on for the rest of the
    session, and the cases that check what happens *without* marks would stop
    checking anything.
    """
    reset_marking_for_tests()
    yield
    reset_marking_for_tests()


def _natten_can_run_this_block() -> bool:
    """Whether NATTEN's varlen FMHA is usable here for the shapes ``_Block`` runs.

    The blocks pin ``backend="natten"`` rather than letting the frontend choose, because
    the policy matches ops by name and only NATTEN's kernels are named "fmha". NATTEN is
    the frontend's pick for varlen on Blackwell but not on Hopper, where flash3 outranks
    it -- pinning is what keeps this coverage on both. Pinning an *incompatible* backend
    raises, though, so compatibility is checked here and the suite skips rather than
    errors on a device NATTEN cannot serve.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from cosmos_framework.model.attention.backends import is_backend_compatible

        return is_backend_compatible(
            backend="natten",
            query_shape=torch.Size((1, TOKENS, HEADS, HEAD_DIM)),
            key_shape=torch.Size((1, TOKENS, KV_HEADS, HEAD_DIM)),
            value_shape=torch.Size((1, TOKENS, KV_HEADS, HEAD_DIM)),
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            requires_grad=True,
            is_causal=False,
            causal_type=None,
            is_varlen=True,
            raise_error=False,
        )
    except Exception:  # noqa: BLE001 - a gate that cannot answer should skip, not error.
        return False


_NATTEN_RUNS_HERE = _natten_can_run_this_block()


def _natten_is_the_frontend_choice() -> bool:
    """Whether the frontend would *pick* NATTEN here, not merely whether it can run.

    A different question from ``_natten_can_run_this_block``, and the peak-memory
    comparison below is the one place the distinction matters: what the blocks pin is
    NATTEN's kernel, but what the surrounding cost of recomputing looks like is the
    arch's, and those diverge.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from cosmos_framework.model.attention.backends import choose_backend

        return (
            choose_backend(
                query_shape=torch.Size((1, TOKENS, HEADS, HEAD_DIM)),
                key_shape=torch.Size((1, TOKENS, KV_HEADS, HEAD_DIM)),
                value_shape=torch.Size((1, TOKENS, KV_HEADS, HEAD_DIM)),
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
                requires_grad=True,
                is_causal=False,
                causal_type=None,
                is_varlen=True,
                raise_error=False,
            )
            == "natten"
        )
    except Exception:  # noqa: BLE001 - a gate that cannot answer should skip, not error.
        return False


_NATTEN_IS_THE_FRONTEND_CHOICE = _natten_is_the_frontend_choice()


def _saved_attention(decisions: list[tuple[str, bool]]) -> list[bool]:
    return [saved for name, saved in decisions if "fmha" in name or "attention" in name]


def _saved_names(decisions: list[tuple[str, bool]]) -> list[str]:
    return [name for name, saved in decisions if saved]


class _Block(nn.Module):
    """Three varlen attention calls; ``mark_at`` says which one marks itself.

    Three because the decomposed layer runs the same kernel several times over
    different folds, which is the case a name-matching policy cannot separate.
    """

    def __init__(self, mark_at: int | tuple[int, ...] | None) -> None:
        super().__init__()
        self.marks = () if mark_at is None else (mark_at,) if isinstance(mark_at, int) else tuple(mark_at)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Derived from the input rather than fixed at construction, so a second token
        # count retraces the region instead of failing the view.
        tokens = x.shape[0]
        segment = tokens // SEGMENTS
        offsets = torch.arange(SEGMENTS + 1, dtype=torch.int32, device=x.device) * segment
        for index in range(3):
            q = x.view(1, tokens, HEADS, HEAD_DIM)
            kv = q[:, :, :KV_HEADS]
            if index in self.marks:
                kv = mark_next_activation(kv)
            out = attention(
                q,
                kv,
                kv,
                cumulative_seqlen_Q=offsets,
                cumulative_seqlen_KV=offsets,
                max_seqlen_Q=segment,
                max_seqlen_KV=segment,
                backend="natten",
            )
            x = out.reshape(tokens, HIDDEN)
        return x


class _InterveningBlock(nn.Module):
    """Marks call 1, then evaluates its other operands -- as the real call site does."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offsets", torch.arange(SEGMENTS + 1, dtype=torch.int32) * SEGMENT, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(3):
            base = x.view(1, TOKENS, HEADS, HEAD_DIM)
            kv = base[:, :, :KV_HEADS]
            if index == 1:
                kv = mark_next_activation(kv)
            # Gathers and slices dispatched *after* the mark, before the kernel.
            gather = torch.arange(TOKENS, device=x.device)
            q = base[:, gather]
            v = base[:, gather][:, :, :KV_HEADS]
            out = attention(
                q,
                kv,
                v,
                cumulative_seqlen_Q=self.offsets,
                cumulative_seqlen_KV=self.offsets,
                max_seqlen_Q=SEGMENT,
                max_seqlen_KV=SEGMENT,
                backend="natten",
            )
            x = out.reshape(TOKENS, HIDDEN)
        return x


def _run(
    mark_at: int | tuple[int, ...] | None,
    compiled: bool,
    save_ops: tuple[str, ...] = ("fmha",),
    save_only_marked_ops: bool = True,
    block: nn.Module | None = None,
) -> tuple[list[tuple[str, bool]], torch.Tensor]:
    torch._dynamo.reset()
    decisions: list[tuple[str, bool]] = []
    # Where ``_apply_selective_ac`` does it: before anything is traced. Inside
    # ``context_fn`` would be too late under compile, since the region is traced before
    # the partitioner calls it -- the marker would already have been a pass-through, and
    # the two compiled cases that check marking fail exactly that way.
    enable_marking()
    block = ptd_checkpoint_wrapper(
        (block if block is not None else _Block(mark_at)).cuda(),
        context_fn=lambda: create_selective_checkpoint_contexts(_policy(decisions, save_ops, save_only_marked_ops)),
    )
    if compiled:
        block = torch.compile(block, fullgraph=True)
    torch.manual_seed(0)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    block(x).float().sum().backward()
    torch.cuda.synchronize()
    return decisions, x.grad.clone()


@pytest.mark.L0
@pytest.mark.CPU
def test_the_attention_op_regex_covers_every_backend_that_can_be_selected() -> None:
    """The marked policy's eligibility list, checked against the ops actually registered.

    A marked call site is only kept if some op matches ``save_ops_regex``, so a backend
    missing from that list makes marking silently inert -- the config says it is on, no
    op is eligible, the mark is never consumed and the layer recomputes everything. Not
    hypothetical: ``["fmha"]`` covers NATTEN on every arch but nothing else, and on sm90
    flash3 is ranked ahead of NATTEN and takes these calls, which is how the mechanism
    came to do nothing on Hopper while reading as enabled.

    Enumerated from the dispatcher rather than hard-coded, so a renamed or newly added
    kernel fails here instead of in a training run's memory profile. cuDNN is out of
    scope by design: it rejects varlen and every marked call site is varlen.
    """
    from cosmos_framework.model.attention.backends import get_backend_list
    from cosmos_framework.configs.base.defaults.activation_checkpointing import (
        ATTENTION_FORWARD_OPS_REGEX,
    )

    patterns = [re.compile(pattern) for pattern in ATTENTION_FORWARD_OPS_REGEX]

    def covered(op_name: str) -> bool:
        # What the policy matches on: ``func.__name__``, which drops the namespace.
        return any(pattern.search(f"{op_name.split('::')[-1]}.default") for pattern in patterns)

    # Backend name -> the namespace its ops register under. ``None`` means no marked call
    # site can reach it: cuDNN rejects varlen and every mark sits on a varlen fold.
    namespaces = {
        "natten": "natten::",
        "flash2": "flash_attn::",
        "flash3": "flash_attn_3::",
        "flash4": "flash_attn_4::",
        "cudnn": None,
    }
    # Every backend the frontend can return, on any arch it supports. Driving the check
    # from here rather than a hardcoded list is the point: ``flash4`` is already installed
    # in the GB200 image but commented out of ``get_backend_list``, and the day it is
    # enabled it ranks ahead of NATTEN on sm100 -- at which point marking would go inert
    # on the primary training arch exactly as it did on Hopper, and this should say so.
    selectable = {backend for arch in (75, 80, 86, 90, 100, 103, 110, 120, 121) for backend in get_backend_list(arch)}
    assert selectable, "no backend is selectable anywhere, so this would pass vacuously"
    unknown = selectable - set(namespaces)
    assert not unknown, f"a backend was added without saying where its ops live: {sorted(unknown)}"

    registered = torch._C._dispatch_get_all_op_names()
    prefixes = tuple(namespaces[backend] for backend in selectable if namespaces[backend])
    # NATTEN also registers neighbourhood kernels (na1d/na2d/na3d), which serve
    # ``multi_dimensional_attention`` rather than the decomposed folds. Nothing marks into
    # those, so they are out of scope: the folds dispatch full attention.
    neighbourhood = ("na1d", "na2d", "na3d")

    def in_scope(name: str) -> bool:
        return name.startswith(prefixes) and not any(kernel in name for kernel in neighbourhood)

    forwards = [name for name in registered if in_scope(name) and name.endswith("_forward")]
    # flash3 ships as ``flash_attn_3_nv`` and is absent from aarch64 images, where it is
    # not a candidate anyway. Its op name is pinned by docker/Dockerfile.base and read from
    # that tag's source, so cover it whether or not this image has the package.
    if "flash3" in selectable:
        forwards.append("flash_attn_3::_flash_attn_forward")

    assert forwards, "no attention backend registered an op, so this would pass vacuously"
    missing = [name for name in forwards if not covered(name)]
    assert not missing, f"selectable backends the marked policy cannot keep: {missing}"

    # Backward ops must not match: the policy runs over the forward, and a pattern loose
    # enough to catch them is matching on something other than what it means.
    caught = [name for name in registered if in_scope(name) and name.endswith("_backward") and covered(name)]
    assert not caught, f"the regex reaches backward ops: {caught}"


@pytest.mark.L0
@pytest.mark.CPU
def test_the_schema_default_is_the_regex_the_test_above_validates() -> None:
    """Every selective config inherits the list checked against the dispatcher, not a subset.

    The default used to be ``["fmha"]``, which is NATTEN's op name rather than attention's:
    it matched on sm100, where cuDNN and flash2 both reject varlen and NATTEN wins the call,
    and matched nothing on sm90, where flash3 is ranked first and takes it. So a config
    asking for selective AC got it on one arch and full recompute on the other, silently
    and with the same ``mode="selective"`` in both. The constant is what makes that
    arch-independent, and it only helps the configs that get it by default -- a config
    naming its own list is a config that can be wrong about the backend again.

    Copied per instance rather than shared: ``attrs`` hands the factory's value to the
    config, so a returned module constant would let one config's ``append`` rewrite the
    policy of every other config built in the same process.
    """
    from cosmos_framework.configs.base.defaults.activation_checkpointing import (
        ATTENTION_FORWARD_OPS_REGEX,
        ActivationCheckpointingConfig,
    )

    assert ActivationCheckpointingConfig().save_ops_regex == ATTENTION_FORWARD_OPS_REGEX

    mutated = ActivationCheckpointingConfig()
    mutated.save_ops_regex.append("randn")
    assert ActivationCheckpointingConfig().save_ops_regex == ATTENTION_FORWARD_OPS_REGEX
    assert "randn" not in ATTENTION_FORWARD_OPS_REGEX


@RunIf(min_gpus=1)
@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(
    not _NATTEN_RUNS_HERE, reason="These blocks pin NATTEN's varlen FMHA, which this device cannot run."
)
class TestActivationMarks:
    """Marking one call among several that dispatch the same kernel."""

    @pytest.mark.parametrize("compiled", [False, True])
    def test_only_the_marked_call_is_saved(self, compiled: bool) -> None:
        """The point of the mechanism, and the compiled case is the one that matters.

        The decomposed attention runs four identical FMHA ops per layer, so a
        name-matching policy keeps all of them or none.
        """
        decisions, _ = _run(mark_at=1, compiled=compiled)
        assert _saved_attention(decisions)[:3] == [False, True, False], _saved_names(decisions)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_an_unmarked_block_saves_nothing(self, compiled: bool) -> None:
        """Under ``save_only_marked_ops`` an eligible op still needs a mark."""
        decisions, _ = _run(mark_at=None, compiled=compiled)
        assert _saved_names(decisions) == [], _saved_names(decisions)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_without_save_only_marked_ops_the_regex_decides_alone(self, compiled: bool) -> None:
        """The default, and what every existing config relies on.

        Marks are inert here: the regex keeps every op it matches, so a marked
        block and an unmarked one save the same set.
        """
        marked, _ = _run(mark_at=1, compiled=compiled, save_only_marked_ops=False)
        unmarked, _ = _run(mark_at=None, compiled=compiled, save_only_marked_ops=False)
        assert _saved_attention(marked)[:3] == [True, True, True], _saved_names(marked)
        assert _saved_attention(unmarked)[:3] == [True, True, True], _saved_names(unmarked)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_a_mark_is_not_taken_by_ops_between_it_and_the_kernel(self, compiled: bool) -> None:
        """A pending mark waits for an *eligible* op, not the next op of any kind.

        The real call site evaluates the gathers for Q and V after the mark line,
        and the attention frontend clones internally, so four ops sit between the
        marker and the kernel -- index, index, slice, clone. Consuming on the next
        op of any kind put the mark on a gather and saved that instead.
        """
        decisions, _ = _run(mark_at=1, compiled=compiled, block=_InterveningBlock())
        assert _saved_attention(decisions)[:3] == [False, True, False], _saved_names(decisions)
        assert all("fmha" in name or "attention" in name for name in _saved_names(decisions)), _saved_names(decisions)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_marking_does_not_change_gradients(self, compiled: bool) -> None:
        """Marking decides what is kept, never what is computed.

        Compared against a floor the test measures rather than a fixed tolerance:
        SDPA's backward is not deterministic, so the same configuration run twice
        already differs, and bitwise equality would be unreachable for reasons
        that have nothing to do with marking.
        """
        _, plain = _run(mark_at=None, compiled=compiled)
        _, plain_again = _run(mark_at=None, compiled=compiled)
        _, marked = _run(mark_at=1, compiled=compiled)

        assert (plain != 0).any(), "a zero gradient would compare equal and prove nothing"
        scale = plain.abs().max()
        # Two floors, because either alone is misleading. The run-to-run spread
        # catches a non-deterministic backward, but two runs can land bit-identical
        # and report zero; bf16 round-off at this scale is the floor underneath
        # that. Both were measured at 6.25e-02 on the NATTEN path, which is ~1.3e-3
        # relative -- bf16's mantissa, not a difference marking caused.
        observed = (plain - plain_again).abs().max()
        floor = torch.maximum(observed * 2, scale * 2**-8)
        difference = (marked - plain).abs().max()
        assert difference <= floor, f"marked differs by {difference}; run-to-run {observed}, bf16 floor {scale * 2**-8}"

    @pytest.mark.parametrize("compiled", [False, True])
    def test_each_mark_is_consumed_by_exactly_one_call(self, compiled: bool) -> None:
        """Two marked call sites keep two calls, and the unmarked one between them is not kept.

        A mark that armed the policy without disarming it would keep every eligible op
        after the first mark, which looks identical to working when only one call is
        marked. Marking calls 0 and 2 is what separates the two.
        """
        decisions, _ = _run(mark_at=(0, 2), compiled=compiled)
        assert _saved_attention(decisions)[:3] == [True, False, True], _saved_names(decisions)

    def test_marks_do_not_leak_between_regions_under_compile(self) -> None:
        """The eager leak check, compiled -- where the policy runs at trace time.

        Two blocks share nothing but the process. If policy state outlived a region, the
        second block's calls would inherit the first block's marking, and under compile
        that mistake is baked into a graph rather than made once per step.

        Each block records into its own list, because the compiled region is traced more
        than once and a single list interleaves the passes.
        """
        torch._dynamo.reset()
        enable_marking()
        marked_block: list[tuple[str, bool]] = []
        plain_block: list[tuple[str, bool]] = []
        blocks = [
            torch.compile(
                ptd_checkpoint_wrapper(
                    _Block(marks).cuda(),
                    context_fn=lambda decisions=decisions: create_selective_checkpoint_contexts(_policy(decisions)),
                ),
                fullgraph=True,
            )
            for marks, decisions in ((2, marked_block), (None, plain_block))
        ]
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        blocks[1](blocks[0](x)).float().sum().backward()
        torch.cuda.synchronize()

        folds = _saved_attention(marked_block)
        assert folds and len(folds) % 3 == 0, f"expected whole passes over three calls, saw {len(folds)}"
        assert folds == [False, False, True] * (len(folds) // 3), _saved_names(marked_block)
        assert _saved_names(plain_block) == [], _saved_names(plain_block)

    def test_marking_holds_across_repeated_steps(self) -> None:
        """What is kept must not drift once training is under way.

        The policy carries state -- a pending-mark flag -- and ``context_fn`` is called
        once per checkpointed forward. A flag left armed at the end of one step would
        keep the *first* call of the next one, silently and only from step two, which a
        single-step test cannot see.

        Eager, because that is where there is something per-step to watch: under compile
        the policy is consulted while tracing and not again, so later steps record no
        decisions at all. The compiled equivalent of this claim is that the graph does
        not drift, which ``test_a_compiled_marked_block_is_stable_across_steps`` checks
        through its gradients and its memory instead.
        """
        torch._dynamo.reset()
        enable_marking()
        decisions: list[tuple[str, bool]] = []
        block = ptd_checkpoint_wrapper(
            _Block(1).cuda(),
            context_fn=lambda: create_selective_checkpoint_contexts(_policy(decisions)),
        )
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for step in range(4):
            decisions.clear()
            block(x).float().sum().backward()
            x.grad = None
            torch.cuda.synchronize()
            assert _saved_attention(decisions)[:3] == [False, True, False], f"step {step}: {_saved_names(decisions)}"

    def test_a_compiled_marked_block_is_stable_across_steps(self) -> None:
        """A compiled marked block repeats itself exactly, and holds nothing between steps.

        Under compile the policy is consulted once while tracing, so what a later step
        does can only be read off its results. Gradients that stay put say the graph is
        not being re-partitioned differently, and flat memory says the kept activation
        and the marker's copy are released with the graph rather than accumulating.

        Against a measured floor rather than bitwise, for the reason
        ``test_marking_does_not_change_gradients`` gives: NATTEN's backward is not
        deterministic, so two runs of one configuration already differ slightly.
        """
        torch._dynamo.reset()
        enable_marking()
        block = torch.compile(
            ptd_checkpoint_wrapper(
                _Block(1).cuda(),
                context_fn=lambda: create_selective_checkpoint_contexts(_policy([])),
            ),
            fullgraph=True,
        )
        torch.manual_seed(0)
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        block(x).float().sum().backward()
        first = x.grad.clone()
        x.grad = None
        block(x).float().sum().backward()
        # The same step twice: whatever these differ by is the backward's own noise, and
        # anything marking did wrong later has to clear it to be visible.
        floor = torch.maximum((x.grad - first).abs().max() * 2, first.abs().max() * 2**-8)
        x.grad = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        resident = []
        for step in range(3):
            block(x).float().sum().backward()
            drift = (x.grad - first).abs().max()
            assert drift <= floor, f"step {step} drifted by {drift}, floor {floor}"
            x.grad = None
            torch.cuda.synchronize()
            gc.collect()
            resident.append(torch.cuda.memory_allocated() - baseline)
        assert len(set(resident)) == 1, f"memory accumulated across steps: {resident}"

    def test_marking_survives_a_recompilation(self) -> None:
        """A second shape retraces the region, and the new graph must mark it too.

        The switch is read while Dynamo traces, so it is a guard on the compiled graph
        rather than a test run per step. Packing gives a different token count from step
        to step, so retracing is routine here rather than exotic, and a retrace that
        dropped the mark would quietly go back to recomputing the expensive fold.

        The graph count is asserted as well, so the test cannot pass by never retracing
        -- a dynamic-shape graph reused for both sizes would check nothing.
        """
        torch._dynamo.reset()
        enable_marking()
        decisions: list[tuple[str, bool]] = []
        block = torch.compile(
            ptd_checkpoint_wrapper(
                _Block(1).cuda(),
                context_fn=lambda: create_selective_checkpoint_contexts(_policy(decisions)),
            ),
            fullgraph=True,
        )
        graphs = []
        for tokens in (TOKENS, TOKENS * 2):
            decisions.clear()
            x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            block(x).float().sum().backward()
            torch.cuda.synchronize()
            graphs.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            assert _saved_attention(decisions)[:3] == [False, True, False], (
                f"{tokens} tokens: {_saved_names(decisions)}"
            )
        assert graphs[1] > graphs[0], f"the second shape reused the first graph, so nothing retraced: {graphs}"

    def test_the_marker_passes_its_gradient_through_unchanged(self) -> None:
        """The marker is bookkeeping, so its backward is the identity.

        Registered by hand, and a marker that scaled or dropped a gradient would move
        the model's without any test of what is *kept* noticing.
        """
        enable_marking()
        source = torch.randn(64, 32, device="cuda", dtype=torch.float32, requires_grad=True)
        seed = torch.randn_like(source)
        mark_next_activation(source).backward(seed)
        assert source.grad is not None
        assert torch.equal(source.grad, seed)

    def test_a_forward_without_a_backward_keeps_nothing_resident(self) -> None:
        """Inference steps interleave with training ones, and must not pin activations.

        Under ``no_grad`` there is no checkpoint to save into, so the marked fold and
        the marker's own copy should both be gone once the forward returns.
        """
        torch._dynamo.reset()
        enable_marking()
        block = ptd_checkpoint_wrapper(
            _Block(1).cuda(),
            context_fn=lambda: create_selective_checkpoint_contexts(_policy([])),
        )
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            block(x)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        resident = []
        for _ in range(3):
            with torch.no_grad():
                block(x)
            torch.cuda.synchronize()
            gc.collect()
            resident.append(torch.cuda.memory_allocated() - baseline)
        assert set(resident) == {0}, f"an inference step pinned memory: {resident}"

    @pytest.mark.skipif(
        not _NATTEN_IS_THE_FRONTEND_CHOICE,
        reason="Peak-memory ordering at this block size is an arch property; measured on sm100.",
    )
    def test_marking_costs_less_memory_than_saving_every_eligible_op(self) -> None:
        """The arithmetic the mechanism exists for, counting the clone it pays for.

        Keeping the one marked call has to cost less than keeping all three, or the
        mechanism buys nothing -- and the marker's copy is inside the measurement, so
        this is the net figure rather than the saving before its cost.

        Only against saving everything, and only where the ordering has been measured.
        Recomputing is not reliably the cheapest of the three: its transient forward
        buffers are live alongside the backward's, which at this block's size puts its
        peak above both others while on the production shape it sits below them.

        The same effect inverts this test's own claim on sm90, where it was measured at
        168.5 MiB marked against 160.6 MiB saving all three -- marking retains ~14 MiB
        less (one output plus the K clone, against three outputs) but pays more than that
        back in workspace for the two forwards it recomputes. So the bytes comparison is
        a property of arch and shape rather than of marking, and is asserted only where
        it was measured. What marking does to the *number* of kept activations is
        arch-independent and covered by ``test_only_the_marked_call_is_saved``.
        """
        peaks = {}
        for label, mark_at, save_only in (("marked", 1, True), ("save_all", None, False)):
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _run(mark_at=mark_at, compiled=True, save_only_marked_ops=save_only)
            torch.cuda.synchronize()
            peaks[label] = torch.cuda.max_memory_allocated()

        assert peaks["marked"] < peaks["save_all"], peaks

    def test_a_mark_binds_to_the_next_op_not_an_index(self) -> None:
        """A call site behind a branch must not shift the mark onto its neighbour.

        ``cross_view`` is conditional and ``gen_to_und`` has two forms, so the
        number of attention calls varies per step. An ordinal policy would drift;
        binding to the next op in trace order does not.
        """
        for mark_at in (0, 1, 2):
            decisions, _ = _run(mark_at=mark_at, compiled=False)
            expected = [index == mark_at for index in range(3)]
            assert _saved_attention(decisions)[:3] == expected, f"mark at {mark_at}: {_saved_names(decisions)}"

    def test_the_regex_still_gates_what_a_mark_can_keep(self) -> None:
        """A mark cannot keep an op the regex does not make eligible."""
        decisions, _ = _run(mark_at=1, compiled=False, save_ops=("nothing_matches_this",))
        assert _saved_names(decisions) == [], _saved_names(decisions)

    def test_without_save_only_marked_ops_every_eligible_op_is_kept(self) -> None:
        """The pre-existing behaviour, unchanged."""
        decisions, _ = _run(mark_at=None, compiled=False, save_only_marked_ops=False)
        assert _saved_attention(decisions)[:3] == [True, True, True], _saved_names(decisions)

    def test_the_marker_does_not_leak_between_regions(self) -> None:
        """Policy state is per region, so a mark cannot arm the next block's first op."""
        torch._dynamo.reset()
        enable_marking()
        decisions: list[tuple[str, bool]] = []
        blocks = [
            ptd_checkpoint_wrapper(
                _Block(2 if index == 0 else None).cuda(),
                context_fn=lambda: create_selective_checkpoint_contexts(_policy(decisions)),
            )
            for index in range(2)
        ]
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        nn.Sequential(*blocks)(x).float().sum().backward()
        torch.cuda.synchronize()
        # First block marks its last call; the second block marks nothing.
        assert _saved_attention(decisions)[:6] == [False, False, True, False, False, False], _saved_names(decisions)

    def test_the_clone_is_transient(self) -> None:
        """The copy the marker makes must not survive the step.

        It is MUST_RECOMPUTE, so a checkpointed region rebuilds it during recompute
        rather than storing it.
        """
        torch._dynamo.reset()
        enable_marking()
        decisions: list[tuple[str, bool]] = []
        block = ptd_checkpoint_wrapper(
            _Block(0).cuda(),
            context_fn=lambda: create_selective_checkpoint_contexts(_policy(decisions)),
        )
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        block(x).float().sum().backward()
        x.grad = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        resident = []
        for _ in range(3):
            block(x).float().sum().backward()
            x.grad = None
            torch.cuda.synchronize()
            gc.collect()
            resident.append(torch.cuda.memory_allocated() - baseline)
        assert len(set(resident)) == 1, f"memory accumulated across steps: {resident}"

    def test_the_op_is_registered_under_its_advertised_name(self) -> None:
        namespace, name = MARK_OP_QUALNAME.split("::")
        assert hasattr(getattr(torch.ops, namespace), name)
        assert is_mark_op(f"{name}.default")
        assert not is_mark_op("blackwell_fmha_forward.default")

    def test_marking_returns_an_equal_tensor(self) -> None:
        """A copy, because inductor rejects a custom op whose output aliases its input."""
        enable_marking()
        source = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        marked = mark_next_activation(source)
        assert torch.equal(marked, source)
        assert marked.data_ptr() != source.data_ptr()

    def test_marking_is_free_until_a_policy_asks_for_it(self) -> None:
        """The copy is the whole cost of a marked call site, so an unmarked model pays none.

        Every model reaching ``multiview_maskless_attention`` runs its mark line, but
        only the configs setting ``save_only_marked_ops`` can do anything with a mark.
        Off, the marker hands its argument straight back -- the same tensor, not a copy
        of it -- so the line costs nothing rather than a clone per layer per step.
        """
        assert marking_enabled() is False, "the fixture should leave this off"
        source = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        assert mark_next_activation(source) is source

        enable_marking()
        assert marking_enabled() is True
        assert mark_next_activation(source) is not source

    @pytest.mark.parametrize(
        "mode,save_only_marked_ops,expected",
        [
            pytest.param("selective", True, True, id="selective_marked"),
            pytest.param("selective", False, False, id="selective_plain"),
            pytest.param("full", True, False, id="full"),
        ],
    )
    def test_wrapping_a_module_turns_marking_on_from_its_config(
        self, mode: str, save_only_marked_ops: bool, expected: bool
    ) -> None:
        """The switch and the policy come from one config field, so they cannot disagree.

        A policy that consults marks while the call sites emit none keeps *nothing* --
        a silent memory regression rather than an error -- so the two are set together
        where the module is wrapped rather than left to a caller to keep in step.

        ``mode="full"`` recomputes the whole block and has no policy to consult, so the
        flag buys nothing there and the clone is not worth paying for. Driven through
        ``apply_ac_to_module``, the entry point the model actually calls, so the
        dispatch is covered along with the switch.

        Each case starts from off, which is what the fixture guarantees. What happens
        when it starts from *on* is a separate question with a separate answer, in
        ``test_a_later_unmarked_model_cannot_turn_marking_off``.
        """
        from cosmos_framework.configs.base.defaults.activation_checkpointing import (
            ActivationCheckpointingConfig,
        )
        from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_ac_to_module

        config = ActivationCheckpointingConfig(mode=mode, save_only_marked_ops=save_only_marked_ops)
        apply_ac_to_module(nn.Identity(), config)
        assert marking_enabled() is expected

    def test_a_later_unmarked_model_cannot_turn_marking_off(self) -> None:
        """The switch latches, and that is the decision rather than an oversight.

        One process can build several networks -- an EMA copy, a distillation teacher --
        and a second one wrapped without marks must not disarm the first. Dynamo guards
        on the switch, so turning it off invalidates the first model's compiled code on
        its very next forward, and the recompiled graph has no marker in it; a policy
        built with ``save_only_marked_ops`` then keeps nothing and the model silently
        reverts to recomputing every fold.

        The cost of latching is the opposite mistake: a later maskless model pays a clone
        it cannot use. That one is bounded -- the marker is ``MUST_RECOMPUTE``, so it is
        transient rather than resident -- and it is the cheaper of the two by more than
        an order of magnitude.
        """
        from cosmos_framework.configs.base.defaults.activation_checkpointing import (
            ActivationCheckpointingConfig,
        )
        from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_ac_to_module

        apply_ac_to_module(nn.Identity(), ActivationCheckpointingConfig(mode="selective", save_only_marked_ops=True))
        assert marking_enabled() is True

        for config in (
            ActivationCheckpointingConfig(mode="selective"),
            ActivationCheckpointingConfig(mode="full"),
        ):
            apply_ac_to_module(nn.Identity(), config)
            assert marking_enabled() is True, f"{config.mode} disarmed a model that needs marks"

        # And no production caller can reach the off path: it takes no argument.
        import inspect

        assert not inspect.signature(enable_marking).parameters, "enable_marking must not take a value"
