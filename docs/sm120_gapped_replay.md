# Experimental SM120 gapped forward replay

This change admits exact SM120 for native gapped **forward** replay only when
FlashAttention-4 (FA4) has the reviewed invalid-work-tile fix and forward API.
It is an experimental review candidate, not a claim of qualified SM120 training
or production support. The current public candidate still needs GPU qualification.

## Why changing the architecture check alone is unsafe

There were two separate failures:

1. The Framework admission check rejected `(12, 0)` before native replay could run
   (`cosmos_framework/model/generator/mot/maskless_attention.py:429`).
2. After admission, an FA4 scheduler-padding work tile could index beyond the
   variable-length batch metadata. The kernel must check `work_tile.is_valid_tile`
   before constructing sequence information. FA4 [PR #2763][fix] adds that guard.
   Stock b24 and b28 predate it; b30 contains it.

The upstream guard precedes `SeqlenInfoQK.create`
([fixed source, `flash_attn/cute/flash_fwd.py:793`][fixed-source]); compare the
[older unguarded source at line 787][old-source]. The historical SM120 reproduction
reported 17 memcheck errors before the companion kernel fix. An architecture-only
patch is therefore insufficient.

## Scope and qualification

The public base is `a2a7409efd6b8559193245d142be03eb57f17819`. This candidate
requires exact stock `4.0.0b30`, complete dependencies and the explicit forward
API. Its four manual GPU cases and native model inference have not yet been run.
Backward/training is explicitly rejected on SM120.

Earlier downstream tests used a different Framework baseline and FA4 b28 with
the PR #2763 source backport. Those results do not satisfy this exact-b30 gate
and must not be relabeled as validation of this candidate. Other execution paths
are likewise not qualification evidence for this native Framework candidate.

Existing SM90/SM10 admission and attention mathematics are unchanged. SM121 and
other architectures remain rejected. Generic Flash4, block-sparse attention,
Transfer-Auto, LiDAR generation and model-quality acceptance are outside this
change. SM120 forward execution is checked both at availability admission and at
the custom-op entry; SM120 backward fails before loading the native backend.

## Dependency prerequisite

The helper at
`cosmos_framework/model/generator/mot/maskless_attention_compat.py:31` checks
distribution metadata and the private `_flash_attn_fwd` signature, including the
ten explicit arguments used by replay. Metadata is not source attestation.
Missing metadata, local-version strings, older releases, future releases and
incompatible signatures fail closed.

This patch deliberately does **not** globally replace the repository's b24 pins
or rewrite its lockfile: doing so would change existing Hopper/B200 environments
without regression evidence. A default b24 installation remains rejected on SM120.

For qualification, build a separate SM120 environment with exact b30 and its
complete dependency set. Preserve the chosen Torch/CUDA constraints and resolve
dependencies together. The [b30 manifest][dependencies] requires, among others,
`nvidia-cutlass-dsl>=4.6.2`, `torch-c-dlpack-ext`,
`apache-tvm-ffi>=0.1.12,<0.2` and `quack-kernels>=0.5.3`. Simply replacing the FA4
wheel with `--no-deps` in the b24 environment is not a supported procedure.
Run `python -m pip check`, archive the resolved environment, and use that
environment's Python directly so an automatic project sync does not restore b24.

The b30 forward API is reviewed at
[`flash_attn/cute/interface.py:525`][forward-api]. The version gate does not prove
the entire dependency stack, arbitrary shapes or backward are correct.

## Verification

CPU tests exercise real Framework admission with mocked device observations and
the CUDA-free compatibility helper. They do not execute FA4 or establish GPU
correctness. In an existing dependency-complete CPU test environment, from the
repository root:

```sh
CUDA_VISIBLE_DEVICES='' COSMOS_TRAINING=0 COSMOS_INTERNAL=0 PYTHONPATH="$PWD" \
  python -m pytest -c /dev/null --rootdir "$PWD" \
  --confcutdir "$PWD/cosmos_framework/model/generator/mot" \
  -p no:cacheprovider \
  cosmos_framework/model/generator/mot/maskless_attention_compat_test.py \
  cosmos_framework/model/generator/mot/maskless_attention_test.py
```

The isolated configuration intentionally avoids the repository-wide GPU setup
fixtures. This is a focused test run, not the full repository suite.

On an authorized, exclusively idle SM120 GPU with the prepared b30 environment,
the manual regression is
`cosmos_framework/model/generator/mot/maskless_attention_sm120_gpu_test.py:49`:

```sh
.venv-sm120/bin/python -m pytest --manual --levels=0 --num-gpus=1 \
  cosmos_framework/model/generator/mot/maskless_attention_sm120_gpu_test.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
  .venv-sm120/bin/python -m pytest --manual --levels=0 --num-gpus=1 \
  cosmos_framework/model/generator/mot/maskless_attention_sm120_gpu_test.py
compute-sanitizer --tool initcheck --error-exitcode 99 \
  .venv-sm120/bin/python -m pytest --manual --levels=0 --num-gpus=1 \
  cosmos_framework/model/generator/mot/maskless_attention_sm120_gpu_test.py
```

All four cases must execute, not skip. Require zero sanitizer errors and archive
the exact GPU, Torch/CUDA, FA4 and complete dependency identities with logs.
Cases cover BF16/FP16, GQA4, seven variable-length views, partial tiles, finite
gap perturbations, input immutability, output/LSE agreement against an independent
FP32 reference, and foreign-view isolation. The two history lengths are synthetic
shapes, **not** a rolling-cache or complete model test.

Before qualification, also repeat a bounded native Sim-Auto inference smoke with
complete video decoding and obtain SM90/SM10 regression results. No new GPU jobs
were submitted while preparing this patch. Merge readiness remains pending those
checks and maintainer approval of the isolated dependency policy.

[fix]: https://github.com/Dao-AILab/flash-attention/pull/2763
[fixed-source]: https://github.com/Dao-AILab/flash-attention/blob/9d61d35ba876834539af7df369fbabef1ad1e5d7/flash_attn/cute/flash_fwd.py#L793
[old-source]: https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/flash_fwd.py#L787
[forward-api]: https://github.com/Dao-AILab/flash-attention/blob/9d61d35ba876834539af7df369fbabef1ad1e5d7/flash_attn/cute/interface.py#L525
[dependencies]: https://github.com/Dao-AILab/flash-attention/blob/9d61d35ba876834539af7df369fbabef1ad1e5d7/flash_attn/cute/pyproject.toml#L31
