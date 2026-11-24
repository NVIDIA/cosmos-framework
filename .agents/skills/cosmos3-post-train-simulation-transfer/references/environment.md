# Environment requirements and troubleshooting

> **Skill note (not in the source):** this file keeps the training-only parts of the Sim-Transfer environment reference: the sentence on Nano conversion that ended its first paragraph and the section "Tokenizer compilation". The system requirements and the sections "Complete the CUDA 13.0 environment" and "Diagnostics and common failures" are in the Sim-Transfer inference skill's [environment reference](../../cosmos3-inference-simulation-transfer/references/environment.md). "The Conda installation above" in "Tokenizer compilation" is the Conda block of [Complete the CUDA 13.0 environment](../../cosmos3-inference-simulation-transfer/references/environment.md#complete-the-cuda-130-environment).

Nano conversion retains its separately pinned checkout and environment; see
[conversion](nano-conversion.md).

## Tokenizer compilation

The training `compile_tokenizer` callback uses AOTInductor. Compute hosts need
`nvcc`, CUDA runtime/CRT/CCCL headers, a C++ compiler and a valid `CUDA_HOME`.
For the Conda installation above, point `CUDA_HOME` at `cuda-toolkit`.

The callback's `compile_after_iterations=3` compiles on the fourth step in each
process. Share `IMAGINAIRE_OUTPUT_ROOT` across nodes so ranks can exchange the
compiled packages; follow the [multi-node launch](multinode-training.md).
