# Modular Dataflow — Tutorial + Cleanup (Plan 6 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

> **EXECUTION ORDER: LAST.** Do not execute the cleanup tasks (Tasks 2–4) until ALL mirror experiments (VLM, videophy2, VFM) have passed golden-batch + loss-curve equivalence from Plans 2/4/5. Deleting the legacy baseline before equivalence is confirmed removes the only ground truth. The docs task (Task 1) can be drafted earlier but is finalized here.

**Goal:** Ship `docs/dataflow.md` (the bring-your-own-dataset tutorial), promote the validated `*_v2` mirror experiments to canonical names, and delete the legacy dataloaders + superseded experiments in one cleanup PR.

**Architecture:** Two phases. (1) Documentation — the user-facing tutorial + cross-links. (2) Cleanup — promote mirrors to default names, delete legacy code (`joint_dataloader.py` family, `data_packer.py`, `packing_iterable_dataset.py`, the legacy `DataPackerDataLoader`/`JointDataPackerDataLoader` + private wrappers, the `VLMDataPacker`/`VideoPhy2DataPacker`), and fix every import that referenced deleted code.

**Tech Stack:** Markdown, Python, pytest, grep-driven import fixes.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md` ("Documentation deliverable", "File layout", Impl phases 5–6). Builds on Plans 1–5.

> **HARD INVARIANT:** Deletion must not break resume/saving for the promoted recipes. After promotion, the promoted (formerly `_v2`) recipes use the new loader, which already passed the resume integration test (Plan 3) and the loss-curve regressions. Re-run the resume test + a short regression after deletion to confirm nothing regressed.

---

## File Structure

| File | Change |
|---|---|
| `docs/dataflow.md` | create — the tutorial |
| `AGENTS.md`, `docs/code_structure.md`, docs index, `cosmos3-post-training` skill | add a link to `docs/dataflow.md` |
| `examples/toml/sft_config/*_v2.toml`, `examples/launch_sft_*_datapacker.sh` | promote to canonical names |
| `cosmos_framework/configs/base/**/experiment/*_v2*.py` | promote to canonical experiment names |
| `cosmos_framework/data/vfm/joint_dataloader.py`, `data_packer.py`, `packing_iterable_dataset.py`, `data_packer_dataloader.py` | delete (or reduce to nothing) |
| call sites importing the above | update imports to `dataflow.*` |

---

### Task 1: Write `docs/dataflow.md` (tutorial)

**Files:**
- Create: `docs/dataflow.md`

- [ ] **Step 1: Write the tutorial following the spec outline**

Write `docs/dataflow.md` with these sections (spec "Documentation deliverable"):
1. **Mental model** — the diagram `DataDistributor → RawItemProcessor → SampleBatcher → BatchCollator`, one sentence each.
2. **Quickstart (60s)** — runnable:
   ```python
   from cosmos_framework.data.vfm.dataflow import (
       CosmosDataLoader, MapDistributor, IdentityProcessor,
   )
   loader = CosmosDataLoader(
       distributor=MapDistributor(my_dataset, shuffle=True, seed=0),
       processor=IdentityProcessor(),
       batch_size=32,
   )
   ```
3. **The four roles** — each: purpose, ABC signature (copy from `dataflow/base.py`), built-ins, a minimal custom example.
4. **Recipes by use-case** — map dataset (`MapDistributor` + shuffle + resume), iterable (`IterableDistributor`), token-budget packing (`PoolPackingBatcher` + `size_fn`), order-preserving packing (`SequentialPackingBatcher`), ratio mixing (`MixtureDistributor`), heterogeneous interleave (`JointCosmosDataLoader`).
5. **Wiring into a training recipe** — Hydra `LazyCall` block + CLI overrides (`dataloader_train.batcher.max_tokens=...`).
6. **Checkpoint / resume** — `MapDistributor` + `DataLoaderStateCallback(distributor_type="data_packer")`; iterable = non-resumable.
7. **Distributed & sharding** — DP rank vs workers, `parallel_dims`, disjoint coverage, `name=` namespacing for joint loaders.
8. **Troubleshooting / FAQ** — OOM → `apply_long_sample_halving`; modality mixing in pool packing; `num_workers`/`persistent_workers` rules; oversized-sample discard.
9. **End-to-end worked example** — a complete local image-caption folder dataset → custom `RawItemProcessor` → `CosmosDataLoader` → a runnable training command.

Use the live `*_v2` recipes (VLM/videophy2/VFM) as linked real-world examples.

- [ ] **Step 2: Cross-link**

Add a one-line link to `docs/dataflow.md` in: `AGENTS.md` (key file locations),
`docs/code_structure.md` (data subpackage tour), the docs index/README docs
table, and the `cosmos3-post-training` skill's doc list. Verify each link path.

- [ ] **Step 3: Verify examples import-clean**

Run a doctest-style smoke: extract the quickstart snippet into a scratch test and
run it against a tiny in-memory dataset to confirm the public imports resolve.
Run: `python -c "from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, MapDistributor, IdentityProcessor, IterableDistributor, SimpleBatcher, PoolPackingBatcher, SequentialPackingBatcher, RankPartitionedDistributor, MixtureDistributor, JointCosmosDataLoader, VFMListCollator, DefaultBatchCollator"`
Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git add docs/dataflow.md AGENTS.md docs/code_structure.md
git commit -m "docs: add docs/dataflow.md bring-your-own-dataset tutorial + cross-links"
```

---

### Task 2: Promote `*_v2` mirrors to canonical names

**Files:**
- experiments, TOMLs, launch wrappers

> Gate: only after all three loss-curve regressions pass.

- [ ] **Step 1: Promote experiments**

For each recipe (`pre_exp012_llava_ov_datapacker`, `videophy2_sft_nano`,
`vision_sft_nano`): replace the original experiment module's dataloader wiring
with the `_v2` four-role wiring (move the `_v2` body into the canonical module and
delete the `_v2` module), so the canonical experiment name now uses
`CosmosDataLoader`. Update the registration name back to the canonical name.
Delete the `*_v2_test.py` registration smokes (the canonical names are tested by
the existing recipe tests).

- [ ] **Step 2: Promote TOMLs + launch wrappers**

Fold each `*_v2.toml` content into the canonical TOML (`[job].experiment` back to
the canonical name; keep the new loader wiring via the promoted experiment). Delete
the `*_v2.toml` and the temporary `launch_sft_*_datapacker.sh` wrappers (the
canonical `launch_sft_*.sh` now drives the new loader unchanged). Keep
`logging_iter`/`max_iter`/wandb as the recipes had them pre-refactor (do NOT bake
the regression overrides into the canonical recipe).

- [ ] **Step 3: Run the recipe registration + golden tests**

Run: `pytest cosmos_framework/data/vfm/dataflow/ -v`
Expected: PASS (golden tests still compare against legacy until Task 3 deletes it —
so run Task 3 only after this passes).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: promote dataflow mirror recipes to canonical names"
```

---

### Task 3: Delete legacy dataloaders

**Files:**
- Delete: `cosmos_framework/data/vfm/joint_dataloader.py`, `cosmos_framework/data/vfm/data_packer.py`, `cosmos_framework/data/vfm/packing_iterable_dataset.py`, `cosmos_framework/data/vfm/data_packer_dataloader.py` (and their `*_test.py`)
- Delete: `VLMDataPacker`/`VideoPhy2DataPacker` and their now-unused helpers in the experiment modules
- Update: every import of the deleted symbols

- [ ] **Step 1: Find every reference**

Run:
```bash
grep -rn "joint_dataloader\|data_packer_dataloader\|packing_iterable_dataset\|\bDataPacker\b\|VLMDataPacker\|VideoPhy2DataPacker\|JointDataPackerDataLoader\|RankPartitionedDataLoader\|PackingDataLoader\|custom_collate_fn" cosmos_framework examples | grep -v "/dataflow/"
```
Expected: a list of call sites to update or delete. The golden tests in
`dataflow/golden_*_test.py` import the legacy loaders — these tests have served
their purpose (equivalence proven); convert them to frozen-fixture tests OR delete
them (the spec chose "delete in cleanup"; per-role unit tests + resume test remain
the permanent guard).

- [ ] **Step 2: Delete the legacy modules + update imports**

Delete the four legacy modules + their tests. Fix any remaining imports to point at
`cosmos_framework.data.vfm.dataflow`. Remove `VLMDataPacker`/`VideoPhy2DataPacker`
from the experiment modules (the processors/collators in `dataflow_roles.py` /
`videophy2_dataflow_roles.py` replace them).

- [ ] **Step 3: Delete the legacy golden tests**

Remove `dataflow/golden_vlm_test.py`, `golden_videophy2_test.py`, `golden_vfm_test.py`
(they import the deleted legacy loaders). The per-role unit tests + `resume_test.py`
are the permanent regression suite.

- [ ] **Step 4: Run the full suite**

Run: `pytest cosmos_framework/data/vfm/dataflow/ -v` and the broader
`pytest cosmos_framework/ -x -q` (or the repo's standard test command) to catch
any missed import.
Expected: PASS, no ImportError referencing deleted modules.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy dataloaders (joint_dataloader, data_packer*, packing_iterable_dataset)"
```

---

### Task 4: Post-deletion verification run

- [ ] **Step 1: Resume regression**

Run: `pytest cosmos_framework/data/vfm/dataflow/resume_test.py -v`
Expected: PASS (resume invariant holds post-cleanup).

- [ ] **Step 2: Short end-to-end smoke on the promoted recipes**

For one promoted recipe (e.g. VLM), launch a short run (`max_iter=20`,
`logging_iter=1`, wandb `cosmos_oss_alignment`, fresh name) via the canonical
`launch_sft_*.sh` + `cosmos3-run-env` + `slurm-node`. Confirm it trains and
checkpoints without error. (Loss-curve equivalence was already proven pre-deletion;
this only confirms the canonical recipe still runs.)

- [ ] **Step 3: Final commit / PR**

```bash
git add -A
git commit -m "chore: dataflow refactor cleanup complete; CosmosDataLoader canonical"
```
Open the PR summarizing: new four-role abstraction, three migrated recipes with
loss-curve-equivalence run URLs, resume parity, legacy deletion.

---

## Self-Review

**Spec coverage:** `docs/dataflow.md` (spec "Documentation deliverable") → Task 1. Promote mirrors (Impl phase 5) → Task 2. Delete legacy (`joint_dataloader.py` family, `data_packer.py`, `packing_iterable_dataset.py`, private wrappers; spec "File layout" deletion list) → Task 3. Post-deletion resume + smoke (HARD INVARIANT) → Task 4. Execution-last gate stated up front.

**Placeholder scan:** Task 1 is a writing task with a concrete section-by-section outline + code snippets; Tasks 2–4 are mechanical with exact grep commands and file lists. No vague items.

**Type consistency:** All public symbols referenced (`CosmosDataLoader`, `MapDistributor`, `IdentityProcessor`, `IterableDistributor`, `SimpleBatcher`, `PoolPackingBatcher`, `SequentialPackingBatcher`, `RankPartitionedDistributor`, `MixtureDistributor`, `JointCosmosDataLoader`, `VFMListCollator`, `DefaultBatchCollator`) match the definitions in Plans 1–5.

**Ordering guard:** Tasks 2–4 must not run before Plans 2/4/5 regressions pass (stated at top). Task 3 deletes the golden tests' dependency, so Task 3 strictly follows Task 2's green run.
