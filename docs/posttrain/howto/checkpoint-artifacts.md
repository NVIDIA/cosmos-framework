# Checkpoint artifacts: publish, verify, fetch, prune, materialize

Use this guide when a `checkpoint_artifacts/iter_*` directory needs to be archived, fetched to local
disk, released from local storage, or converted into the Hugging Face directory an OPD teacher loads.

A checkpoint artifact contains selected model, optimizer or training-state components and a native
`.posttrain-checkpoint-artifact.json` manifest. It is not an exact-resume checkpoint. The trainer
writes an artifact for every partial component plan; expert merge writes the same layout offline.
Full resumable checkpoints remain under `checkpoints/`, with their own validation publication and
latest/last-N retention rules ([`checkpoint.md`](../reference/checkpoint.md)).

The shared publication library is [`checkpoint/archive/publisher.py`](../../../cosmos_framework/posttrain/checkpoint/archive/publisher.py).
Training and offline artifact publication use its same `publish_artifact` operation. Archive reads
and verification live in [`archive/reader.py`](../../../cosmos_framework/posttrain/checkpoint/archive/reader.py), marker schemas in
[`archive/marker.py`](../../../cosmos_framework/posttrain/checkpoint/archive/marker.py), and deletion policy in
[`retention.py`](../../../cosmos_framework/posttrain/checkpoint/retention.py). The commands below live under `checkpoint/tools/`.

Every command imports the checkpoint package, which imports torch. Run inside the training
container. [`utils/object_store.py`](../../../cosmos_framework/posttrain/utils/object_store.py) remains the torch-free login-node
tool for moving whole prefixes. The as-built architecture is recorded in
the internal design record.

## Produce an artifact

**Trainer.** A partial component plan writes an artifact at every save. With upload enabled, the
trainer also archives it:

```bash
rl.checkpoint_components=[model] \
rl.upload.enabled=true rl.upload.bucket=<bucket> rl.upload.credentials=<json> \
rl.upload.prefix=<project>/<group>/<run>/<ts>/checkpoints
```

The archive lands at `s3://<bucket>/<project>/<group>/<run>/<ts>/checkpoint_artifacts/iter_*/`.
[`artifact.py`](../../../cosmos_framework/posttrain/checkpoint/artifact.py) owns the native manifest and local commit;
[`archive/publisher.py`](../../../cosmos_framework/posttrain/checkpoint/archive/publisher.py) owns the archive. Both consolidated
files and DCP shards receive SHA-256 verification. Ordinary trainer publication catches archive
failures and keeps the local artifact; the offline command reports them as a failed command.

**No cold restart over an artifact.** `rl.resume=auto` cannot resume from a model-only artifact, and
the entrypoint refuses to fall back to a step-0 restart inside an attempt directory that already holds
committed `checkpoint_artifacts/iter_*` (`utils/entry.py::pin_run_attempt`): that restart would collide
with the committed iteration at its first save, and under Slurm requeue it did so in a loop. A resumable
full checkpoint under the same attempt's `checkpoints/` still resumes normally when its iteration is at or
after the newest committed artifact; one behind it (a full save sparser than the artifact schedule) is
refused for the same reason, since resuming it would replay up to that artifact and collide one save later
(an explicit legacy resume path without an `iter_*` segment is placed by its validated trainer state,
`optimizer_step`/`iter` in `trainer.pt` or at the top level of a single-file checkpoint; one whose iteration
cannot be established that way is refused too, since nothing proves it is at or beyond the newest artifact).
A fresh attempt and an interrupted artifact directory without the commit-last manifest are unaffected. The refusal names the
attempt directory and the remedies, each an operator decision: continue from the artifact as a NEW run
For a public run, use a new `job.name`, set `rl.init_from` to the committed model artifact, advance
the source cursor by the exact number of consumed examples, and start with fresh optimizer state.
Alternatively, set `rl.resume=off` to mint a fresh attempt beside the original run, or save the full
component plan when exact resume across requeues is required. Nothing automatically resumes a run
from a model-only artifact.

**Expert merge.** [`checkpoint/tools/merge.py`](../../../cosmos_framework/posttrain/checkpoint/tools/merge.py) merges consolidated
experts (`torch.save({"net": state_dict})`, bf16, Hugging Face key names), streaming one tensor at a
time:

```bash
# uniform average of the experts (the base cancels)
python -m cosmos_framework.posttrain.checkpoint.tools.merge --mode avg \
  --experts <a>/model.pt,<b>/model.pt,<c>/model.pt \
  --out <run>/<ts>/checkpoint_artifacts/iter_000300/model.pt

# task-vector sum onto the HF base, with per-tensor delta diagnostics
python -m cosmos_framework.posttrain.checkpoint.tools.merge --mode tv --lam 1.0 --base <hf-base-dir> \
  --experts <a>/model.pt,<b>/model.pt,<c>/model.pt \
  --out <run>/<ts>/checkpoint_artifacts/iter_000300/model.pt --diag <diag.json>
```

`--out` must be `<...>/iter_*/model.pt`; an explicit `--iteration N` must match the directory name.
Expert order is part of the request identity. The tool writes and verifies `model.pt`, writes the
diagnostics and native manifest, then writes `.merge-complete.json` last. The same request skips on a
valid completed output; an incomplete or damaged output is rebuilt. A completed different request
is refused unless `--force` is supplied. Records from the retired `merge_model_pt.py` have an older
tool version and require `--force` to rebuild.

## Publish, verify, prune

[`checkpoint/tools/archive_cli.py`](../../../cosmos_framework/posttrain/checkpoint/tools/archive_cli.py) exposes the shared archive and
retention APIs:

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli publish \
  --artifact-dir <run>/<ts>/checkpoint_artifacts/iter_000300 \
  --upload-prefix <project>/<group>/<run>/<ts>/checkpoints \
  [--bucket <bucket>] [--credentials <json>] [--delete-local]

python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli verify \
  --artifact-dir <run>/<ts>/checkpoint_artifacts/iter_000300 [--credentials <json>]

python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli prune-verified \
  --paths <artifact-dir-or-model.pt>,<artifact-dir-or-model.pt>,... [--credentials <json>]
```

`--upload-prefix` is the run's `rl.upload.prefix`; the artifact is placed under the sibling
`checkpoint_artifacts/<iter>` prefix. `--bucket` and `--credentials` default to
`<checkpoint-experiments-bucket>` and `credentials/<credential>.secret`.

**`publish`** selects the native manifest's physical components, the merge completion and preserved
legacy-manifest records, and any diagnostic recorded by the merge completion whose path is inside
the artifact directory. It hashes those files, checks existing native descriptor digests, and adds
missing file-component digests. Other files, such as a stale unselected `optim.pt`, are not archived.
A payload that contradicts its manifest is refused. Publication then follows this order:

1. Create `.posttrain-archive.json` remotely, before uploading payloads. This immutable declaration
   records protocol `posttrain.artifact-archive.v1`, the native manifest digest, and every archived
   file's relative path, size and SHA-256, including the native manifest itself.
2. Upload and verify the declared payloads.
3. Commit `.posttrain-checkpoint-artifact.json` last, then verify the remote inventory and write the
   local `.uploaded` marker.

The declaration is created conditionally: S3 uses a boto3 `If-None-Match: *` request; GCS uses a
separate `GOOG4-HMAC-SHA256` XML PUT with `x-goog-if-generation-match: 0`, because mixing that
header into boto3's AWS4 request is rejected by GCS. Unsupported conditional writes fail closed. A retry with the same
declaration may resume; a different inventory is rejected before it can overwrite a payload. The
declaration is retained permanently, including after a failed attempt. Use another destination for
different contents; an abandoned declaration does not authorize replacing those contents.

The native manifest remains the completion record. A declaration without that manifest represents
an incomplete archive. An already committed archive must agree with the local inventory and pass
remote verification before it can be reused. `--delete-local` releases the native manifest's payload
components only after checking both local digests and the live remote copy; the local manifest and
marker remain. These rules apply equally to trainer-produced and merged artifacts.

The command prints a JSON report with `status` (`published`, `adopted` or `skipped`), destination,
inventory and removed payloads. A committed identical archive without a matching local receipt is
adopted; one already matching that receipt is skipped. Both paths still verify the remote objects.

**`verify`** checks the destination bound into `.uploaded` against the supplied credentials and
HEADs its recorded objects. Missing, changed or unverifiable objects fail verification. HEAD checks
size and stored SHA-256 metadata; it does not download the contents. Fetch performs the corresponding
content-level check.

**`prune-verified`** releases payloads only from `checkpoint_artifacts/<iter_*>`, keeping full
checkpoint retention separate. It requires a valid native manifest, a marker for the addressed
destination, coverage of the current local directory, matching local digests, and live remote
objects with the recorded sizes and digests. Missing digests, legacy markers, changed local bytes
or remote verification failures keep the copy with a reason. A remote object claiming the expected
digest with another size causes an error and keeps that artifact. Extra local files not covered by
the archive receipt also prevent pruning; publication does not silently include them to satisfy the
deletion gate. The report lists `removed`, `kept` and `freed_bytes`.

## Fetch

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli fetch \
  --artifact-uri s3://<bucket>/<...>/checkpoint_artifacts/iter_000300 --dst <stage-dir> [--credentials <json>]
```

The command prints `<stage-dir>/iter_000300`. It reads the native manifest and archive declaration
before downloading. The declaration supplies the authoritative inventory for the manifest's
physical components, including DCP shards: a shard missing from the live listing is still required.
The recorded merge diagnostic and merge/legacy records are archived but are not fetched as model components. Every
downloaded file is verified against its declared size and digest before the staged artifact is
marked complete.

A `.staged.json` ownership record is written before downloading; each file is written through a
`.part` temporary. Payloads precede the local native manifest, and the staging record is completed
last. A directory holding files without this tool's record is foreign and is never replaced or
deleted. Its own incomplete copy resumes where previously staged files still match; replacing its
own copy of another source removes only the files named in its record.

Legacy file artifacts without an archive declaration remain readable if each file's native
descriptor or object metadata supplies a digest. Legacy DCP artifacts without a declaration are
refused because a live listing cannot prove that every shard is present. This includes DCP artifacts
archived by the earlier trainer implementation. New trainer DCP archives use the same complete,
digest-bound inventory as offline publication.

Artifacts whose legacy file descriptors and object metadata both lack a digest are unverifiable.
A conflicting committed prefix is never overwritten to upgrade it; preserve the local copy or
publish it at a new destination.

## Materialize an OPD teacher

[`checkpoint/tools/materialize_hf.py`](../../../cosmos_framework/posttrain/checkpoint/tools/materialize_hf.py) converts a
consolidated artifact into the Hugging Face directory used by an OPD teacher
([`opd.md`](../guides/opd.md)). It owns source resolution, output identity, staging and the CLI.
[`checkpoint/conversion/hf.py`](../../../cosmos_framework/posttrain/checkpoint/conversion/hf.py) is the reusable conversion and
verification library: strict key accounting, fused MoE expert conversion, tokenizer/processor
sidecar copying and shard verification. It has no independent CLI.

```bash
# from a local artifact directory (trainer or merge output)
python -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf \
  --ckpt-dir <run>/<ts>/checkpoint_artifacts/iter_000300 --base <hf-base-dir> --out <teacher-dir> \
  [--expect-tensors N] [--min-tensors N] [--no-verify] [--force] [--unified-chat-template]

# from an archive: payloads are fetched only when a build is needed
python -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf \
  --artifact-uri s3://<bucket>/<...>/checkpoint_artifacts/iter_000300 --base <hf-base-dir> --out <teacher-dir> \
  [--stage-dir <dir>] [--keep-staged] [--credentials <json>]
```

The output is built in `<out>.tmp` and renamed to `<out>` after verification. Its `.materialized`
marker records conversion version `materialize_model_pt_to_hf/3`, source digest and size, base
identity, and shard inventory. An intact output of the same request skips. An incomplete output is
rebuilt; a completed output for a different source, base or conversion version requires `--force`
or another `--out`. Outputs from the retired materializer have an older version and require a
rebuild.

For an archived source, the native manifest's model descriptor supplies the source identity used
for the skip decision. A build fetches the artifact through `archive/reader.py` and checks its
contents. Staging defaults to `<out>.stage`; cleanup removes only paths this invocation created,
and `--keep-staged` retains them. `--no-verify` skips only torch-side tensor probing: identity,
index and sidecar checks still run. `--expect-tensors` defaults to `0` (no exact count) and
`--min-tensors` to `800`. `--unified-chat-template` (default off) writes both template files of the
export (`tokenizer_config.json`'s and `chat_template.json`'s `chat_template`) as the base's processor
template in its canonical form (the reasoning branch rendered trimmed), every other key unchanged, so a
request with or without tools renders the same reasoning form; the marker records the processor
template's sha, the shipped sha of each file and the one-hunk diff, `verify_sidecars` checks those two
files by rule instead of byte identity, and the option is part of the output's identity (an output
built with it is refused by a request without it, and the reverse, until `--force`). The supported base
ships its templates in those two JSON files and carries no template sidecar a loader prefers over them
(`chat_template.jinja`, `additional_chat_templates/`, a `chat_template` key in `processor_config.json`).
The materializer's preflight, the torch-free checks that read the base alone before anything is removed
or written, enforces that: such a base is refused in unified mode before anything is published, with or
without `--no-verify`, naming the file (a sidecar equal to the canonical text is allowed and recorded; the
directory is refused whatever it holds, empty included, by the one helper the preflight and
`verify_sidecars` share). The preflight runs before anything is written: `<out>.tmp` is created and the
overlay run only after it has passed, so a refused request leaves an existing output byte-identical; and
the construction is staged, so a failure after the preflight keeps it too (below). It
also parses every JSON sidecar of the base
(`tokenizer_config.json` and `chat_template.json` as objects), refuses in the default path an
`additional_chat_templates/` whose entries are not all regular files, and refuses an unreadable base with
a named reason instead of a traceback; runs on every base shard the index lists the safetensors
header check the output gets (`shard_status`: present, readable, header parsed, size = 8 + header + data, header
keys = the index's), because the overlay opens every shard in the staged build; requires every JSON sidecar to be a
JSON object and `chat_template.json`, when present, to carry a non-empty `chat_template` string (the loaders index
these after the overlay); validates the index's `weight_map` (a non-empty object of tensor name -> shard name); and
raises every refusal in the one message family `REFUSED before anything is published: <what was found; what the
staged build would have done with it>`; decodes every JSON file through ONE decoder (strict UTF-8
text, what the loaders and the materializer's own readers use in the staged build), so a UTF-8 byte-order mark in front of
valid JSON is refused by name instead of passing through `json.loads(bytes)`; and validates the safetensors header's shape
in `shard_status` (a JSON object of tensor entries with a dtype string, a shape list and two-integer `data_offsets`, an
optional `__metadata__` object), so a malformed header is a named refusal before the staged build and a named failure of the
output verifier; asks the pinned safetensors reader itself for every base shard the index lists
(`safe_open` on the header: dtype names, string metadata values, contiguous offsets, byte count per tensor, file
length) and decodes every Jinja template the loaders read (`chat_template.jinja`, `additional_chat_templates/*.jinja`)
as strict UTF-8, refusing by file and reader message / decode position. the
construction is STAGED: the whole build (overlay, sidecar sync, `config.json <- base`, verification, marker) happens
in `<out>.tmp` beside an existing output, which is replaced only afterwards by a rename swap (`<out>` ->
`<out>.previous`, `<out>.tmp` -> `<out>`, `<out>.previous` removed); on any failure before the swap the staging
directory is removed, the error names its stage and the existing output is byte-identical, under `--force
--no-verify` too. Transient disk: two exports side by side from the overlay to the swap. A leftover
`<out>.previous` is refused before the build unless `--force`. Under `--force` no complete marked tree is discarded while it is the only one (in the crash state between the
swap's two renames `<out>.previous` is renamed back to `<out>` first; `--force` rebuilds from the source, discarding a death-left
`<out>.tmp` once `<out>` is complete again, and a kept staged copy is published only by a request without `--force`, after the
same verification as a fresh build, where `--no-verify` trusts its bytes on size and tensor count; an inherited
sole complete marked `<out>.tmp` is never deleted: its sidecars are re-synced from the base before the checks (files through a
temporary and a replace, the template directory through a staged copy and renames; never a shard), a missing staged
`config.json` is repaired the same way, a failed recovery check refuses by name and keeps the tree (shards and marker
untouched, sidecars possibly repaired: a `.copying` or `.writing` temporary left by an interrupted copy or unified-template
rewrite, under `--unified-chat-template` the two template files back in the base's form when that rewrite did not reach its
rename; the state at `<out>` named), and the overlay is never run while it is the sole
complete tree; a leftover `<out>.previous` beside an existing `<out>` goes at the swap), and one
materializer per output: `<out>.lock` (`fcntl.flock`, the file stays beside the output) refuses a second materializer on the
same `--out` by name (pid, host, start time) without touching anything, and the marker / skip / mismatch decision is taken under
that lock (a materializer paused before the lock cannot overwrite an export published meanwhile); a held lock is
waited for up to 120 s (`--lock-wait-seconds`, polling), so two simultaneous identical requests of a published source both
end with 0 and a differing one is refused on the marker; a holder still there after the bound is refused by name (the bound must be a finite number >= 0, `inf` and `nan` are refused); the source's identity of record is read
under that lock through a file object opened once there, and the overlay memory-maps that same open file, as does the default
verification, so the marker names the bytes the overlay consumed and the staged shards are verified against them
even when `model.pt` was replaced while the request waited or while the overlay ran; an in-place
change of the opened inode refuses the build when the client reports the change: best effort on a networked file system; the
guarantee of record is against a replaced path, by construction; the file system must support flock
(Lustre: the `flock` mount option). The weights-free loader calls
(`AutoProcessor` / `AutoTokenizer` / `GenerationConfig`) stay in the overlay, in the staged build with the
existing output kept: they are not torch-free, their
verdicts depend on the installed transformers rather than on the files, and the synthetic test bases are not
loadable by them. See the [SFT guide](../guides/sft.md#7-checkpoints-resume-and-export).

Tensor verification supports dense Qwen VLM checkpoints, including Qwen3.8's `qwen3_5`
architecture, without `--no-verify`. Dense exports retain exact checkpoint/output key coverage,
config and processor identity checks, and a bitwise non-expert weight probe; only expert layout
checks are absent. A base weight index containing MoE experts still requires checkpoint experts
and retains sampled expert-layout checks. The CLI's configured `--min-tensors` floor is applied
before tensor verification, which does not impose a second hard-coded floor.

CPU regressions use tiny local tensor shards and real HF configs/processors, including verified
publication of a staged dense export. They do not qualify full 27B conversion, GPU inference, or
vLLM compatibility. To verify an export previously built with `--no-verify`, choose a fresh output
or explicitly rebuild with `--force`; an intact existing output follows the idempotent skip path.

## Markers and records

| file                                         | owner                                                   | meaning                                                                                                                                                                                                                    |
| -------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.posttrain-archive.json`                    | `archive/publisher.py`                                  | Permanent remote declaration `posttrain.artifact-archive.v1`: native manifest digest and the complete file inventory with sizes and digests. Created before payload upload; not the completion marker                      |
| `.posttrain-checkpoint-artifact.json`        | `artifact.py`, `tools/merge.py`, `archive/publisher.py` | Native artifact manifest; committed last locally and remotely                                                                                                                                                              |
| `.uploaded`                                  | `archive/publisher.py`                                  | Local archive receipt `posttrain.archive-marker.v1`, bound to destination and inventory. Artifact entries carry digests; full-checkpoint bulk entries retain size-only semantics, with digests for transactional overrides |
| `.posttrain-checkpoint-artifact.legacy.json` | `archive/publisher.py`                                  | Original bytes preserved when converting a legacy merge manifest to the native schema                                                                                                                                      |
| `.merge-complete.json`                       | `tools/merge.py`                                        | Merge request identity and completed artifact inventory                                                                                                                                                                    |
| `.staged.json`                               | `archive/reader.py`                                     | Ownership record for fetched files and staging completion                                                                                                                                                                  |
| `.materialized`                              | `tools/materialize_hf.py`                               | HF output identity and shard inventory                                                                                                                                                                                     |
| `.overlay_source.json`                       | `tools/materialize_hf.py`                               | Identity of the build using an intermediate HF output directory                                                                                                                                                            |

Legacy text `.uploaded` markers still satisfy the existing full-checkpoint retention gate, which
checks existence. They do not prove a destination or digest and cannot authorize verified artifact
pruning. Both `.uploaded` and its interrupted-write temporary `.uploaded.tmp` are excluded from
checkpoint provenance, so archive bookkeeping cannot invalidate an external checkpoint successor.

## Credentials

Commands accept the same JSON credential format as `rl.upload.credentials`:
`{aws_access_key_id, aws_secret_access_key, endpoint_url, region_name}`. Markers contain only the
non-secret endpoint authority, bucket and prefix. Verification recomputes that destination identity
from the supplied credential file and refuses a marker for a different endpoint.

## Verification boundary

The CPU regression suites under `tests/test_checkpoint_*.py` and `tests/test_utils_object_store.py`
exercise marker/provenance behavior, publication races, immutable inventory declarations, missing
DCP shards, staging ownership, merge and HF conversion. Archive tests use an in-memory object store;
conversion tests need their torch, transformers and safetensors dependencies.

No real S3/GCS bucket canary or production model/HF-base conversion is claimed by this change.
Request-serialization and fake-store tests do not establish that a deployed endpoint enforces its
conditional-write header. Qualify that behavior against the intended endpoint before depending on
fleet `--delete-local` operations.
