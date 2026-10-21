# Bimanual sessions with decoded feedback

Use `OmniInference.open_bimanual_session` with a loaded, distilled schema-5
Bimanual checkpoint. The session calls the native causal generator and keeps
one cached VAE decoder open across steps. Each step receives fresh controls
and returns the decoded result before another control is requested.

This interface supports one process, eager execution, and VAE decoding on the
inference device. Disable torch.compile and CUDA graphs in the normal
[Bimanual setup](inference_bimanual.md). Distributed control delivery and a
separate VAE device are rejected. Serialize generation on each model: close
the active session before using the batch API or opening another session.
Drive the session serially from one controlling thread.

## Drive a session

Create `pipe` through the normal Framework setup API, as the native inference
script does: `setup.get_inference_cls().create(setup)`. Supply an already
downloaded checkpoint and initial RGB image. A recorded action file is not
required. The initial image uses the native action-input preprocessing.

Your policy and human-control adapters must each accept the latest observation
and required action-row count and return a schema-5 action payload. They are
application callbacks, not bundled policies. For example, with those adapters
available as `policy` and `human`:

```python
from pathlib import Path

with pipe.open_bimanual_session(
    vision_path=Path("initial.png"),
    prompt="Move the object with both arms.",
    domain_name="agibotworld",
    fps=30,
    seed=2026,
) as session:
    for step_index in range(12):
        source = "policy" if step_index < 6 else "human"
        adapter = policy if source == "policy" else human
        rows = session.inspect()["action_rows_per_step"]
        payload = adapter(session.observation, rows)
        observation = session.step(payload, control_source=source)
        print(session.inspect())
```

The `control_source` label records the adapter selected by the caller; changing
the label alone does not change controls or execute a policy. Use the generated
observation to make the next decision. To compare changed and unchanged
controls, start two separate sessions from the same image, prompt, seed and
model; retain the actual actions and observations from both runs.

`observation` is the latest CPU RGB tensor `[1,3,T,H,W]` in `[0,1]`, on the
decoded model canvas. The first observation contains the conditioned prefix;
later observations contain the complete newly generated chunk. The session
holds only that latest RGB chunk, not the full output video. Save chunks in
temporal order if the application needs a video.

## Control contract and timing

`session.step` accepts the same `normalized_unified_v1` object as schema-5
action sidecars: `action` and scalar or per-row `domain_names`. Supply exactly
`teacher_forcing_frames_per_chunk * temporal_compression_factor` action rows.
For the released chunk-2, compression-4 model this is eight rows per step.
Controls take effect at the next chunk boundary; already generated frames
cannot be changed. At 30 FPS, eight action transitions cover about 0.267 seconds.

Keep the checkpoint's 59D layout, normalization, pose anchoring and domain
validity masks. The session validates those rules and appends zero padding to
the model's 64D width. It does not normalize controls again. Camera and robot
domains can change row by row; the domain IDs travel with each new chunk and
are not sliced from a preloaded clip. The low-level sampler's clean-cache
refresh receives the corresponding action and domain rows for each frame.

Invalid rows fail before advancing generation, so callers can correct them
and retry. A sampler or decoder failure closes the session and releases both
KV and decoder state. Create or reset a session before continuing after such
a runtime failure.

## Inspect history and reset

`session.inspect()` returns the session ID, selected seed/source, decoded-frame
and action counters, and the actual retained conditional/unconditional KV
frame IDs for every layer. `minimum_retained_history_span_seconds` is the
smallest first-to-last retained-frame span across conditional layers. The frame
lists show any gaps; a span does not imply every intermediate frame is present.
Counts describe retained state, not semantic memory or video quality.

The conditioned image is yielded before the generator's first cache prefill,
so initial history is empty. History snapshots are taken after clean-cache
refresh, ready for the next generation chunk. Capacity and sink behavior follow
the model's configured KV policy; a long rollout alone does not prove retention.

`session.reset(seed=2026)` closes the current generator and cached decoder,
then starts from a private copy of the original image/prompt batch. It creates
a new session ID, resets counters/history, and returns a fresh initial
observation. Reset does not retain the previous rollout or its normalization
mutations. Use a `with` block or `session.close()` to release the session.

Validate history beyond one second and a state-dependent delayed action or
occlusion/reappearance using the actual model and task inputs. CPU tests of
session lifecycle and KV storage do not establish policy response or model
quality. No second-robot example, 60-second trace, benchmark threshold or demo
release artifact is provided by this interface.
