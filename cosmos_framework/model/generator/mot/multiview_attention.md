# Multi-view attention

This document summarizes the multi-view FlexAttention mask used by the GEN tower.
The implementation lives in `flex_attention.py`; task-specific packing into
`MaskItem` records is built in `cosmos3_vfm_network._multiview_mask_items`.

The real attention is rectangular when text/UND tokens are present: GEN tokens are
queries, and keys are `[UND | GEN]`. Every GEN query attends to every same-sample
UND key. The diagrams below omit UND keys and materialize only the GEN-to-GEN
quadrant.

The diagrams are group-level views. A cell such as `V10` stands for all spatial
tokens in vision frame `t=1`, view `v=0`; every token in that group has the same
mask visibility. Rows are keys, columns are queries, and `1` means the query may
attend to that key.

## Shared predicate

Each GEN token carries these mask fields:

- `sample_id`: keeps attention inside one packed sample.
- `frame_id` and `view_id`: define the multi-view grid.
- `is_control`: marks clean control streams.
- `is_noisy`: records whether the token came from a noisy frame. The current
  sensor-to-sensor rules do not distinguish clean sensor tokens from noisy sensor
  tokens; both use the same `attention_scope`.
- `timestamp`: used only by windowed `decomposed` attention.

Within one sample, the predicate is:

- sensor query -> sensor key: allowed inside `attention_scope`.
- sensor query -> control key: same view, any frame.
- control query -> control key: same view, any frame.
- control query -> sensor key: same view, any frame, only when
  `control_attends_sensor=True`.

`attention_scope` only affects sensor-to-sensor edges:

- `all_views`: any sensor frame and any view in the sample.
- `same_view`: any frame in the query's own view.
- `decomposed`: same view at any frame, plus the query's own frame across views.
  With `decomposed_temporal_window_seconds`, the own-frame term becomes keys at
  real capture times within an inclusive `(start, end)` offset from the query.
  Always use explicit `(start, end)` tuples in new configs and examples (two-element
  lists in YAML): `(-0.4, 0.0)` is past-only; `(-0.2, 0.2)` is symmetric with a
  0.4-second total width. Both bounds must be finite with `start <= end`; they do
  not have to straddle zero. Attention helpers reject scalar inputs. At config
  construction only, saved legacy scalar values are migrated to tuples: `0.4`
  becomes `(-0.4, 0.0)`, preserving the old past-only attention edges. Checked-in
  recipes use explicit tuples; interactive replay retains its separate causal-duration
  setting and translates it to bounds when calling shared attention.

## Maskless counting modes and temporal windows

Both counting modes of `backend="maskless"` support `decomposed_temporal_window_seconds`:

```python
MultiviewAttentionConfig(
    backend="maskless",
    deduplicate_cross_view=False,  # True excludes own-view keys from the temporal pass.
    mask=MultiviewAttentionMaskConfig(
        attention_scope="decomposed",
        decomposed_temporal_window_seconds=(-0.4, 0.0),
    ),
)
```

Its temporal pass groups sensor queries by float32 frame-start capture time and
gathers keys from all sensor views within the interval. This includes RGB↔LiDAR
with different frame rates. Each query occurs once in this pass; keys can repeat
across different query times. Variable-length dense attention computes the overlapping
rectangles in one call, or the exact rectangles in one call per hierarchy depth,
without a token mask. Plans are built once per forward.

With `deduplicate_cross_view=False`, this is **not exact-count attention**: own-view sensor keys in the window also
occur in the unchanged same-view/all-time pass, so their softmax mass is counted
twice. Controls do not enter the temporal pass. Same-view/control/caption rules,
sample isolation and the single-view-only bypass are unchanged. A lone camera
plus LiDAR is not a single-group sample and does use cross-sensor attention.

`deduplicate_cross_view=True` instead partitions view pairs into disjoint
rectangles using a binary hierarchy. Own-view keys occur only in the same-view
pass, so the merged result counts each permitted key once. Both modes share the
capture-time window planner and positive log-sum-exp merge.

`mask.decomposed_temporal_window_includes_first_frame=True` also admits each
sample's first frame (capture time 0, the given frame under I2V) for every query
time. The planner widens the same window table rather than adding a pass, so a
first frame already inside the window is admitted once and exact mode still keeps
own-view keys out. It needs a window and `backend="maskless"`: the Flex mask has
no such rule.

`None` preserves the original midpoint-quantised instant grouping, including its
same-frame overlap when deduplication is off. Existing experiments stay unchanged;
matched exact/overlapping symmetric-window prediction recipes are registered in
`configs/base/experiment/multiview/av/av_maskless_exact_16b.py`. Enabling a window adds
no weights but changes connectivity and duplicates more KV storage/work; benchmark
memory and training speed before scaling. It does not make same-view attention causal.

## Generation With Camera-Trajectory Controls

MultiCamVideo forward-dynamics generation packs a target vision item and one
camera-pose action control item. The first vision frame is clean conditioning.
The remaining vision frames are noisy targets. Camera actions use
`action_start_frame_offset=1` in their mRoPE positions, so action block `A1v`
sits at the temporal position of target vision frame `V1v`; there is no `A0v`
for the clean source frame. The mask does not pair action steps with frames.

This is the effective mask for 3 views and two generated target frames, under the
current generation config's `attention_scope="all_views"`, with
`control_attends_sensor=False`:

```text
K \ Q | A10 A11 A12 A20 A21 A22 | V00 V01 V02 V10 V11 V12 V20 V21 V22
------+-------------------------+-------------------------------------
A10   |  1   .   .   1   .   .  |  1   .   .   1   .   .   1   .   .
A11   |  .   1   .   .   1   .  |  .   1   .   .   1   .   .   1   .
A12   |  .   .   1   .   .   1  |  .   .   1   .   .   1   .   .   1
A20   |  1   .   .   1   .   .  |  1   .   .   1   .   .   1   .   .
A21   |  .   1   .   .   1   .  |  .   1   .   .   1   .   .   1   .
A22   |  .   .   1   .   .   1  |  .   .   1   .   .   1   .   .   1
------+-------------------------+-------------------------------------
V00   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V01   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V02   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V10   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V11   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V12   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V20   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V21   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
V22   |  .   .   .   .   .   .  |  1   1   1   1   1   1   1   1   1
```

The important property is that camera controls are confined by view, not by
time: `V10` can see `A10` and `A20`, the whole view-0 trajectory, but not `A11`,
`A12`, or any other view's action block. Every frame of a view reads the same
trajectory, including the clean first frame. Vision queries still attend all
vision keys, including the clean first frame and all noisy target frames from
all views.

## AV Transfer With Clean Video Or Sensor Controls

AV transfer packs each control stream before the target stream in the same
modality. For WSM video transfer this means `[clean control video, target RGB]`.
For LiDAR-only transfer it means `[HD-map/range control, target range]`. For
joint camera+LiDAR transfer, camera and LiDAR items use disjoint `view_offset`
ranges so a camera token cannot match a LiDAR token merely because it shares a
local frame index.

In `_multiview_mask_items`, every vision or LiDAR item except the stream's last
item is marked `is_control=True`; the last item is the target sensor item. The
target item may contain clean conditioning frames and noisy target frames, but
both are non-control sensor tokens for the mask.

The common 11-view WSM transfer ablations use the same predicate with different
`attention_scope` settings. The materialization below uses 3 views, two frames,
one clean control video `C`, one target video `V`, `attention_scope="decomposed"`,
and `control_attends_sensor=False`:

```text
K \ Q | C00 C01 C02 C10 C11 C12 | V00 V01 V02 V10 V11 V12
------+-------------------------+-------------------------
C00   |  1   .   .   1   .   .  |  1   .   .   1   .   .
C01   |  .   1   .   .   1   .  |  .   1   .   .   1   .
C02   |  .   .   1   .   .   1  |  .   .   1   .   .   1
C10   |  1   .   .   1   .   .  |  1   .   .   1   .   .
C11   |  .   1   .   .   1   .  |  .   1   .   .   1   .
C12   |  .   .   1   .   .   1  |  .   .   1   .   .   1
------+-------------------------+-------------------------
V00   |  .   .   .   .   .   .  |  1   1   1   1   .   .
V01   |  .   .   .   .   .   .  |  1   1   1   .   1   .
V02   |  .   .   .   .   .   .  |  1   1   1   .   .   1
V10   |  .   .   .   .   .   .  |  1   .   .   1   1   1
V11   |  .   .   .   .   .   .  |  .   1   .   1   1   1
V12   |  .   .   .   .   .   .  |  .   .   1   1   1   1
```

This matrix shows three transfer-specific effects:

- Target sensor queries see every clean control cell of their own view, at any
  time.
- Clean control queries attend every control cell of their own view, and do not
  see target sensor keys unless `control_attends_sensor=True`.
- Target sensor-to-sensor attention follows `decomposed`: same view across
  time, plus same frame across views.

For `attention_scope="all_views"`, replace the target-sensor quadrant with all
ones. For `attention_scope="same_view"`, keep only the same-view columns in that
quadrant. If `control_attends_sensor=True`, add the reverse same-view edges from
control queries to target sensor keys: `Ctv -> Vsv` becomes visible for every
pair of times `t`, `s` and each view `v`.

For joint camera+LiDAR transfer, the same matrix applies independently to each
sensor's control/target pair on its own view-offset range. Cross-sensor
sensor-to-sensor visibility then comes only from `attention_scope`: `all_views`
can couple camera and LiDAR target tokens; `decomposed` needs
`decomposed_temporal_window_seconds` so the temporal half compares real capture
time instead of unrelated per-sensor frame indexes.
