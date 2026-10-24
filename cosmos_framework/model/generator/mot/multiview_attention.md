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
  With `cross_view_temporal_window.bounds`, the own-frame term becomes keys at
  real capture times within an inclusive `(start, end)` offset from the query.
  Always use explicit `(start, end)` tuples in new configs and examples (two-element
  lists in YAML): `(-0.4, 0.0)` is past-only; `(-0.2, 0.2)` is symmetric with a
  0.4-second total width. Both bounds must be finite with `start <= end`; they do
  not have to straddle zero. Configs and attention helpers reject scalar inputs.
  Checked-in recipes use explicit tuples; interactive replay retains its separate causal-duration
  setting and translates it to bounds when calling shared attention.

## Backend selection and maskless temporal windows

`backend="auto"` prefers maskless attention when the configuration is supported,
then FlexAttention with FlashAttention-4 when available, then FlexAttention with
Triton. Maskless supports `same_view` and `decomposed`; `all_views` uses a masked
backend. Enabling `action_gen` permits maskless attention for camera-pose controls
and the dedicated `multiview_action_conditioning` path, which builds plans for
separate video and action blocks. Explicit `maskless`, `flex_flash`, or
`flex_triton` selections pin the backend and reject unsupported configurations.
Batch geometry is validated after selection; an unsupported batch does not
silently switch backends.

Maskless attention always counts each permitted query-key edge once. It supports
`cross_view_temporal_window.bounds`:

```python
MultiviewAttentionConfig(
    backend="maskless",
    mask=MultiviewAttentionMaskConfig(
        attention_scope="decomposed",
        cross_view_temporal_window=CrossViewTemporalWindowConfig(
            bounds=(-0.4, 0.0),
            includes_first_frame=False,
        ),
    ),
)
```

Saved configs must use these nested cross-view settings and `same_view_temporal_window`
below. Legacy flat temporal-window fields are no longer migrated.

Its temporal passes group sensor queries by float32 frame-start capture time and
gather keys from other sensor views within the interval. This includes RGB↔LiDAR
with different frame rates. A binary hierarchy partitions directed view pairs
into disjoint rectangles. Each query occurs at most once per hierarchy depth;
its keys are disjoint across depths. Keys can repeat across different queries.
Variable-length dense attention computes one call per hierarchy depth without a
token mask. Plans are built once per forward.

Own-view keys occur only in the same-view pass. The positive log-sum-exp merge
therefore counts each permitted key once. Controls do not enter the temporal
passes. Same-view/control/caption rules and sample isolation remain unchanged.
Samples with only one view group have no cross-view work. A lone camera plus
LiDAR has two view groups and does use cross-sensor attention.

`mask.cross_view_temporal_window.includes_first_frame=True` also admits each
sample's first frame (capture time 0, the given frame under I2V) for every query
time. The planner widens the same window table rather than adding a pass, so a
first frame already inside the window is admitted once and own-view keys remain
excluded. It needs a window and a maskless backend (explicit or selected by
`auto`): the Flex mask has no such rule.

`None` preserves midpoint-quantised instant grouping, while still excluding
own-view keys from the cross-view passes. With shared clocks this matches the
masked decomposed scope. With mixed frame rates, midpoint grouping can differ
from the masked backend's frame-index matching; explicit temporal windows use
the same float32 timestamp bounds and tolerance as the masked backend.
Existing checkpoints that used overlapping maskless passes now receive the
disjoint counting behavior. Enabling a window adds no weights but changes
connectivity and can duplicate KV storage/work across query times; benchmark
memory and training speed before scaling. It does not make same-view attention
causal.

## Same-view temporal windows

`mask.same_view_temporal_window` is a `SameViewTemporalWindowConfig` with four
optional bounds: `sensor_to_sensor_bounds`, `sensor_to_control_bounds`,
`control_to_control_bounds`, and `control_to_sensor_bounds`. Each has the same
`tuple[float, float] | None` type as `CrossViewTemporalWindowConfig.bounds` and
uses seconds relative to the query. `None` keeps that edge unrestricted in time.

For example, `SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-0.4, 0.0))`
admits the current sensor frame and up to 0.4 seconds of earlier frames. At a
0.2-second latent frame period, the neighborhood spans offsets `(-2, 0)`.
The planner uses the packed streams' real frame periods, not a fixed FPS.

The current maskless neighborhood backend supports centered `(-N, N)` and causal
`(-N, 0)` bounds, including `(0, 0)`. Centered neighborhoods retain their existing
inward shift at sequence boundaries to preserve width; causal neighborhoods clip
at the beginning. The batched kernel requires common frame/spatial geometry and
bounds that resolve to the same frame offsets across its streams. Unsupported
mixed-rate geometry raises instead of treating seconds as frames. FlexAttention
does not implement these same-view windows. A control-to-sensor bound does not
re-enable edges disabled by `control_attends_sensor=False`.

## Separate-camera robot action conditioning

`multiview_action_conditioning` configures packing for the bimanual RGB/action
layout, including camera codes in padded action channels. The packer records
`PackedSequence.multiview_action_temporal_factor`, so attention identifies this
layout from the pack rather than interpreting every multi-item sample as cameras.

The shared multiview item builder describes each ragged RGB clip as a separate
camera target and each camera's replicated action rows as `(T-1, 1, factor)`.
An action item shares its camera's view identity and starts at latent frame one;
all full-rate action rows are retained. RGB and action items have sensor roles
and read the shared caption, even though the actions are clean conditioning.

These items use the same maskless plan builder and `multiview_attention` settings
as camera-major RGB, LiDAR/radar, and camera-pose control layouts. Cross-view
temporal windows and first-frame inclusion therefore share one configuration
path, with same-view keys excluded from cross-view passes. The separate-camera action layout still requires maskless
attention and baseline geometry: geometry encodings, cross-view bands, and
same-view neighborhood windows are explicitly rejected. Ordinary single-camera
packs without layout metadata continue to use two-way attention.

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
`cross_view_temporal_window.bounds` so the temporal half compares real capture
time instead of unrelated per-sensor frame indexes.
