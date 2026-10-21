# AV per-view caption

Produce captions for one shared physical scene, described separately from each
camera. Ground object layout and motion in synchronized WSM frames.

## Options and defaults

Specify options in natural language; there are no CLI flags to memorize.

| Option | Choices                                          | Default when omitted                                   |
| ------ | ------------------------------------------------ | ------------------------------------------------------ |
| Format | One paragraph, or timestamped blocks             | Timestamped blocks                                     |
| Length | Short, long, or an explicit word range per view  | Long: 300–400 body words per view, total across blocks |
| Views  | All supplied views, front-only, or named cameras | All supplied views                                     |

Examples:

- “Use av-per-view-caption with these WSM controls and scenario properties.”
  Uses the defaults above.
- “Generate long, all-view, timestamped captions, 300–400 words per view.”
- “Generate front-only paragraph captions, 80–120 words.”
- “Use av-per-view-caption: show me the options and defaults.”

For an options/help request, show this table and examples without requiring media
or starting captioning. For generation, state the selected format, length and
views before inspecting media, including defaults used. Do not require a separate
confirmation just to use defaults. Long means 300–400 body words per view unless
the user specifies another range; short is qualitative unless a range is supplied.
Never pad or invent claims to hit a quota.

Both paragraph and timestamped captions are supported. Our training data uses
timestamped descriptions, and we have empirically found them more effective for
generation. The [reference scene's two variants](../../shared/reference-scene.md#choose-a-caption-style)
show both formats and how to select them. Prefer timestamped captions for
post-training too, using the [target-grounded training workflow](../../post-training/references/data-contract.md#training-captions).

Read the [caption policy](#caption-policy) section completely before authoring.
It contains the traffic, appearance, four-property, and timing rules.

## Establish the scope

Create new camera-local caption bodies, or upsample supplied text into a new
output. Preserve compatible explicit appearance; repair traffic narration that
contradicts WSM and record the source correction.
Announce the selected scenarios/views/interval before work.
Use a fresh output directory; leave original media, captions, and manifests intact.

### When existing per-view captions are supplied

For an upsampling request, use the supplied bodies as source text, not as traffic
ground truth. Preserve compatible explicit appearance and setting details, enrich
them where needed, and correct traffic/motion or timing claims against inspected
WSM while preserving the four required properties. Record corrections in
`NOTES.md`, not in the caption bodies. WSM cannot decide between conflicting paint
colors for the same actor; use explicit source precedence or ask which appearance
to retain rather than claiming the controls resolve it.

Write a new bundle; do not overwrite the originals. Source prose format does not
override the output defaults: use long timestamped captions unless the user asks
for another format/length or explicitly requests preserving the source format.
Existing captions do not replace WSM inspection; request missing controls or
required properties rather than silently doing a text-only rewrite. If the user
asks to ignore original captions, exclude them from authoring entirely.

Actual WSM inspection for authoring is required. Extra visual review is optional,
off by default, and outside this generation guide.

Keep the task focused on the requested captions. Keep confidential media local
unless the user authorizes an external captioning service.

## 1. Establish the input contract

Read the user's inputs first; ask only for missing information that affects the
result. Record:

| Input              | Required interpretation                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scenario identity  | Keep variants distinct; use `(clip_id, video_uuid)` when present, not just a shared clip ID.                                                                                                            |
| Four properties    | Time of day, weather/intensity, road material/condition, geographic setting; record each value and its user/metadata source. Common field names are `time_of_day`, `weather`, `road_surface`, `region`. |
| View mapping       | Explicit camera key/direction to control URI/path, plus the source of that mapping; retain the selected view set and keys.                                                                              |
| Timing             | Requested interval, media timestamps/cadence, trim offsets and synchronization information. Do not assume a standard clip length.                                                                       |
| WSM interpretation | Applicable renderer/legend or verified conventions for classes, signals, lines, and any track IDs.                                                                                                      |
| Text               | Existing per-view bodies and their format, if supplied as source text for upsampling.                                                                                                                   |
| Output             | Requested format/length and destination. Default to long timestamped captions, 300–400 body words per supplied view in total, unless explicitly overridden.                                             |

Do not guess the four properties from WSM class colors or its black background.
If a property is absent, request it or an authoritative source. A user-confirmed
unspecified property may remain explicitly unconstrained; mark that property's
check as unspecified rather than passed. `keepOriginal` needs a supported original
setting; the token alone does not identify a geography.

A new user instruction can explicitly replace a metadata property; record that
override. If sources conflict without clear precedence, ask rather than silently
choosing the easier setting. Existing prose is not an implicit override of the
four properties or a mandatory traffic-event specification.

Check for missing/duplicate keys, suspicious mappings, and timing inconsistencies
before captioning. Verify camera directions against calibration and rendered
views, and use source timestamps to check synchronization. Resolve mapping
conflicts before assigning camera-local directions.

If controls, mapping, renderer semantics, or image-viewing capability are missing,
do the supported text/property checks and report the specific evidence gap. Do
not accept the captions as WSM-grounded or silently switch to caption-only
generation. Unknown semantics block affected claims, not unrelated observations.

## 2. Inspect synchronized WSM evidence

1. Probe the selected controls for dimensions, frame counts, FPS/PTS, duration,
   offsets, and decode errors. Resolve the selected interval onto a common
   sample-local clock. For uniform cadence, `N / FPS` is the half-open duration;
   `(N - 1) / FPS` is the last frame timestamp. Never derive a trim offset from
   an ambiguously named field without confirming its semantics.
2. Decode and **view** an overview spanning the selected interval, including its
   beginning and end, across all selected views. Roughly eight synchronized
   timestamps is a useful starting point. Label camera keys,
   frame indices and actual timestamps. Contact sheets must remain legible.
3. Build a shared provisional traffic inventory. Request additional adjacent
   frames and full-resolution crops for material questions: clipped entries,
   exits, occlusion versus detector dropout, small actors/signals, simultaneous
   detections, and uncertain cross-view identity. Name the question each request
   answers. Do not infer absence of a small object from a thumbnail.
4. Keep a compact evidence record linking observations to view/time/frame files.
   Distinguish files prepared on disk from pixels actually inspected or supplied
   to the authoring API. Only cite visual verification for the latter.

Use available image-viewing tools on decoded frames. Decoding alone is not
inspection. Do not use reference/generated RGB as a substitute for WSM evidence
in this workflow. If the user needs RGB-conditioned captioning, clarify that as
a different scope rather than silently changing the source of truth.

Cache by source identity/digest and frame/time, not basename. On resume, preserve
the evidence ledger, validate cached images before reusing them, and resend the
relevant overview/detail frames to the caption writer. Keep original controls
unchanged.

Bound additional inspection to claims that matter. If time/budget ends with a
material uncertainty, report it and stop short of acceptance; do not erase an
actor, relax a requested property, or claim verification to fit the budget.

## 3. Establish one scene, then generate

Separate the working record into:

- **Required setting:** the four properties, their sources, and explicit overrides.
- **WSM observations:** objects/classes, geometry, positions and visibility by
  camera/time, supported motion/signals, and unresolved associations.
- **Appearance:** retained source attributes or newly assigned plausible paint,
  glazing, material and texture. Assigned appearance is not recovered WSM truth.

Assign appearance once per sufficiently resolved object/landmark and reuse it
across views and time. Do not force uncertain detections into distinct physical
identities. Record corrected source narratives separately from prompt bodies.

Write only the requested format(s), from that shared record. Each body describes
what its camera can show in its interval. Preserve all four properties semantically,
without forcing a sea, sky or other out-of-frame landmark into every view. Use
concrete compatible appearance, not generic quality slogans or length-padding.

Timestamped captions use verified sample-local bounds such as `[0.0s-3.5s]:`;
paragraph captions need no timestamps. Keep tracking IDs, evidence, uncertainty
discussions and cross-camera annotation commentary out of generation prose.
Do not add an event script merely to fill an interval.

By default, finish after authoring and the inexpensive consistency/format checks
below; record `visual_review: not_run`. Do not re-view the evidence, start a review
agent, or enter a review/revision loop just because captions were generated.

## 4. Validate and hand off

Always check the written views together for compatible appearance, four-property
meaning, coherent ego motion and correct event tense, using the already inspected
evidence and shared record. Check traffic claims and camera-local visibility
during authoring; bracket narrated entries/exits with adjacent observations when
needed, including partial visibility. These checks do not require a separate
visual-review pass. Retain uncertainty that cannot affect a supported caption,
but do not hide uncertainty that invalidates one of its claims.

Mechanically check exact view-key coverage, requested format/length, and valid,
ordered timestamp bounds when present. Cross-view prose boundaries may differ;
outer sidecar frame spans must agree if sidecars are requested.

Default deliverables in a new local output directory:

- `captions.json`: a camera-key-to-body mapping, not a runnable inference request.
- `NOTES.md`: input/property sources, shared appearance, inspected evidence,
  checks, source corrections and unresolved questions.

Record `status: generated` and `visual_review: not_run` for completed authoring
without extra review. Use `draft` for incomplete work and record the unresolved
items in `NOTES.md`.

Only package sidecars/manifests when requested. Preserve non-caption metadata,
controls, initial images and settings; do not duplicate rig/camera headers. Check
the actual target reader and resolved per-view text before calling such a package
run-ready. Do not assume a top-level scene prompt is the text consumed by each
camera. Do not silently broadcast it to side/rear views.

## Caption policy

Apply this policy when generating captions or upsampling supplied text. Reason about one
physical scene jointly across cameras; write each caption from its own view.

### Source authority

| Source                                           | What it establishes                                                                                               | What it does not establish                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| User/scenario metadata                           | Required time, weather, road condition and geographic setting; explicit mandatory events when separately supplied | Traffic visibility or object trajectories in a particular view              |
| Inspected WSM plus verified renderer conventions | Supported traffic objects/classes, geometry, visibility, signals and motion for inspected times                   | Real paint, make/model, geography, photometric lighting or weather          |
| Compatible original prose                        | Intended appearance/background details worth preserving                                                           | Authoritative traffic events or an override of explicit scenario properties |
| Appearance assignments                           | Plausible concrete appearance consistent with the above                                                           | A claim that WSM recovered the real appearance                              |

Correct source prose when it contradicts WSM traffic evidence. Record a
`Source correction` in the notes, not in the caption. For example, a story about
waiting for green does not require a stop if the selected WSM interval only
supports a moving approach. Do not invent the later departure. Only a separately
specified mandatory event turns an event/control conflict into an input blocker.
Do not infer mandatory events from filenames, dataset titles or repeated prose.

WSM is the captioning reference, not infallible physical ground truth. A detection
dropout is not proof of disappearance. Two simultaneous cuboids establish two
rendered detections, not necessarily two physical vehicles; an upstream duplicate
may exist. Inspect adjacent evidence, retain unresolved interpretation and avoid
unsupported counts. Do not invent actors to repair apparent perception omissions.

### Traffic, camera locality and identity

- Do not add unsupported vehicles, people, cyclists, traffic signs/sign text,
  signals, lane markings, crossings or traffic geometry. Preserve supported
  class and coarse shape; a cuboid is not evidence of a brand or exact subtype.
- Verify the legend before interpreting colors, signals or class labels. Class
  color is not paint and shared color is not an object ID. Keep unknown classes
  unknown rather than promoting them to a pedestrian or equipment subtype.
- Scope every count and position to its camera and time. A clip-wide inventory
  is not a simultaneous frame count. Avoid unearned “all,” “only,” “throughout,”
  precise speeds/distances, or exact unique-object totals.
- Distinguish image-left/right, ego-left/right and road-relative lanes/maneuvers.
  Screen motion can reverse across cameras without an object changing course.
  One ego vehicle cannot simultaneously be stationary and translating.
- Different cameras may see signals for different approaches; do not assume
  every visible light governs the ego lane.
- Associate objects with continuity, geometry, dimensions, overlap and verified
  track IDs where available. Local labels are annotations, not recovered IDs.
  Keep uncertain associations explicit rather than inventing identity certainty.
- Preserve compatible supplied appearance; otherwise assign plausible paint,
  glazing, finish or clothing once and share it across visible views/times.
  Do not repaint an object, duplicate a landmark, or assign distinct colors as
  supposed proof that ambiguous detections are different physical objects.
- Synthesize non-traffic surface/background detail only where compatible with
  controlled geometry and the requested setting. Do not block a lane with a new
  structure, change road topology, or put an invented landmark in every view.
- Do not paste a forward-driving narrative into side/rear views. Shared weather
  and physical identity do not require identical prose or identical screen-space
  positions. Explain cross-view associations in authoring notes, not model prompts.

### Four properties: meaning, not keyword presence

Read the exact selected scenario's values. Do not borrow them from another weather
variant of the same clip. Normalize field names without losing their semantics.

| Property           | Check                                                                 | Examples of a conflict or overreach                                                          |
| ------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Time of day        | Sky, shadows and illumination agree with the requested period         | “Night” with sunlit facades or a daylight-blue sky                                           |
| Weather            | Category and intensity remain recognizable; visibility agrees with it | “Dense fog” with perfectly clear distant scenery; replacing heavy snow with a few flakes     |
| Road condition     | Preserve specified material and condition                             | `wetConcrete` described as dry asphalt; `damaged` rewritten as pristine pavement             |
| Geographic setting | Compatible environment, without forcing invisible landmarks           | Treating `coastal` as ocean visible in every camera; inventing a country from `keepOriginal` |

Each body should naturally communicate the requested setting where applicable
and remain compatible with all four properties. They are shared conditions, not
a demand that every view show the same evidence. A rear view without sea can
still be coastal. A desert road need not be sand. `keepOriginal` requires an
identified original setting or an explicit unresolved limitation.

Preserve combinations rather than normalizing them: fog does not imply wet
pavement, snowfall does not authorize covering explicitly dry asphalt, and
`slippery` alone does not imply ice. Flag genuinely irreconcilable requirements
instead of silently replacing one. Do not decide weather from WSM's background.

Daytime does not switch traffic signals or vehicle running lights off. Conversely,
do not invent lamp posts or other traffic infrastructure to explain night lighting.
Different camera directions can have different reflections and brightness without
different sun positions, weather, or assigned vehicle paint.

#### Atmospheric weather versus optical blur

Preserve rain, fog, snow and night; do not remove them to promise sharpness. An
unobstructed optical path can coexist with fog or rain. Avoid adding camera-fixed
grease, condensation, adhered droplets or full-frame smearing unless requested.
If lens contamination is explicitly required, preserve it rather than overriding
it with a “clean lens” template.

Describe fog with depth-dependent attenuation, rain/snow at the requested
intensity, and nearby detail only to the extent visibility permits. Do not
invent a visibility distance or promise all-distance sharpness. Road reflections,
spray and accumulation must also agree with the road-condition requirement.
Do not invent windshields, wipers or actors to explain clear optics. Keep any
optics clause brief; it is not a replacement for specific scene appearance or
a demonstrated cure for generation artifacts.

### Time, visibility and prose

Use a shared sample-local clock, not identical text intervals in every camera.
Inspect around narrated changes rather than inferring exact times from a sparse
overview. A narrow clipped edge sliver counts as visible. Distinguish first
partial visibility from becoming prominent, and FOV exit from occlusion/dropout.

For example, if an object is clipped at 3.2667s and absent at 3.3000s, a caption
starting at 3.5s should say it “has cleared the edge,” not “leaves” in that interval.
If an object is already clipped at an interval start, describe that initial state.

Timestamped prose should use actual selected bounds and enough intervals for
meaningful changes, not an arbitrary equal split. In paragraph prose, distinguish
sequential from simultaneous sightings; qualify time-varying positions/signals
instead of freezing them across the clip. Omitting motion narration does not
mean saying everything is still.

Concrete appearance is preferable to generic filler about natural colors,
perspective, image quality or visible detail. Do not add sounds, intentions,
safety judgments, license plates, brands or decorative sign text to meet length.
If the requested length would require unsupported claims or repetition, report
the limitation rather than padding. Keep camera/rig headers separate from bodies.
