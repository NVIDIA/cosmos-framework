# Skill evaluations

[evals.json](evals.json) contains agent tasks with bundled inputs. Paths in
`files` are relative to the skill directory. Give the agent the
[skill](../SKILL.md), the listed input files, and the case's prompt. Use a fresh
output directory and assess the results against the assertions.

## Bundled case 2: coordinates and missing fields

[coordinate-scene.json](files/coordinate-scene.json) is a synthetic scene with
observations at 0, 500, and 1000 milliseconds. It contains a moving camera,
a delivery van, a pavement-to-grass boundary, and an amber traffic light.
Geometry uses centimetres and forward-left-up coordinates; quaternions use
`wxyz` order. Motion flags, signal thickness, and several map layers are absent.

This case exercises unit and axis conversion, camera projection, box-center
placement, class mapping, motion inference, and missing-data handling. It needs
the standalone package's dependencies but no GPU or recordings. CPU validation has
passed; this case does not test rendered video. Case ID 2 is retained from the
original evaluation list. It explicitly skips Hyperion retargeting to exercise
the opt-out and native camera projection checks.

## Bundled case 5: default Hyperion retargeting

This case uses the same synthetic input and its specified rear-axle ground
origin. The agent must retain a validated native conversion, then produce the
eleven Hyperion views using the bundled calibration: seven at 1924×1084 and four
fisheyes at 1720×1080. The three frames remain at 2 Hz. Verify vehicle motion
plus mounting offsets, exact annotation preservation, and CPU loading of all
camera models. RGB and video rendering are outside this case's scope.

The helper tests cover vehicle turns, camera projection, scene preservation,
and invalid poses. They run on CPU with the package's `wm` and test dependencies:

From the `wsm` directory:

```bash
python -m pytest --confcutdir=. \
    -c pyproject.toml \
    skills/scenario-to-rds-hq/evals -q
```

These checks do not validate rendered videos.

## Manual case 1: ClipGT/parquet conversion and rendering

This development check used separately supplied data. To repeat it, provide a
ClipGT/parquet clip, its camera calibration and poses, any source schema or
class definitions needed to interpret it, and an output directory. Include
synchronized camera recordings for visual alignment checks. The source data is
not bundled with this skill.

Example request:

> Use the scenario-to-rds-hq skill to convert the ClipGT/parquet clip at
> `<source-path>` into RDS-HQ under `<output-path>`. Validate the archives,
> render a representative WSM4 view, and inspect it against the source.
> Save the adapter and a report of mappings, assumptions, and omissions.

Check that the actual package reader loads all required archives, coordinates
and timing agree across components, and output counts account for the source
features. Check object eligibility and lane color/style mappings against source
evidence. Confirm the video completes and decodes, then inspect alignment at
the beginning, middle, and end using the native cameras. Then check the default
Hyperion output separately against the scene geometry. The native-camera manual
workflow has been run; it is not a bundled evaluation.

## Proposed cases: fixtures still needed

These cases have not been run. Add matching input files before adding them to
`evals.json`.

### Case 3: repeated observations and calibration ambiguity

Provide repeated map observations in a moving sensor frame, repeated placeholder
IDs, unknown numeric labels, gaps in signal observations, and calibration at a
different image size. Include evidence for the label meanings and image
resize/crop where known.

Check that distinct features remain separate, repeated observations form one
fixed map, numeric labels are not guessed, and signal gaps become `unknown`
outside a justified observation interval. The conversion should verify the
image transform or identify the missing calibration fact that prevents it.

### Case 4: actors, static props, and intermittent motion

Provide moving and parked cars, a parked car that later pulls out, a waiting
pedestrian, an animal, a stroller, protruding objects, traffic cones, and other
construction props. Represent them as tracked boxes and include source evidence
for unfamiliar labels.

Check that actors retain their classes, supported animals/strollers/protruding
objects map to `Others`, and standalone static props are omitted and counted.
The car that pulls out must have `object_is_moving=true` in every record,
including its parked portion. Unknown labels need source evidence before they
can be included as `Others`.
