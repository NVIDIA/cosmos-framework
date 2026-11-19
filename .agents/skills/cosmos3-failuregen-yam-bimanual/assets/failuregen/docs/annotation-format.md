# Label task phases and failure events

Annotate each recording on two independent timelines: **task phase** describes
what the robot is doing, and **failure state** describes whether a failure event
is visibly occurring. Also record an outcome for the complete episode or clip.
These instructions apply to source recordings and generated videos. The three
fixed notebook examples already have supplied timestamps and need no new labeling.

## Review the recording

1. Watch the complete recording before assigning labels. Inspect the overhead
   and wrist views together where available, using their shared timestamps.
2. Preserve the original episode ID and video timing. Record interval boundaries
   in seconds relative to the start of the recording being annotated. For an
   extracted clip, retain its start offset in the source episode.
3. Label the task phases and failure events separately. Record the affected arm
   and task-cycle identity when an episode contains repeated attempts.
4. Review the boundaries frame by frame. Use `unknown` wherever occlusion,
   corruption, or ambiguous evidence prevents a defensible label.
5. Assign the overall outcome after reviewing the complete recording. Mark each
   timeline reviewed only after checking its labels and boundaries.

## Task-phase labels

Assign the phase that describes the visible action, even when that action fails.
Use the same boundary conventions across recordings.

| Phase              | What to label                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `pick_grasp`       | The gripper approaches the target object and attempts to close around it.                            |
| `lift_extract`     | The robot attempts to lift the object clear of its support or extract it from the pickup location.   |
| `move_to_handover` | The robot moves from the pickup region toward the position for transferring the object between arms. |
| `handover`         | The grippers attempt the transfer: the receiving gripper engages and the giving gripper releases.    |
| `transport`        | Motion from the handover region toward the placement destination.                                    |
| `place_release`    | The robot lowers or positions the object at the destination and attempts to release it.              |
| `recovery`         | Corrective motion or a retry in response to a failed attempt.                                        |
| `idle`             | The robot is waiting or stationary without an active task operation.                                 |
| `unknown`          | The visible evidence does not establish the phase.                                                   |

Start a phase at the first frame showing that operation and end it when the next
operation begins. Do not require every recording to contain every phase, and do
not label a successful grasp or handover solely because the robot enters that
phase. Mark repeated task cycles consistently.

## Failure-state labels

| State           | Visual evidence                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `normal`        | No active failure event is visible at this timestamp. This does not mean the episode succeeded.                           |
| `pickup_miss`   | A pickup attempt fails to acquire the object; include the visible empty withdrawal.                                       |
| `pickup_slip`   | An acquired object slips or falls during pickup, lift, or movement toward handover, before the transfer between grippers. |
| `handover_drop` | The object falls during the attempted transfer between grippers.                                                          |
| `unknown`       | Occlusion, ambiguity, or corruption prevents a defensible failure-state label.                                            |

Distinguish a miss from a slip by checking whether the object was acquired before
it was lost. Distinguish a pickup slip from a handover drop by checking whether
the transfer between grippers was underway. Do not guess from the intended motion
or the episode's name.

## Interval boundaries

Use intervals **[start, end)**: the start timestamp is included and the end
timestamp is excluded. Intervals within each timeline should not overlap. Cover
unreviewable spans explicitly with `unknown`.

Label **only the active failure event**. Start at the first visible evidence of
the miss, slip, or drop. End at the first frame after the fall or empty withdrawal
finishes. Return to `normal` afterward, even if the dropped object remains on the
table or the robot continues moving an empty arm. Retain the episode's failure
history in its overall outcome.

If the event is still active when the clip ends, end the annotation at the clip
boundary and record `event_end_observed=false`. Do not infer an unseen conclusion.
If a clip starts during an event, label the visible portion and note that its
onset was not observed.

Recovery belongs on the task-phase timeline and does not extend a failure event.
For example, a drop can end while recovery continues: the phase is `recovery`
and the failure state is `normal` once the active drop has finished. Record a
new failure interval if another failure occurs during recovery.

## Source-episode outcomes

| Outcome                  | When to assign it                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `success_clean`          | The complete task succeeds without a failure or recovery. Only these episodes qualify as clean hosts.                         |
| `success_after_recovery` | The task eventually succeeds after a failed attempt and corrective action.                                                    |
| `failed_pickup`          | The episode ends unsuccessfully because of a pickup failure.                                                                  |
| `failed_handover`        | The episode ends unsuccessfully because of a handover failure.                                                                |
| `indeterminate`          | The recording does not establish the final outcome, or the outcome cannot be represented confidently by the available labels. |
| `unreviewed`             | The episode has not yet been fully reviewed.                                                                                  |

Keep all observed failure intervals even when the final outcome is successful
after recovery. A recovered episode must not be marked `success_clean`.

## Generated-clip outcomes

Apply the same phase labels, failure states, and timing rules to generated clips.
Judge the visible result independently of its donor label or generation recipe.
First check that the video is coherent enough to interpret the robot–object
interaction, then assign an outcome:

| Outcome              | When to assign it                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `intended_failure`   | The intended failure is visibly present in a usable video.                                                 |
| `no_failure`         | No failure is observed during the clip; this does not establish success beyond the visible window.         |
| `different_failure`  | A failure occurs, but it differs from the intended type. Record the observed type and supporting evidence. |
| `invalid_generation` | Visual artifacts or incoherent motion make the clip unusable.                                              |
| `uncertain`          | The video does not provide enough evidence to decide the outcome.                                          |
| `unreviewed`         | The clip has not yet been reviewed.                                                                        |

If an observed failure falls outside the listed failure-state vocabulary,
describe it in notes and resolve the label definition before using it for
training. Do not force it into the intended class. Exclude unusable clips and
resolve uncertain labels before selecting training examples.

Keep labels associated with the clip identity, donor and host episode IDs,
source-window timestamps, and generation seed. Preserve the conditioning actions
and source references alongside the video so the observed labels can be traced
back to the generated example.
