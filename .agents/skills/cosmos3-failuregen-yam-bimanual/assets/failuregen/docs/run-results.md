# Synthetic failure data for Cosmos Edge fine-tuning

We used synthetic failure videos to supplement real recordings when fine-tuning
Cosmos Edge for failure detection. We compared this with fine-tuning Cosmos Edge
on real data alone. The experiment and results are described below; the notebook
generates three synthetic examples and does not run this downstream comparison.

## What the synthetic data adds

FailureGen reuses observed failure motion in compatible host scenes. Cosmos
predicts the resulting video from the host observation, transferred motion, and
neutral task text. Different hosts vary the recorded scene and starting context;
additional sampling seeds provide more renders of a pairing.

In the industrial experiment, the source corpus contains 80 recordings of a
bimanual YAM robot picking up, handing over, and placing reels. Only the 66
training recordings supply synthetic donors and hosts. The reference batch
contains 108 failure-transfer candidates, 216 matched controls, and 54 donor
replays.

| Synthetic-data result                         | Reported value                                       |
| --------------------------------------------- | ---------------------------------------------------- |
| Selected failure clips                        | 75 from 108 transfer candidates                      |
| Distinct source contexts among selected clips | 26 hosts and 23 failure donors                       |
| Observed outcomes among selected clips        | 69 intended failures and six other failures          |
| Selected failure types                        | 18 pickup misses, 27 pickup slips, 30 handover drops |

The generated event determines its label. A visually coherent successful
continuation is not a confirmed synthetic failure. Review filters invalid or
uncertain outputs and records observed event intervals. Retained examples keep
the video, conditioning motion, labels, and source/checkpoint provenance.

## Cosmos Edge fine-tuning comparison

We fine-tuned Cosmos Edge using real training data plus the 75 selected synthetic
failure clips and compared it with a model fine-tuned on real data alone. Both
were evaluated on the same real evaluation subset.

| Training data                          | Detected failure events | Average precision |
| -------------------------------------- | ----------------------- | ----------------- |
| Real data only                         | 28/36                   | 0.4085            |
| Real data + 75 synthetic failure clips | 33/36                   | 0.5119            |

This preliminary comparison used different sampling and optimization budgets,
so it does not isolate the effect of synthetic augmentation alone. Cosmos Edge
fine-tuning and evaluation are outside this cookbook's generation workflow.

## Notebook scope

The notebook uses a supplied adapted Cosmos3-Nano FD checkpoint and matching
prepared EEF inputs. It generates three fixed failure transfers: handover drop
(donor 42 → host 79), pickup miss (40 → 43), and pickup slip (61 → 46). The
labeling guide describes task phases and failure events for other recordings.
The notebook uses the supplied timestamps without reading source annotations
and does not run the original 378-clip experimental batch. FD training is needed only when using new or
different data and is shown in one commented cell.

See [the generation workflow](generation-workflow.md) for the action/checkpoint
contract and [the labeling guide](annotation-format.md) for observed-event review.
