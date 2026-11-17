# Synthetic smoke set

Ten 33-frame clips at 480x288 and 30 fps, drawn by `../../scripts/make_smoke_set.py` with Pillow and
encoded with ffmpeg: a grey road with lane markings, a moving car rectangle that stops in the clips whose
`q2` answer is `yes`, and a pedestrian rectangle on the road in the clips whose `q1` answer is `yes`.
The labels are a function of the clip index (see the script), so the set is learnable and the
answers are balanced per question. `annotations_train.json` holds clips 0 to 7, `annotations_val.json`
clips 8 and 9, in the trainer's LLaVA conversation format (`<video>` placeholder, question text,
one-word answer). The clips themselves are generated on the target machine (about 1 MB in total)
rather than vendored; run the script before the smoke.
