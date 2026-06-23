# The DROID conversion script, in plain English

This explains `tools/lance_datagen/build_composed_droid.py` — what it does and the video
jargon (GOP, keyframes, codec, blob) — so it can be explained without a video background.

## The one-sentence version
For each recorded robot session, we take its 3 camera videos, pre-combine them into one
small video laid out the way the model wants, and save that as a chunk of bytes in a
database — so that during training the computer does almost no work to fetch a clip.

## The problem we're solving
A DROID episode has **3 cameras** (a wrist camera + 2 side cameras). During training, the
model repeatedly asks for a short **window** of ~17 frames, and for each window the normal
loader has to, *every single time*:
1. open 3 separate video files,
2. decode (uncompress) frames from each,
3. shrink the 2 side cameras to half size,
4. stitch the 3 views into one picture (wrist on top, two side views on the bottom).

That stitching+shrinking is the same every epoch, and decoding 3 videos is the slow part
(~98% of the time). We do all of it **once, offline**, and store the result.

## Key terms (plain English)
- **Frame** — one still picture. A video is just many frames shown quickly (here 15 per
  second).
- **Resolution** — how many pixels in a picture. Each DROID camera is 320×180; our combined
  picture is 270×320.
- **Codec / H.264** — the standard way to squash video so it's small on disk. Think "ZIP,
  but for video." "Decode" = unzip back into pictures.
- **Keyframe (a.k.a. I-frame)** — a frame stored as a *complete* picture, all by itself
  (like a standalone photo / JPEG). You can jump straight to it and see it immediately.
- **Delta frame (P/B-frame)** — a frame stored only as *"what changed since the previous
  picture"* (e.g. "same as before, but the arm moved a bit"). Very small to store, but to
  see frame #50 the computer must first replay frames #1→#49 to build it up.
- **GOP = "Group Of Pictures"** — how often a keyframe appears. GOP=30 means: 1 keyframe,
  then 29 delta frames, then another keyframe, and so on.
  - **Big GOP** (e.g. 30): smaller files (lots of cheap delta frames) but **slow random
    access** — to grab a frame in the middle you must decode back to the previous keyframe.
  - **GOP=1, "all-intra"**: **every** frame is a keyframe. Files are bigger (you lose the
    "what changed" savings) but you can jump to **any** frame instantly. Perfect for
    training, which grabs random windows constantly.
- **Blob** — a single opaque chunk of bytes (here, one small `.mp4`) stored as one cell in
  a database table. LanceDB ("blob v2") can fetch just the bytes it needs for one episode,
  even from cloud storage.

## What the script actually does, step by step
For every episode:
1. **Decode** the 3 camera videos into raw frames (using the exact same routine the normal
   loader uses).
2. **Compose** each moment in time into one 270×320 picture: wrist on top, the two side
   cameras shrunk to half and placed side-by-side underneath — the *exact* layout the model
   trains on.
3. **Re-encode** that sequence of composed pictures into one small `.mp4`, using **GOP=1
   (all-intra)** so any training window can be grabbed instantly.
4. **Store** that `.mp4` as a **blob** (one row per episode) in a LanceDB table.

At training time the loader now just: fetch the episode's small clip → decode the few frames
of the window. No 3-file juggling, no shrinking, no stitching. That's the ~2–2.5× speedup.

## Why this is smaller on disk, not bigger (the surprising part)
GOP=1 normally *inflates* a video (you give up the "what changed" savings). But we also went
from **3 camera pictures down to 1 half-size combined picture** — far fewer pixels. The
pixel savings more than cancel the GOP=1 penalty, so the result is **~0.35× the original**
size. (Using GOP=2–8 instead would shrink it further, trading a little random-access speed.)

## The one honest cost
Re-encoding compresses the video a second time, which loses a tiny bit of quality — like
re-saving a JPEG. We measured the difference at ~1–2% (≈32–37 dB PSNR), visually invisible,
and the robot-action labels are untouched (bit-identical). If a use case needs *exactly* the
original pixels, we also keep a no-re-encode variant (`LanceDROIDDataset`) that's slower but
byte-perfect.
```
Original:  [wrist.mp4] [side1.mp4] [side2.mp4]  --decode x3 + shrink + stitch EVERY time-->  frame window
Ours:      [one small combined.mp4 per episode] --decode once, already combined-->            frame window
```
