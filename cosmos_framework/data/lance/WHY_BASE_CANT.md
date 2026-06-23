# Why the base (non-Lance) cosmos loaders can't capture these wins

The base cosmos loaders are bound to two canonical on-disk formats:
- DROID action → LeRobot v3: three separate per-view mp4s, seeked by timestamp,
  composed (resize + concat) at load time, every epoch.
- VLM / vision-SFT → WebDataset tar shards (sequential) or HF streaming.

Our wins split into two honest categories.

## A. Structural capabilities the base formats fundamentally lack (Lance-exclusive)
1. **True random access + global shuffle.** A WebDataset tar is sequential-only:
   to read sample N you scan from the shard start, and its "shuffle" is a bounded
   in-memory buffer (approximate, locally correlated). Lance is columnar with O(1)
   row addressing → true global shuffle via the Permutation API. No amount of
   base-loader tuning gives a tar random access — it's a format property.
   (Measured: lance ~18× raw random-read locally; webdataset cannot do it at all.)
2. **Columnar selective + filtered reads.** Want only some columns (captions without
   video), or a curriculum / quality-filtered subset? Lance reads only those
   rows/columns. A tar must stream + decode whole shards and discard the rest.
3. **blob-v2 byte-range reads from object storage.** Lance fetches only the bytes a
   decoder touches from a per-episode blob on S3. File/tar loaders fetch whole files
   (or FUSE-mount with coarse page caching). Per-blob range reads inside a queryable,
   versioned table is a Lance storage-layer feature.

## B. Representation optimizations Lance makes practical (not theoretically Lance-only,
##    but un-doable without reinventing Lance)
4. **Pre-composed / pre-resized / short-GOP per-episode clips** — the 2.0–2.5× action
   win. The base loader decodes 3 full views + `F.interpolate` + concat *per sample,
   every epoch*. We do that transform ONCE, offline, and store one small all-intra
   clip per episode. Anyone could pre-transcode to files in principle — but to *train*
   off that representation you need an index/manifest, per-clip lifecycle management, a
   shuffling sampler over millions of clips, object-store range reads, dataset
   versioning, and co-located tabular + caption + metadata. That is a data lake — i.e.
   you would be rebuilding Lance. The base loaders are hardcoded to the canonical
   LeRobot/WebDataset formats and have nowhere to put an optimized representation and no
   machinery to serve it. Lance *is* that machinery.

## The honest distinction
(A) are capability gaps in tar/file formats that no base-loader tuning closes. (B) are
representation changes that are *possible* off-Lance only by reimplementing Lance's
storage + sampling + versioning layer — at which point you've built Lance. As cosmos
ships them, the base loaders cannot adopt either without that substrate.

Note: the base could in principle add GPU/NVDEC decode — but research showed NVDEC is
8–21× *slower* than many-core CPU decode at these small robot-frame resolutions, so that
is not a win for either side. The win is the representation + access layer, which is
exactly what Lance provides and the canonical formats do not.
