# Making the LanceDB DROID loader faster than base (decode-bound) — researched roadmap

Counterintuitive headline: NVDEC is NOT the win at 320x180/640x360. torchcodec perf
docs + LeRobot PR #913 show GPU decode 8-21x SLOWER than many-core CPU decode for small
robot frames (PCI-e + per-clip init dominate; L40S has only 3 NVDEC units). The win is to
make the STORED representation cheaper to decode — which the base loader cannot do (it
reads canonical raw DROID mp4s). All levers below stay video-encoded (no disk blowup).

Ranked by (speedup x ease):
1. seek_mode="approximate" (torchcodec): base uses exact -> full-file scan per decoder
   open. Real DROID = thousands of per-episode files, shuffled -> constant decoder
   creation -> scan paid repeatedly. Approximate skips it. Trivial. Near-exact (validate).
2. Pre-composed + pre-resized + short-GOP per-episode video: store ONE clip per episode
   with the 3 views laid out at training res (270x320) + tiny GOP (g=2). Loader decodes
   one ~half-pixel stream instead of 3 full views + F.interpolate + concat. One-time
   transcode (lossy vs original, standard practice). Biggest structural lever.
3. Per-episode Blob-V2 byte-range reads: only touched bytes move from S3; small files ->
   cheap decoder init.
4. Batched decode across the whole DataLoader batch (already in our __getitems__).
Not recommended for this workload: NVDEC (small frames), DALI/PyNvVideoCodec (only if CPU
saturates / large frames). Sources: meta-pytorch torchcodec perf docs, lerobot PR#913,
lancedb blob-v2, NVIDIA DALI/PyNvVideoCodec docs, L40S datasheet.
