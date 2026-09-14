# rearm_warpgroup_DRAFT_untested.patch

Uncompiled, untested draft (2026-09-14, session interrupted): moves the TMEM bias re-arm of the pre-biased promotion
(`-DG128_OPT_BIAS`) out of the promotion warps into a dedicated warpgroup (warps 12-15, one per TMEM lane quadrant, 24 registers).
Consumers arrive on a new per-stage `rearm_full` mbarrier instead of releasing the accumulator stage; the re-arm warps wait on it,
write 0x4B400000 into their 32 lanes x all columns with one non-unrolled `tcgen05.st.x8` loop from an 8-register opaque constant,
`tcgen05.wait::st` + `tcgen05.fence::before_thread_sync`, then do the accumulator `consumer_release`. Barrier counts changed:
accumulator consumer_arv_count = 2SM factor x 128, CLC consumer count + 128, TMEM-alloc named barrier + 128, `rearm_full[i].init(256)`.
Apply with `git apply include/rearm_warpgroup_DRAFT_untested.patch` on top of commit 1ff5e7b's headers, build with `-DG128_OPT_BIAS`,
then check `-Xptxas -v` spills (expect the promotion warps back at ~48 B) before measuring. See README "TMEM pre-bias promotion: status".
