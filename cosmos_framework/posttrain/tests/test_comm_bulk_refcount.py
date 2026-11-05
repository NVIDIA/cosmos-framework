# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest

from cosmos_framework.posttrain.comm.bulk import BulkStore

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_single_reader_pops_immediately():  # readers=1 == today's consume-once
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch(ref, readers=1) == {"x": 1}
    assert len(st) == 0
    with pytest.raises(KeyError):
        st.fetch(ref, readers=1)


def test_n_readers_pop_at_zero():
    st = BulkStore("s")
    ref = st.put({"x": 1})
    for _ in range(3):
        assert st.fetch(ref, readers=4) == {"x": 1}
    assert len(st) == 1  # still live: 1 read left
    assert st.fetch(ref, readers=4) == {"x": 1}
    assert len(st) == 0  # popped at zero


def test_release_decrements_like_fetch():
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch(ref, readers=2) == {"x": 1}
    st.release(ref)  # second reader dropped its claim without pulling
    assert len(st) == 0


def test_peek_does_not_decrement():
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch(ref, consume=False, readers=2) == {"x": 1}
    assert st.fetch(ref, readers=2) == {"x": 1}
    assert len(st) == 1


def test_release_missing_key_raises():
    # Fail-loud: release on never-put ref or after pop raises KeyError
    st = BulkStore("s")
    ref = st.put({"x": 1})
    # Consume and pop
    assert st.fetch(ref, readers=1) == {"x": 1}
    assert len(st) == 0
    # Try to release the already-popped ref
    with pytest.raises(KeyError):
        st.release(ref)
    # Try to release a ref that was never put
    from cosmos_framework.posttrain.comm.transfer.mover import BulkRef

    fake_ref = BulkRef.for_bulk("s", 999, 0, "")
    with pytest.raises(KeyError):
        st.release(fake_ref)


def test_release_does_not_count_as_fetch():
    # fetch_count is instrumentation meaning "data pulled"; release must not count
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch_count == 0
    # Release one reader's claim (no data transfer, decrements from 2 to 1)
    st.release(ref, readers=2)
    assert st.fetch_count == 0, "release() should not increment fetch_count"
    assert len(st) == 1, "entry should still be live after release"
    # Real fetch should increment and pop (decrements from 1 to 0)
    assert st.fetch(ref, readers=2) == {"x": 1}
    assert st.fetch_count == 1, "fetch() should increment fetch_count"
    assert len(st) == 0, "entry should be popped after final fetch"


def test_release_before_any_fetch_initializes_counter():
    # release(readers=3) before any fetch must initialize counter correctly so
    # later fetches don't prematurely pop
    st = BulkStore("s")
    ref = st.put({"x": 1})
    # One reader releases before any fetch (with readers=3 contract)
    st.release(ref, readers=3)
    # Two other readers fetch with readers=3 — both should succeed, second pops
    assert st.fetch(ref, readers=3) == {"x": 1}
    assert len(st) == 1  # still live: 1 read left
    assert st.fetch(ref, readers=3) == {"x": 1}
    assert len(st) == 0  # popped at zero (third read was the release)


def test_concurrent_put_fetch_release_is_race_free():
    # The store is NO LONGER single-threaded: _Host serves fetch_bulk/release_bulk from a
    # dedicated "bulk" concurrency group (workers/ray.py), so puts on the invoke thread race
    # fetches/releases on the bulk thread. BulkStore's lock must serialize those mutations.
    # This drives N threads over one shared store; the final state is DETERMINISTIC because
    # every key gets EXACTLY `readers` decrement ops (so no interleave can pop a key early or
    # touch an already-popped one), which is exactly what makes the counts add up ⇔ the lock
    # held. Without the lock the fetch_count increment / dict pops would drop or double under
    # contention and the deterministic assertions below would flake.
    import threading

    N_THREADS = 8
    KEYS = 200
    READERS = 2
    st = BulkStore("race")
    refs: list = [None] * KEYS
    errors: list = []

    # Phase 1 — 8 threads concurrently PUT all keys (hammers put() under the lock). refs[i] is
    # written by exactly one thread (disjoint index strides), so the list is race-free by index.
    def _put(indices):
        try:
            for i in indices:
                refs[i] = st.put({"k": i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    put_threads = [threading.Thread(target=_put, args=(range(t, KEYS, N_THREADS),)) for t in range(N_THREADS)]
    for th in put_threads:
        th.start()
    for th in put_threads:
        th.join()
    assert not errors, errors
    assert len(st) == KEYS  # all live before phase 2

    # Op plan: even keys -> 2 fetches (fetch_count += 2, 2nd fetch pops); odd keys -> 1 fetch +
    # 1 release (fetch_count += 1, whichever runs 2nd pops). Every key gets exactly READERS ops.
    ops: list = []  # (op, index)
    expected_fetch = 0
    for i in range(KEYS):
        if i % 2 == 0:
            ops += [("fetch", i), ("fetch", i)]
            expected_fetch += 2
        else:
            ops += [("fetch", i), ("release", i)]
            expected_fetch += 1

    # Phase 2 — 8 threads concurrently FETCH/RELEASE. Round-robin (ops[t::N]) puts a key's two
    # adjacent ops on different threads ⇒ genuine same-key concurrent contention on the store.
    def _work(chunk):
        try:
            for op, i in chunk:
                if op == "fetch":
                    assert st.fetch(refs[i], readers=READERS)["k"] == i
                else:
                    st.release(refs[i], readers=READERS)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    work_threads = [threading.Thread(target=_work, args=(ops[t::N_THREADS],)) for t in range(N_THREADS)]
    for th in work_threads:
        th.start()
    for th in work_threads:
        th.join()

    assert not errors, errors
    assert len(st) == 0  # every key fully popped
    assert st.fetch_count == expected_fetch  # exact: only fetches count, releases do not


def test_readers_must_be_at_least_one():
    # Fix: readers < 1 is a nonsense reader count (would pop on the initializing call, or never).
    st = BulkStore("s")
    ref = st.put({"x": 1})
    with pytest.raises(ValueError, match=r"readers must be >= 1"):
        st.fetch(ref, readers=0)
    with pytest.raises(ValueError, match=r"readers must be >= 1"):
        st.release(ref, readers=0)


def test_inconsistent_nondefault_readers_for_live_key_raises():
    # Fix (silent-leak trap): once a key's refcount is initialized with a reader count, a later
    # fetch/release passing a DIFFERENT non-default readers for the same LIVE key is a routing-map
    # disagreement — the counter would never reach 0 and the bulk would leak. Fail loud.
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch(ref, readers=4) == {"x": 1}  # initializes the counter with 4
    with pytest.raises(ValueError, match=r"inconsistent reader count"):
        st.fetch(ref, readers=3)
    with pytest.raises(ValueError, match=r"inconsistent reader count"):
        st.release(ref, readers=2)


def test_default_readers_stays_compatible_after_nondefault_init():
    # readers omitted/default (== 1) must never trip the consistency guard, even after a non-default
    # initialization — the common single-reader path stays backward compatible.
    st = BulkStore("s")
    ref = st.put({"x": 1})
    assert st.fetch(ref, readers=3) == {"x": 1}  # init 3, 2 left
    assert st.fetch(ref) == {"x": 1}  # default readers=1: compatible, decrements to 1 left
    assert len(st) == 1
    assert st.fetch(ref, readers=3) == {"x": 1}  # same non-default value: fine, pops at zero
    assert len(st) == 0
