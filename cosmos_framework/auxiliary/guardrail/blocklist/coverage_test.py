# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.coverage import (
    ZERO_WIDTH_SPACE,
    compare,
    main,
    measure,
    spellings,
    summarize,
)


@pytest.mark.L1
def test_spellings_cover_the_four_evasions():
    """Each probe must still read as the entry to a person."""
    probes = spellings("deskin")

    assert probes["verbatim"] == "a photo of deskin in the scene"
    assert probes["spaced"] == "a photo of des kin in the scene"
    assert probes["invisible"] == f"a photo of des{ZERO_WIDTH_SPACE}kin in the scene"
    assert probes["fullwidth"] == "a photo of ｄｅｓｋｉｎ in the scene"


@pytest.mark.L1
def test_spellings_handle_a_short_entry():
    """A two-character entry must still produce a split, not an empty half."""
    probes = spellings("ab")

    assert probes["spaced"] == "a photo of a b in the scene"


@pytest.mark.L1
def test_measure_keys_results_by_index_and_spelling():
    """Entry text never appears in the results, so they are safe to paste."""
    results = measure(["deskin", "chromebook"], lambda text: "deskin" in text)

    assert results["0:verbatim"] is True
    assert results["1:verbatim"] is False
    assert all(":" in key for key in results)
    assert not any("deskin" in key for key in results)


@pytest.mark.L1
def test_summarize_counts_per_spelling():
    results = {"0:verbatim": True, "0:spaced": False, "1:verbatim": True, "1:spaced": True}

    summary = summarize(results)

    assert summary["verbatim"] == {"blocked": 2, "passed": 0}
    assert summary["spaced"] == {"blocked": 1, "passed": 1}
    assert summary["total"] == {"blocked": 3, "passed": 1}


@pytest.mark.L1
def test_compare_names_what_a_change_gave_up():
    """`lost` is the answer an unblocking change has to provide."""
    before = {"0:verbatim": True, "0:spaced": True, "1:verbatim": False}
    after = {"0:verbatim": True, "0:spaced": False, "1:verbatim": True}

    diff = compare(before, after)

    assert diff["lost"] == ["0:spaced"]
    assert diff["gained"] == ["1:verbatim"]


@pytest.mark.L1
def test_compare_reports_keys_present_on_only_one_side():
    """A list that changed length must not be silently diffed against itself."""
    diff = compare({"0:verbatim": True}, {"0:verbatim": True, "1:verbatim": True})

    assert diff["only_in_after"] == ["1:verbatim"]
    assert diff["lost"] == []


@pytest.mark.L1
def test_compare_mode_exits_non_zero_when_something_was_lost(tmp_path, capsys):
    """CI can gate on it: a lost spelling is a failing exit code."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"results": {"0:spaced": True}}))
    after.write_text(json.dumps({"results": {"0:spaced": False}}))

    assert main(["--compare", str(before), str(after)]) == 1
    assert "0:spaced" in capsys.readouterr().out


@pytest.mark.L1
def test_compare_mode_exits_zero_when_nothing_was_lost(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"results": {"0:spaced": True}}))
    after.write_text(json.dumps({"results": {"0:spaced": True, "1:spaced": True}}))

    assert main(["--compare", str(before), str(after)]) == 0


@pytest.mark.L1
def test_sweep_over_a_synthetic_list_runs_the_deployed_path(tmp_path):
    """End to end on a small list, so the machinery is exercised without a checkpoint."""
    blocklist_dir = tmp_path / "custom"
    blocklist_dir.mkdir()
    (blocklist_dir / "list").write_text("deskin\n", encoding="utf-8")

    assert main([str(blocklist_dir), "--json", str(tmp_path / "out.json")]) == 0

    written = json.loads((tmp_path / "out.json").read_text())
    assert written["results"]["0:verbatim"] is True
    assert written["summary"]["total"]["blocked"] >= 1
