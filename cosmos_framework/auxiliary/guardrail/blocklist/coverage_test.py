# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.coverage import (
    ZERO_WIDTH_SPACE,
    _build_is_blocked,
    build_probes,
    compare,
    corpus_id,
    main,
    measure,
    probe_id,
    spellings,
    summarize,
)
from cosmos_framework.auxiliary.guardrail.blocklist.lookalikes import fold_lookalikes


@pytest.mark.L1
def test_spellings_cover_the_five_evasions():
    """Each probe must still read as the entry to a person."""
    probes = spellings("deskin")

    assert probes["verbatim"] == "a photo of deskin in the scene"
    assert probes["spaced"] == "a photo of des kin in the scene"
    assert probes["invisible"] == f"a photo of des{ZERO_WIDTH_SPACE}kin in the scene"
    assert probes["fullwidth"] == "a photo of ｄｅｓｋｉｎ in the scene"
    assert probes["lookalike"] != probes["verbatim"]
    assert probes["lookalike"].isascii() is False
    assert fold_lookalikes(probes["lookalike"]) == probes["verbatim"]


@pytest.mark.L1
def test_spellings_handle_a_short_entry():
    """A two-character entry must still produce a split, not an empty half."""
    probes = spellings("ab")

    assert probes["spaced"] == "a photo of a b in the scene"


@pytest.mark.L1
def test_probe_ids_do_not_depend_on_position_or_show_the_entry():
    """Reordering the list must not change what is compared, and ids are safe to paste."""
    forward = build_probes(["deskin", "chromebook"])
    backward = build_probes(["chromebook", "deskin"])

    assert forward == backward
    assert corpus_id(forward) == corpus_id(backward)
    assert probe_id("deskin", "spaced") in forward
    assert probe_id("  deskin ", "spaced") == probe_id("deskin", "spaced")
    assert probe_id("Deskin", "spaced") != probe_id("deskin", "spaced")
    assert not any("deskin" in key or "chromebook" in key for key in forward)


@pytest.mark.L1
def test_measure_keys_results_by_probe_id():
    probes = build_probes(["deskin", "chromebook"])

    results = measure(probes, lambda text: "deskin" in text)

    assert results[probe_id("deskin", "verbatim")] is True
    assert results[probe_id("chromebook", "verbatim")] is False
    assert set(results) == set(probes)


@pytest.mark.L1
def test_summarize_counts_per_spelling():
    results = {"a:verbatim": True, "a:spaced": False, "b:verbatim": True, "b:spaced": True}

    summary = summarize(results)

    assert summary["verbatim"] == {"blocked": 2, "passed": 0}
    assert summary["spaced"] == {"blocked": 1, "passed": 1}
    assert summary["total"] == {"blocked": 3, "passed": 1}


@pytest.mark.L1
def test_compare_names_what_a_change_gave_up():
    """`lost` is the answer an unblocking change has to provide."""
    before = {"corpus": "c", "results": {"a:verbatim": True, "a:spaced": True, "b:verbatim": False}}
    after = {"corpus": "c", "results": {"a:verbatim": True, "a:spaced": False, "b:verbatim": True}}

    diff = compare(before, after)

    assert diff["comparable"] is True
    assert diff["lost"] == ["a:spaced"]
    assert diff["gained"] == ["b:verbatim"]


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "after"),
    [
        ("different probe set", {"corpus": "other", "results": {"a:spaced": True}}),
        ("probe missing after", {"corpus": "c", "results": {}}),
        ("probe added after", {"corpus": "c", "results": {"a:spaced": True, "b:spaced": True}}),
    ],
)
def test_compare_refuses_runs_over_different_probes(tmp_path, capsys, name, after):
    """Results over different probes are incomparable, never a clean comparison."""
    before = {"corpus": "c", "results": {"a:spaced": True}}
    (tmp_path / "before.json").write_text(json.dumps(before))
    (tmp_path / "after.json").write_text(json.dumps(after))

    assert compare(before, after)["comparable"] is False, name
    assert main(["--compare", str(tmp_path / "before.json"), str(tmp_path / "after.json")]) == 2
    assert "incomparable" in capsys.readouterr().out


@pytest.mark.L1
def test_compare_mode_exits_non_zero_when_something_was_lost(tmp_path, capsys):
    """CI can gate on it: a lost spelling is a failing exit code."""
    (tmp_path / "before.json").write_text(json.dumps({"corpus": "c", "results": {"a:spaced": True}}))
    (tmp_path / "after.json").write_text(json.dumps({"corpus": "c", "results": {"a:spaced": False}}))

    assert main(["--compare", str(tmp_path / "before.json"), str(tmp_path / "after.json")]) == 1
    assert "a:spaced" in capsys.readouterr().out


@pytest.mark.L1
def test_compare_mode_exits_zero_when_nothing_was_lost(tmp_path):
    (tmp_path / "before.json").write_text(json.dumps({"corpus": "c", "results": {"a:spaced": True}}))
    (tmp_path / "after.json").write_text(json.dumps({"corpus": "c", "results": {"a:spaced": True}}))

    assert main(["--compare", str(tmp_path / "before.json"), str(tmp_path / "after.json")]) == 0


def _write_list(directory, words):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "list").write_text("".join(f"{word}\n" for word in words), encoding="utf-8")
    return str(directory)


@pytest.mark.L1
def test_removing_an_entry_shows_as_lost_over_frozen_probes(tmp_path):
    """Removing an entry must not silently remove its probes from the comparison."""
    before_dir = _write_list(tmp_path / "before" / "custom", ["deskin", "batman"])
    after_dir = _write_list(tmp_path / "after" / "custom", ["deskin"])
    probes, before, after = (str(tmp_path / name) for name in ("probes.json", "before.json", "after.json"))

    assert main([before_dir, "--write-probes", probes]) == 0
    assert main([before_dir, "--probes", probes, "--json", before]) == 0
    assert main([after_dir, "--probes", probes, "--json", after]) == 0

    diff = compare(json.loads(open(before).read()), json.loads(open(after).read()))
    assert diff["comparable"] is True
    assert probe_id("batman", "verbatim") in diff["lost"]
    assert not any(key.startswith(probe_id("deskin", "")[:16]) for key in diff["lost"])
    assert main(["--compare", before, after]) == 1


@pytest.mark.L1
def test_removing_an_entry_without_frozen_probes_is_incomparable(tmp_path):
    """Each run building its own probes cannot hide the removed entry either."""
    before_dir = _write_list(tmp_path / "before" / "custom", ["deskin", "batman"])
    after_dir = _write_list(tmp_path / "after" / "custom", ["deskin"])
    before, after = str(tmp_path / "before.json"), str(tmp_path / "after.json")

    assert main([before_dir, "--json", before]) == 0
    assert main([after_dir, "--json", after]) == 0

    assert main(["--compare", before, after]) == 2


@pytest.mark.L1
def test_reordering_the_list_compares_cleanly(tmp_path):
    """Moving an entry changes nothing, because probes are not keyed by position."""
    before_dir = _write_list(tmp_path / "before" / "custom", ["deskin", "batman"])
    after_dir = _write_list(tmp_path / "after" / "custom", ["batman", "deskin"])
    before, after = str(tmp_path / "before.json"), str(tmp_path / "after.json")

    assert main([before_dir, "--json", before]) == 0
    assert main([after_dir, "--json", after]) == 0

    assert main(["--compare", before, after]) == 0


@pytest.mark.L1
def test_an_edited_probe_file_is_refused(tmp_path):
    custom = _write_list(tmp_path / "custom", ["deskin"])
    probes = tmp_path / "probes.json"
    assert main([custom, "--write-probes", str(probes)]) == 0

    frozen = json.loads(probes.read_text())
    frozen["probes"].popitem()
    probes.write_text(json.dumps(frozen))

    assert main([custom, "--probes", str(probes)]) == 2


@pytest.mark.L1
def test_sweep_over_a_synthetic_list_runs_the_deployed_path(tmp_path):
    """End to end on a small list, so the machinery is exercised without a checkpoint."""
    custom = _write_list(tmp_path / "custom", ["deskin"])

    assert main([custom, "--json", str(tmp_path / "out.json")]) == 0

    written = json.loads((tmp_path / "out.json").read_text())
    assert written["results"][probe_id("deskin", "verbatim")] is True
    assert written["corpus"] == corpus_id(build_probes(["deskin"]))
    assert written["summary"]["total"]["blocked"] >= 1


@pytest.mark.L1
def test_sweep_uses_the_exact_match_list_beside_the_blocklist(tmp_path):
    """The sweep runs is_safe, so an entry only on the exact-match list still counts."""
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "list").write_text("deskin\n", encoding="utf-8")
    (tmp_path / "exact_match").mkdir()
    (tmp_path / "exact_match" / "list").write_text("chromebook\n", encoding="utf-8")

    _, is_blocked, _ = _build_is_blocked(str(tmp_path / "custom"), None, None)

    assert is_blocked("a photo of chromebook in the scene")
    assert not is_blocked("a photo of a laptop in the scene")


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("no results key", {"summary": {}}),
        ("empty results", {"corpus": "c", "results": {}}),
        ("no corpus", {"results": {"a:spaced": True}}),
        ("not an object", ["a:spaced"]),
    ],
)
def test_compare_refuses_a_file_that_is_not_a_run(tmp_path, capsys, name, content):
    """Found in review: a malformed file must not crash into the "lost" exit code, or compare clean."""
    good = {"corpus": "c", "results": {"a:spaced": True}}
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text(json.dumps(content))

    assert main(["--compare", str(tmp_path / "good.json"), str(tmp_path / "bad.json")]) == 2, name
    assert main(["--compare", str(tmp_path / "bad.json"), str(tmp_path / "good.json")]) == 2, name
    assert "incomparable" in capsys.readouterr().out


@pytest.mark.L1
def test_case_variants_of_an_entry_keep_their_own_probes():
    """Found in review: folding case in the id collapsed entries whose probes differ."""
    probes = build_probes(["Batman", "batman"])

    assert len(probes) == 2 * len(spellings("batman"))


@pytest.mark.L1
def test_a_run_records_the_lists_it_measured(tmp_path):
    custom = _write_list(tmp_path / "custom", ["deskin"])
    whitelist = _write_list(tmp_path / "whitelist", ["flat"])

    assert main([custom, "--whitelist-dir", whitelist, "--json", str(tmp_path / "out.json")]) == 0

    setup = json.loads((tmp_path / "out.json").read_text())["setup"]
    assert set(setup) == {"blocklist", "whitelist", "exact_match", "lemmatizer"}
    assert "deskin" not in json.dumps(setup)
