# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.hygiene import (
    audit_entries,
    build_reachability_probe,
    check_entry,
    colliding_splits,
    read_keyword_files,
)

# Stand-in for WordNet so the tests need no corpus download. Only the words the
# cases below split into are members.
VOCABULARY = {"de", "desk", "in", "skin", "skinned", "inning", "chrome", "book", "game", "boy", "a"}


def is_word(word: str) -> bool:
    return word in VOCABULARY


@pytest.mark.L1
def test_entry_that_splits_into_two_words_is_flagged():
    """The reported false positive: "desk" + "in" fuses back into the entry."""
    assert ("desk", "in") in colliding_splits("deskin", is_word)


@pytest.mark.L1
def test_longest_left_half_is_reported_first():
    """ "desk in" is the split prose produces; "de skin" is incidental."""
    splits = colliding_splits("deskin", is_word)
    assert splits[0] == ("desk", "in")
    assert ("de", "skin") in splits


@pytest.mark.L1
def test_entry_without_a_dictionary_split_is_not_flagged():
    assert colliding_splits("huracan", is_word) == []


@pytest.mark.L1
def test_multi_word_entries_are_never_flagged():
    """A phrase cannot be fused out of prose, so it has no collision shape."""
    assert colliding_splits("game boy", is_word) == []


@pytest.mark.L1
def test_single_letter_halves_do_not_count():
    """ "a" is a dictionary word; counting it would flag every entry."""
    assert colliding_splits("askin", is_word) == []


@pytest.mark.L1
def test_entry_is_lowercased_and_stripped():
    assert colliding_splits("  DeskIn  ", is_word) == colliding_splits("deskin", is_word)


@pytest.mark.L1
def test_reachability_probe_keeps_only_splits_the_matcher_blocks():
    """A split is a problem only if the deployed matcher fires on the spaced form."""
    is_reachable = build_reachability_probe(["deskin"])

    assert is_reachable("desk", "in") is True
    assert is_reachable("chrome", "book") is False


@pytest.mark.L1
def test_check_entry_drops_splits_that_the_matcher_does_not_block():
    """ "chromebook" is not on this list, so its split is theoretical here."""
    is_reachable = build_reachability_probe(["deskin"])

    assert check_entry("deskin", is_word, is_reachable=is_reachable) == [("desk", "in"), ("de", "skin")]
    assert check_entry("chromebook", is_word, is_reachable=is_reachable) == []


@pytest.mark.L1
def test_audit_deduplicates_case_variants():
    findings = audit_entries(["deskin", "DeskIn", "huracan"], is_word)

    assert [entry for entry, _ in findings] == ["deskin"]


@pytest.mark.L1
def test_read_keyword_files_skips_blanks_and_comments(tmp_path):
    (tmp_path / "list").write_text("deskin\n\n# a comment\nchromebook\n", encoding="utf-8")

    assert read_keyword_files(tmp_path) == ["deskin", "chromebook"]


@pytest.mark.L1
def test_probe_honours_a_whitelisted_phrase():
    """The remedy this tool prints must be visible to the tool's own probe.

    A whitelist entry that is a phrase never reaches better_profanity; exempting
    it is something censor_prompt does on top. A probe built on a bare matcher
    therefore kept reporting "desk in" as reachable after the phrase had been
    whitelisted -- telling the caller to do something that changed nothing.
    """
    without = build_reachability_probe(["deskin"])
    with_phrase = build_reachability_probe(["deskin"], ["desk in"])

    assert without("desk", "in") is True
    assert with_phrase("desk", "in") is False
    # Only that one spelling is exempt; the entry's other splits still fire.
    assert with_phrase("de", "skin") is True


@pytest.mark.L1
def test_gate_probe_includes_the_candidate_being_gated(tmp_path):
    """gate checks an entry that is not on the list yet, which is the point.

    Building the probe from the existing list alone means the matcher can never
    fire on the candidate's own split, so every split is discarded as
    unreachable and the gate passes anything handed to it.
    """
    from cosmos_framework.auxiliary.guardrail.blocklist import hygiene

    blocklist_dir = tmp_path / "custom"
    blocklist_dir.mkdir()
    (blocklist_dir / "list").write_text("snow white\n", encoding="utf-8")

    argv = ["--confirm-reachable", str(blocklist_dir), "gate", "deskin"]
    monkey_vocabulary = hygiene.wordnet_vocabulary

    try:
        hygiene.wordnet_vocabulary = lambda _dir=None: is_word
        assert hygiene.main(argv) == 1  # flagged, so non-zero exit
    finally:
        hygiene.wordnet_vocabulary = monkey_vocabulary
