# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.blocklist import Blocklist, exempt_fused_prose_default


@pytest.mark.L1
def test_exact_match():
    """Test exact word matching."""
    prompt = "this contains badword in the middle"
    word = "badword"

    match, message = Blocklist.check_partial_match(prompt, word, 0.4)

    assert match is True
    assert "badword" in message


@pytest.mark.L1
def test_no_match():
    """Test when there's no match."""
    prompt = "this is a clean prompt"
    word = "badword"

    match, message = Blocklist.check_partial_match(prompt, word, 0.4)

    assert match is False
    assert message == ""


@pytest.mark.L1
def test_partial_match_with_threshold():
    """Test partial matching with character difference threshold."""
    prompt = "this contains badw0rd here"  # One character different
    word = "badword"

    # With threshold of 1.0 character difference, should match
    match, _ = Blocklist.check_partial_match(prompt, word, 1.0)
    assert match is True

    # With threshold of 0.5 character difference, should not match
    match, _ = Blocklist.check_partial_match(prompt, word, 0.5)
    assert match is False


def _blocklist_with_words(words: list[str], dictionary: dict | None = None) -> Blocklist:
    """Build a Blocklist around a small word list, without downloading a checkpoint.

    censor_prompt needs only the matcher, the whitelist and the fused-join
    bookkeeping, so the heavyweight __init__ (checkpoint download, nltk data) is
    bypassed. The dictionary is injected rather than read from WordNet so the
    tests stay hermetic and exercise the heuristic directly instead of whatever
    the corpus happens to contain.
    """
    from better_profanity.better_profanity import Profanity

    bl = Blocklist.__new__(Blocklist)
    bl.profanity = Profanity()
    bl.profanity.load_censor_words(custom_words=words, whitelist_words=[])
    bl.whitelist_words = []
    bl.blocklist_words = words
    bl.guardrail_exempt_fused_prose = True
    # Same derivation production uses, so the join bookkeeping under test is not
    # a reimplementation of it.
    bl._configure_join_bookkeeping()
    bl._dictionary_cache.update(dictionary or {})
    return bl


@pytest.mark.L1
def test_markdown_emphasis_is_not_treated_as_censorship():
    """Text containing Markdown emphasis must not be reported as blocked.

    Regression test: censorship used to be detected by searching the censored
    output for a colourised "*". termcolor strips the colour when stdout is not
    a TTY, leaving a bare "*", so any "**bold**" in the input matched the
    sentinel and the prompt was blocked despite no blocklist hit.
    """
    bl = _blocklist_with_words(["badword"])

    blocked, message = bl.censor_prompt("The robot moves a **box** across the floor.")

    assert blocked is False
    assert message == ""


@pytest.mark.L1
def test_asterisks_alone_do_not_trigger_a_block():
    """Bare asterisks are ordinary characters, not evidence of censorship."""
    bl = _blocklist_with_words(["badword"])

    for prompt in ("a * b", "**", "5 * 3 = 15", "*emphasis* and **strong**"):
        blocked, _ = bl.censor_prompt(prompt)
        assert blocked is False, f"{prompt!r} was wrongly reported as blocked"


@pytest.mark.L1
def test_sentinel_in_the_input_does_not_trigger_a_block():
    """A NUL arriving in the input must not be mistaken for censorship.

    to_ascii only replaces [^\\x00-\\x7F], so \\x00 survives normalization and
    would otherwise reach the sentinel check unmodified.
    """
    bl = _blocklist_with_words(["badword"])

    blocked, message = bl.censor_prompt("a clean prompt with \x00 in it")

    assert blocked is False
    assert message == ""


@pytest.mark.L1
def test_sentinel_cannot_be_used_to_split_a_blocked_word():
    """Stripping the sentinel must not open an evasion: the word fuses back."""
    bl = _blocklist_with_words(["badword"])

    blocked, message = bl.censor_prompt("a bad\x00word in the text")

    assert blocked is True
    assert "badword" not in message


@pytest.mark.L1
def test_blocked_word_is_still_detected_alongside_markdown():
    """The fix must not weaken detection: a real hit still blocks, and still reports."""
    bl = _blocklist_with_words(["badword"])

    blocked, message = bl.censor_prompt("A **bold** heading and a badword in the text.")

    assert blocked is True
    assert "Censored Prompt:" in message
    # The Markdown the model emitted is preserved in the message; only the
    # blocked word is masked.
    assert "**bold**" in message
    assert "badword" not in message


@pytest.mark.L1
def test_adjacent_dictionary_words_do_not_form_a_blocked_word():
    """Ordinary prose must not match by having its spaces deleted.

    Regression test: better_profanity concatenates adjacent words with their
    separators removed to catch evasions, so "a desk in the background" formed
    "deskin" and tripped the blocklist.
    """
    bl = _blocklist_with_words(
        ["deskin"],
        {"a": True, "desk": True, "in": True, "the": True, "background": True},
    )

    blocked, _ = bl.censor_prompt("a desk in the background")

    assert blocked is False


@pytest.mark.L1
def test_evasion_by_splitting_a_word_is_still_caught():
    """The fix must not open an evasion route: a mid-word split still blocks.

    A part that is not an ordinary English word marks the join as deliberate
    rather than coincidental.
    """
    bl = _blocklist_with_words(
        ["toyota"], {"a": True, "to": True, "yota": False, "car": True}
    )

    for prompt in ("a to yota car", "a toyota car"):
        blocked, _ = bl.censor_prompt(prompt)
        assert blocked is True, f"{prompt!r} should be blocked"


@pytest.mark.L1
def test_single_letter_split_is_treated_as_evasion():
    """"n ike" blocks even though WordNet knows both "n" and "ike"."""
    bl = _blocklist_with_words(
        ["nike"], {"n": True, "ike": True, "shoes": True}
    )

    blocked, _ = bl.censor_prompt("n ike shoes")

    assert blocked is True


@pytest.mark.L1
def test_multi_word_blocklist_phrase_still_matches():
    """Entries that genuinely contain spaces keep matching with spacing intact."""
    bl = _blocklist_with_words(
        ["boston dynamics"],
        {"a": True, "boston": True, "dynamics": True, "robot": True, "walks": True},
    )

    blocked, _ = bl.censor_prompt("a Boston Dynamics robot walks past")

    assert blocked is True


@pytest.mark.L1
def test_single_token_match_across_punctuation_still_blocks():
    """Joins inside one token, e.g. "desk-in", are evasions and must still block."""
    bl = _blocklist_with_words(
        ["deskin"], {"the": True, "desk": True, "in": True, "corner": True}
    )

    blocked, _ = bl.censor_prompt("the desk-in the corner")

    assert blocked is True


@pytest.mark.L1
def test_leet_spelling_of_a_multi_word_entry_still_blocks():
    """Phrase entries are stored as the library stores them, so o->0 still matches.

    Regression test: holding them as plain strings let "b0ston dynamics" through
    while stock better_profanity blocked it.
    """
    bl = _blocklist_with_words(
        ["boston dynamics"],
        {"a": True, "boston": True, "dynamics": True, "robot": True},
    )

    for prompt in ("a boston dynamics robot", "a b0ston dynamics robot", "a boston dynamic5 robot"):
        blocked, _ = bl.censor_prompt(prompt)
        assert blocked is True, f"{prompt!r} should be blocked"


@pytest.mark.L1
def test_join_window_follows_the_library_reach():
    """The window comes from the matcher, not from the custom list's longest entry.

    Regression test: a local constant of 3 let a 4-token split escape while stock
    better_profanity blocked it.
    """
    bl = _blocklist_with_words(
        ["supercalifragil"],
        {"su": True, "per": True, "cali": True, "fragil": False, "now": True},
    )

    blocked, _ = bl.censor_prompt("su per cali fragil now")

    assert blocked is True


@pytest.mark.L1
def test_leet_characters_are_not_stripped_from_tokens():
    """Characters the library treats as letters must survive tokenisation.

    Regression test: string.punctuation overlaps ALLOWED_CHARACTERS on " $ ' * @,
    so stripping it emptied the "$" token and aborted the window scan, letting
    "wear $ ike shoes" through while stock blocked it.
    """
    bl = _blocklist_with_words(
        ["sike"], {"wear": True, "s": True, "ike": True, "shoes": True}
    )

    for prompt in ("wear s ike shoes", "wear $ ike shoes"):
        blocked, _ = bl.censor_prompt(prompt)
        assert blocked is True, f"{prompt!r} should be blocked"


@pytest.mark.L1
def test_environment_variable_selects_strict_mode(monkeypatch):
    """The stricter behaviour is reachable without editing code.

    presets.py builds Blocklist() with no arguments, so without an environment
    path the strict setting could only be chosen by editing source.
    """
    monkeypatch.delenv("COSMOS_GUARDRAIL_EXEMPT_FUSED_PROSE", raising=False)
    assert exempt_fused_prose_default() is True

    monkeypatch.setenv("COSMOS_GUARDRAIL_EXEMPT_FUSED_PROSE", "0")
    assert exempt_fused_prose_default() is False

    bl = _blocklist_with_words(
        ["deskin"],
        {"a": True, "desk": True, "in": True, "the": True, "background": True},
    )
    bl.guardrail_exempt_fused_prose = exempt_fused_prose_default()

    blocked, _ = bl.censor_prompt("a desk in the background")

    assert blocked is True


@pytest.mark.L1
def test_strict_mode_restores_the_previous_fused_behaviour():
    """guardrail_exempt_fused_prose=False gives up the fix for maximum strictness.

    Deployments that would rather block ordinary prose than ever let a fused match
    through can opt out; this pins that switch so it cannot rot.
    """
    bl = _blocklist_with_words(
        ["deskin"],
        {"a": True, "desk": True, "in": True, "the": True, "background": True},
    )
    bl.guardrail_exempt_fused_prose = False

    blocked, _ = bl.censor_prompt("a desk in the background")

    assert blocked is True
