# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.blocklist import Blocklist


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


def _blocklist_with_words(words: list[str]) -> Blocklist:
    """Build a Blocklist around a small word list, without downloading a checkpoint.

    censor_prompt only needs the profanity matcher and the whitelist, so the
    heavyweight __init__ (checkpoint download, nltk data) is bypassed here.
    """
    from better_profanity.better_profanity import Profanity

    bl = Blocklist.__new__(Blocklist)
    bl.profanity = Profanity()
    bl.profanity.load_censor_words(custom_words=words, whitelist_words=[])
    bl.whitelist_words = []
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
