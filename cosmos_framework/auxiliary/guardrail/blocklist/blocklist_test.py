# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import unicodedata

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


def _blocklist_with_words(words: list[str], whitelist: list[str] | None = None) -> Blocklist:
    """Build a Blocklist around a small word list, without downloading a checkpoint.

    censor_prompt only needs the profanity matcher and the whitelist, so the
    heavyweight __init__ (checkpoint download, nltk data) is bypassed here.
    """
    from better_profanity.better_profanity import Profanity

    whitelist = whitelist or []
    bl = Blocklist.__new__(Blocklist)
    bl.profanity = Profanity()
    bl.profanity.load_censor_words(custom_words=words, whitelist_words=whitelist)
    bl.whitelist_words = whitelist
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
def test_multi_word_match_beside_whitelisted_word_does_not_crash():
    """A multi-word blocklist hit must not make a later whitelisted word crash.

    Regression test: the censored text has one token per match, so a two-word
    entry made it one token shorter than the input. The whitelist restore walked
    the two lists by position, so a whitelisted word after the match indexed past
    the end of the censored list and raised IndexError on an ordinary prompt.
    """
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("Snow White is flat")

    assert blocked is True
    assert "is flat" in message


@pytest.mark.L1
def test_multi_word_match_does_not_rewrite_a_later_word():
    """The reported prompt must quote the input, not a shifted copy of it.

    Same off-by-N as above, but landing inside the list instead of past its end:
    the write went to the wrong token and silently replaced it.
    """
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("a Snow White poster on a flat wall")

    assert blocked is True
    assert "on a flat wall" in message
    assert "flat flat" not in message


@pytest.mark.L1
def test_several_multi_word_matches_stay_aligned():
    """Every additional multi-word hit shifts the two lists one token further."""
    bl = _blocklist_with_words(["snow white", "boston dynamics"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("Snow White and Boston Dynamics on a flat wall")

    assert blocked is True
    assert "on a flat wall" in message


@pytest.mark.L1
def test_whitelisted_word_is_not_censored():
    """The whitelist still works: a whitelisted word is reported as written."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("a Snow White poster on a flat desk")

    assert blocked is True
    assert "a flat desk" in message


@pytest.mark.L1
def test_whitelisted_word_alone_does_not_block():
    """A prompt with only whitelisted words is safe."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("the floor is flat")

    assert blocked is False
    assert message == ""


@pytest.mark.L1
def test_whitelisted_word_inside_a_blocked_phrase_stays_censored():
    """A whitelisted word must not dissolve a phrase that genuinely matched.

    'snow flat' is on the blocklist and 'flat' is whitelisted. The phrase is the
    match, so it stays censored; whitelisting a word does not license the phrase
    that contains it.
    """
    bl = _blocklist_with_words(["snow flat"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("a snow flat scene")

    assert blocked is True
    assert "snow flat" not in message


@pytest.mark.L1
def test_whitelisted_word_at_the_end_of_the_prompt():
    """The final token is where the positional walk ran off the end."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("a snow white poster flat")

    assert blocked is True
    assert message.endswith("poster flat")


@pytest.mark.L1
def test_repeated_multi_word_matches():
    """Two hits in a row shift the alignment twice over."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("snow white snow white flat wall")

    assert blocked is True
    assert "flat wall" in message


@pytest.mark.L1
def test_punctuation_around_a_whitelisted_word_is_preserved():
    """Punctuation belongs to the token and must survive into the message."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    blocked, message = bl.censor_prompt("snow white, flat, wall.")

    assert blocked is True
    assert "flat, wall." in message


@pytest.mark.L1
def test_extra_spaces_do_not_defeat_a_multi_word_entry():
    """A two-word entry was evaded by typing two spaces between its words."""
    bl = _blocklist_with_words(["snow white"])

    blocked, _ = bl.censor_prompt("a snow  white poster")

    assert blocked is True


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("zero width space", "a snow​white poster"),
        ("zero width non joiner", "a snow‌white poster"),
        ("soft hyphen", "a snow­white poster"),
        ("right to left override", "a snow‮white poster"),
        ("word joiner", "a snow⁠white poster"),
        ("byte order mark", "a snow﻿white poster"),
    ],
)
def test_invisible_characters_do_not_split_a_blocked_phrase(name, prompt):
    """An invisible character renders as nothing, so the prompt still reads as the entry.

    Splitting "snow white" with one of these produced a token the matcher did not
    recognise, while a reader saw the blocked phrase unchanged.
    """
    bl = _blocklist_with_words(["snow white"])

    blocked, _ = bl.censor_prompt(prompt)

    assert blocked is True, f"{name} was not folded away"


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("non breaking space", "a snow white poster"),
        ("en quad", "a snow white poster"),
        ("ideographic space", "a snow　white poster"),
        ("narrow no break space", "a snow white poster"),
    ],
)
def test_unusual_spaces_do_not_split_a_blocked_phrase(name, prompt):
    """Any Unicode space between the words of an entry must still match."""
    bl = _blocklist_with_words(["snow white"])

    blocked, _ = bl.censor_prompt(prompt)

    assert blocked is True, f"{name} was not folded away"


@pytest.mark.L1
def test_fullwidth_letters_are_folded_to_ascii():
    """Fullwidth forms are drawn as the ASCII letters they normalize to."""
    bl = _blocklist_with_words(["snow white"])

    blocked, _ = bl.censor_prompt("a ｓｎｏｗ white poster")

    assert blocked is True


@pytest.mark.L1
def test_combining_marks_do_not_make_a_new_word():
    """An accent added to a letter must not create a word the blocklist misses."""
    bl = _blocklist_with_words(["snow white"])

    for prompt in ("a snów white poster", unicodedata.normalize("NFD", "a snów white poster")):
        blocked, _ = bl.censor_prompt(prompt)
        assert blocked is True, f"{prompt!r} was not folded away"


@pytest.mark.L1
def test_normalization_does_not_block_ordinary_text():
    """Folding must not invent matches in text that contains none."""
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    for prompt in (
        "a robot on a flat desk",
        "an em dash — and an arrow → in the text",
        "café scene with Élodie",
        "snow  and  white  are  separate  words  here",
        "こんにちは from the model",
    ):
        blocked, message = bl.censor_prompt(prompt)
        assert blocked is False, f"{prompt!r} was wrongly reported as blocked: {message}"


@pytest.mark.L1
def test_normalization_preserves_the_earlier_sentinel_and_markdown_behaviour():
    """Folding runs after the sentinel strip and must not resurrect the "*" bug."""
    bl = _blocklist_with_words(["badword"])

    blocked, _ = bl.censor_prompt("A **bold** heading with ​ in it")
    assert blocked is False

    blocked, message = bl.censor_prompt("a bad\x00word in the text")
    assert blocked is True
    assert "badword" not in message
