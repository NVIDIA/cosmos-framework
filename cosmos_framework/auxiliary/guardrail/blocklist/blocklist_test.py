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


class _IdentityLemmatizer:
    """Stand-in for WordNetLemmatizer, so a test needs no downloaded corpus.

    The lemmatizer belongs to the second pass, which re-checks the prompt with
    words reduced to their stems. Nothing here tests lemmatization, and taking a
    dependency on nltk data would make these tests fail in any environment
    without the checkpoint's nltk_data directory.
    """

    @staticmethod
    def lemmatize(token: str) -> str:
        return token


def _blocklist_with_words(words: list[str], whitelist: list[str] | None = None) -> Blocklist:
    """Build a Blocklist around a small word list, without downloading a checkpoint.

    The heavyweight __init__ (checkpoint download, nltk data) is bypassed, but
    from_lists loads the lists through the same code as __init__, so a test
    exercises the real bookkeeping and can call the real entry point, is_safe.
    """
    return Blocklist.from_lists(
        words,
        whitelist,
        lemmatizer=_IdentityLemmatizer(),
        guardrail_partial_match_min_chars=100,
    )


def _blocked(bl: Blocklist, prompt: str) -> tuple[bool, str]:
    """Run the real entry point, is_safe, and report whether it blocked.

    is_safe rather than censor_prompt: is_safe also runs to_ascii, a lemmatized
    pass and the exact-match check, and a hole in any of them does not show up
    when censor_prompt is called on its own.
    """
    safe, message = bl.is_safe(prompt)
    return not safe, message


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

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} was not folded away"


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

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} was not folded away"


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("zero width space", "a sn\u200bow white poster"),
        ("zero width joiner", "a snow wh\u200dite poster"),
        ("soft hyphen", "a sn\u00adow white poster"),
        ("several", "a s\u200bn\u200bo\u200bw white poster"),
        ("with a lookalike", "a sn\u200b\u043ew white poster"),
    ],
)
def test_invisible_characters_inside_one_word_of_a_phrase(name, prompt):
    """An invisible character inside one word of a multi-word entry must not hide it.

    Turning it into a space reads "sn<U+200B>ow white" as "sn ow white", which
    matches nothing, while a reader sees the entry. Deleting it is the other
    reading, and both are checked.
    """
    bl = _blocklist_with_words(["snow white"])

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} let the entry through"


@pytest.mark.L1
def test_invisible_characters_between_words_still_read_as_a_space():
    """Deleting invisible characters is an extra reading; the space reading stays."""
    bl = _blocklist_with_words(["snow white"])

    safe, _ = bl.is_safe("a snow\u200bwhite poster")

    assert safe is False


@pytest.mark.L1
def test_deleting_invisible_characters_does_not_block_ordinary_text():
    """Text that uses invisible characters legitimately must still come back safe."""
    bl = _blocklist_with_words(["snow white", "deskin"])

    for prompt in (
        "a family emoji \U0001f468\u200d\U0001f469\u200d\U0001f467 on a desk",
        "a long\u00adword with a soft hyphen",
        "two\u200bwords joined by a zero width space",
    ):
        safe, message = bl.is_safe(prompt)
        assert safe is True, f"{prompt!r} was wrongly reported as blocked: {message}"


class _PluralLemmatizer:
    """Strips a trailing "s", the way WordNet reduces "dynamics" to "dynamic".

    With the identity stand-in, the lemmatized pass re-checks the ASCII prompt
    unchanged and catches anything the first pass missed. An entry whose words
    lemmatize to something else, such as "boston dynamics", gets no such rescue,
    which is the case these tests need.
    """

    @staticmethod
    def lemmatize(token: str) -> str:
        return token[:-1] if token.endswith("s") else token


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("middle dot", "a boston·dynamics robot"),
        ("bullet", "a boston•dynamics robot"),
        ("arrow", "a boston→dynamics robot"),
    ],
)
def test_separators_the_folding_keeps_still_split_a_blocked_phrase(name, prompt):
    """A separator NFKC leaves alone reaches the raw pass as part of one token.

    Only to_ascii turns it into the space the entry needs, so the ASCII pass has
    to run as well as the raw one, not instead of it.
    """
    bl = _blocklist_with_words(["boston dynamics"])
    bl.lemmatizer = _PluralLemmatizer()

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} separator let the entry through"


@pytest.mark.L1
def test_fullwidth_letters_are_folded_to_ascii():
    """Fullwidth forms are drawn as the ASCII letters they normalize to."""
    bl = _blocklist_with_words(["snow white"])

    safe, _ = bl.is_safe("a ｓｎｏｗ white poster")

    assert safe is False


@pytest.mark.L1
def test_combining_marks_do_not_make_a_new_word():
    """An accent added to a letter must not create a word the blocklist misses."""
    bl = _blocklist_with_words(["snow white"])

    for prompt in ("a snów white poster", unicodedata.normalize("NFD", "a snów white poster")):
        safe, _ = bl.is_safe(prompt)
        assert safe is False, f"{prompt!r} was not folded away"


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
def test_first_pass_on_raw_non_ascii_does_not_block_ordinary_text(monkeypatch):
    """The first pass now sees the prompt before to_ascii, so pin what that means.

    Running censor_prompt ahead of to_ascii is what lets the matcher fold a
    fullwidth or accented spelling. The cost is that non-ASCII text no longer
    arrives flattened to spaces, so ordinary prose in another script -- or with
    an accent, an em dash, a CJK sentence -- reaches the matcher as written and
    must still come back safe.

    word_tokenize is stubbed to a split so the test does not depend on the
    nltk data that ships in the checkpoint directory; the second pass is not
    what is under test here.
    """
    monkeypatch.setattr(
        "cosmos_framework.auxiliary.guardrail.blocklist.blocklist.nltk.word_tokenize",
        str.split,
    )
    bl = _blocklist_with_words(["snow white"], whitelist=["flat"])

    for prompt in (
        "café scene with Élodie",
        "an em dash — and an arrow → in the text",
        "こんにちは from the model",
        "a robot on a flat desk",
        "Grüße aus München",
    ):
        safe, message = bl.is_safe(prompt)
        assert safe is True, f"{prompt!r} was wrongly reported as blocked: {message}"


# Written as escapes: the point of these letters is that they cannot be told
# apart from Latin ones on screen.
_CYRILLIC_A = "\u0430"
_CYRILLIC_O = "\u043e"
_GREEK_O = "\u03bf"


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "entry", "prompt"),
    [
        ("cyrillic o", "snow white", f"a sn{_CYRILLIC_O}w white poster"),
        ("greek o", "snow white", f"a sn{_GREEK_O}w white poster"),
        ("capital cyrillic o", "snow white", "a SN\u041eW WHITE poster"),
        ("single-word entry", "batman", f"a b{_CYRILLIC_A}tman toy"),
        ("every letter swapped", "pace", "a \u0440\u0430\u0441\u0435 car"),
        ("mixed with fullwidth", "snow white", f"a \uff53\uff4e{_CYRILLIC_O}\uff57 white poster"),
        ("mixed with an invisible split", "batman", f"a b\u200b{_CYRILLIC_A}tman toy"),
    ],
)
def test_lookalike_letters_do_not_spell_a_new_word(name, entry, prompt):
    """A letter from another script drawn like a Latin one must not hide the entry.

    NFKC leaves these letters alone and to_ascii deletes them, so every other
    pass reads the word with a letter missing.
    """
    bl = _blocklist_with_words([entry])

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} let the entry through"


@pytest.mark.L1
def test_lookalike_folding_leaves_real_words_alone():
    """A word with letters that have no Latin twin is real text, not a disguise.

    The Russian word for "pine" starts with three lookalikes (c, o, c) and then
    has letters with no Latin twin. Folding the lookalikes anyway, and letting
    to_ascii drop the rest, leaves Latin fragments that fuse into the entry.
    """
    bl = _blocklist_with_words(["coca"])

    for prompt in ("\u0441\u043e\u0441\u043d\u0430 \u0443 \u043e\u043a\u043d\u0430", "\u0441\u043e\u0441\u043d\u0430"):
        safe, message = bl.is_safe(prompt)
        assert safe is True, f"{prompt!r} was wrongly reported as blocked: {message}"


@pytest.mark.L1
def test_ascii_prompt_skips_the_lookalike_pass(monkeypatch):
    """English prompts are untouched: the lookalike pass never runs on ASCII."""

    def fail(prompt):
        raise AssertionError(f"lookalike pass ran on {prompt!r}")

    monkeypatch.setattr("cosmos_framework.auxiliary.guardrail.blocklist.blocklist.fold_lookalikes", fail)
    bl = _blocklist_with_words(["snow white"])

    assert bl.is_safe("a robot sitting at a table")[0] is True
    assert bl.is_safe("a snow white poster")[0] is False


@pytest.mark.L1
@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("arrow", "a snow→white poster"),
        ("emoji", "a snow🙂white poster"),
        ("ellipsis", "a snow…white poster"),
        ("middle dot", "a snow·white poster"),
    ],
)
def test_non_ascii_separator_is_caught_by_the_flattened_pass(monkeypatch, name, prompt):
    """The two passes cover different evasions, and both are needed.

    A visible non-ASCII character between the words of an entry is not an
    invisible mark, so normalize_for_matching leaves it in place and the first
    pass -- which now sees raw text -- does not match. to_ascii rewrites it to a
    space, so the second pass does. The reverse holds for a fullwidth or
    accented spelling, which only the first pass can fold.

    Neither pass is redundant: dropping either one opens the family the other
    covers. This test fails if the flattened pass is ever removed or moved
    ahead of the first.
    """
    monkeypatch.setattr(
        "cosmos_framework.auxiliary.guardrail.blocklist.blocklist.nltk.word_tokenize",
        str.split,
    )
    bl = _blocklist_with_words(["snow white"])

    safe, _ = bl.is_safe(prompt)

    assert safe is False, f"{name} separator was not caught by either pass"


@pytest.mark.L1
def test_normalization_preserves_the_earlier_sentinel_and_markdown_behaviour():
    """Folding runs after the sentinel strip and must not resurrect the "*" bug."""
    bl = _blocklist_with_words(["badword"])

    blocked, _ = bl.censor_prompt("A **bold** heading with ​ in it")
    assert blocked is False

    blocked, message = bl.censor_prompt("a bad\x00word in the text")
    assert blocked is True
    assert "badword" not in message
