# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import unicodedata

import pytest

from cosmos_framework.auxiliary.guardrail.blocklist.lookalikes import (
    LOOKALIKES,
    fold_lookalikes,
    format_table,
    table_from_confusables,
)

_CONFUSABLES_SAMPLE = """\
# confusables.txt excerpt; fields are source ; target ; type
043E ;\t006F ;\tMA\t# ( о → o ) CYRILLIC SMALL LETTER O → LATIN SMALL LETTER O
0410 ;\t0041 ;\tMA\t# ( А → A ) CYRILLIC CAPITAL LETTER A → LATIN CAPITAL LETTER A
FF4F ;\t006F ;\tMA\t# ( ｏ → o ) FULLWIDTH LATIN SMALL LETTER O → LATIN SMALL LETTER O
0031 ;\t006C ;\tMA\t# ( 1 → l ) DIGIT ONE → LATIN SMALL LETTER L
0448 ;\t0077 ;\tMA\t# ( ш → w ) CYRILLIC SMALL LETTER SHA → LATIN SMALL LETTER W
0444 ;\t0278 ;\tMA\t# ( ф → ɸ ) CYRILLIC SMALL LETTER EF → LATIN SMALL LETTER PHI
0433 0335 ;\t0072 0335 ;\tMA\t# two code points on each side
"""


@pytest.mark.L1
def test_table_from_confusables_keeps_single_letters_nfkc_misses():
    """Only one code point onto one Latin letter, and only where NFKC does not already fold it."""
    table = table_from_confusables(_CONFUSABLES_SAMPLE.splitlines())

    assert table == {"о": "o", "А": "a", "ш": "w"}


@pytest.mark.L1
def test_format_table_round_trips():
    """The vendored literal is what format_table prints, so regenerating it is exact."""
    namespace: dict = {}
    exec(format_table(LOOKALIKES), {"_expand": lambda d: {c: k for k, v in d.items() for c in v}}, namespace)

    assert namespace["LOOKALIKES"] == LOOKALIKES


@pytest.mark.L1
def test_vendored_table_shape():
    """Every key is one non-ASCII code point NFKC leaves alone; every value is a Latin letter."""
    assert len(LOOKALIKES) > 700
    for char, letter in LOOKALIKES.items():
        assert len(char) == 1 and not char.isascii(), repr(char)
        assert letter in "abcdefghijklmnopqrstuvwxyz", repr(letter)
        assert unicodedata.normalize("NFKC", char).lower() != letter, repr(char)
    assert LOOKALIKES["о"] == "o"
    assert LOOKALIKES["ο"] == "o"
    assert LOOKALIKES["а"] == "a"


@pytest.mark.L1
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("plain ascii text", "plain ascii text"),
        ("a snоw poster", "a snow poster"),
        ("расе", "pace"),
        ("сосна", "сосна"),
        ("café こんにちは", "café こんにちは"),
        ("snоw сосна", "snow сосна"),
    ],
)
def test_fold_lookalikes_folds_only_disguised_words(prompt, expected):
    """A word is folded when it mixes in Latin letters or is made only of lookalikes."""
    assert fold_lookalikes(prompt) == expected
