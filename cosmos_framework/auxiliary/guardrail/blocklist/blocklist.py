# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import os
import re
import unicodedata
from difflib import SequenceMatcher

import nltk
from better_profanity import profanity

from cosmos_framework.auxiliary.guardrail.blocklist.lookalikes import fold_lookalikes
from cosmos_framework.auxiliary.guardrail.blocklist.utils import read_keyword_list_from_dir, to_ascii
from cosmos_framework.auxiliary.guardrail.common.core import (
    GUARDRAIL1_CHECKPOINT,
    ContentSafetyGuardrail,
    GuardrailRunner,
)
from cosmos_framework.utils import log, misc

# Sentinel marking characters that the censor replaced. It must be a character
# that cannot appear in ordinary text, because censorship is detected by looking
# for it. Do not colourise it: termcolor strips ANSI escapes when stdout is not a
# TTY, which reduced the sentinel to a bare "*" in any run whose output was piped
# to a file, so text containing Markdown emphasis ("**bold**") was reported as
# blocked even when the blocklist matched nothing.
CENSOR_SENTINEL = "\x00"

# Displayed to the user in the "Censored Prompt" message. Presentation only --
# never used to decide whether something was censored.
CENSOR = misc.Color.red("*")

# Characters that occupy no width, or that only reorder what is drawn: zero-width
# spaces and joiners, the soft hyphen, and the bidirectional controls. They render
# as nothing, so a blocklist entry split by one still reads as the entry to a human
# while reaching the matcher as a token it does not recognise.
INVISIBLE_CHARS = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")


class _NoLemmatizer:
    """Leaves every token as it is. The default for Blocklist.from_lists."""

    @staticmethod
    def lemmatize(token: str) -> str:
        return token


class Blocklist(ContentSafetyGuardrail):
    def __init__(
        self,
        guardrail_partial_match_min_chars: int = 6,
        guardrail_partial_match_letter_count: float = 0.4,
    ) -> None:
        """Blocklist model for text filtering safety check.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            guardrail_partial_match_min_chars (int, optional): Minimum number of characters in a word to check for partial match. Defaults to 6.
            guardrail_partial_match_letter_count (float, optional): Maximum allowed difference in characters for partial match. Defaults to 0.4.
        """
        self.checkpoint_dir = os.path.join(GUARDRAIL1_CHECKPOINT.download(), "blocklist")
        nltk.data.path.append(os.path.join(self.checkpoint_dir, "nltk_data"))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.profanity = profanity
        self.guardrail_partial_match_min_chars = guardrail_partial_match_min_chars
        self.guardrail_partial_match_letter_count = guardrail_partial_match_letter_count

        self.load_lists(
            blocklist_words=read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "custom")),
            whitelist_words=read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "whitelist")),
            exact_match_words=read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "exact_match")),
        )

    @classmethod
    def from_lists(
        cls,
        blocklist_words: list[str],
        whitelist_words: list[str] | None = None,
        exact_match_words: list[str] | None = None,
        lemmatizer=None,
        guardrail_partial_match_min_chars: int = 6,
        guardrail_partial_match_letter_count: float = 0.4,
    ) -> "Blocklist":
        """Build a Blocklist from keyword lists in memory, without the checkpoint.

        For the tests and the list-maintenance tools. It goes through load_lists
        like __init__ does, so callers exercise the production bookkeeping rather
        than a copy of it. The matcher is a fresh instance, not the module-level
        singleton production shares. ``lemmatizer`` defaults to leaving tokens
        unchanged; pass nltk's WordNetLemmatizer to match production exactly.
        """
        from better_profanity.better_profanity import Profanity

        blocklist = cls.__new__(cls)
        blocklist.profanity = Profanity()
        blocklist.lemmatizer = lemmatizer if lemmatizer is not None else _NoLemmatizer()
        blocklist.guardrail_partial_match_min_chars = guardrail_partial_match_min_chars
        blocklist.guardrail_partial_match_letter_count = guardrail_partial_match_letter_count
        blocklist.load_lists(blocklist_words, whitelist_words or [], exact_match_words or [])
        return blocklist

    def load_lists(self, blocklist_words: list[str], whitelist_words: list[str], exact_match_words: list[str]) -> None:
        """Load the keyword lists into the matcher.

        The single place that turns lists into matcher state, shared by __init__
        and from_lists.
        """
        self.blocklist_words = blocklist_words
        self.whitelist_words = whitelist_words
        self.exact_match_words = exact_match_words

        self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=self.whitelist_words)
        log.debug(f"Loaded {len(self.blocklist_words)} words/phrases from blocklist")
        log.debug(f"Whitelisted {len(self.whitelist_words)} words/phrases from whitelist")
        log.debug(f"Loaded {len(self.exact_match_words)} exact match words/phrases from blocklist")

    @staticmethod
    def normalize_for_matching(prompt: str) -> str:
        """Fold away the ways a word can be written to look normal but not match.

        The matcher compares whitespace-delimited tokens, so anything that splits
        a word without being visible, or that spells its letters with a different
        code point, walks past it while the prompt still reads as the blocked word.
        Three foldings, all of them lossless as far as a reader is concerned:

        * compatibility normalization (NFKC), which maps fullwidth and other
          presentation forms onto the ASCII letters they are drawn as;
        * removal of combining marks, so an accent added to a letter does not
          make a new word;
        * invisible characters rewritten to a space, then runs of whitespace
          collapsed, so a word split by any of them is matched as the split it
          renders as.

        The collapse also closes a plainer hole: a multi-word entry was defeated
        by typing two spaces between its words.

        The code-point folding is a separate method because is_safe also needs it
        on its own: see alternate_spellings.
        """
        return Blocklist.fold_invisible_characters(Blocklist.normalize_code_points(prompt))

    @staticmethod
    def normalize_code_points(prompt: str) -> str:
        """NFKC, combining marks removed, runs of whitespace collapsed."""
        prompt = unicodedata.normalize("NFKC", prompt)
        prompt = "".join(c for c in unicodedata.normalize("NFKD", prompt) if not unicodedata.combining(c))
        return " ".join(prompt.split())

    @staticmethod
    def fold_invisible_characters(prompt: str) -> str:
        """Rewrite zero-width and bidirectional characters to a space."""
        return " ".join(INVISIBLE_CHARS.sub(" ", prompt).split())

    def censor_prompt(self, input_prompt: str) -> tuple[bool, str]:
        """Censor the prompt using the blocklist with better-profanity fuzzy matching.

        Args:
            input_prompt: input prompt to censor

        Returns:
            bool: True if the prompt is blocked, False otherwise
            str: A message indicating why the prompt was blocked
        """
        # Strip any sentinel already present in the input. to_ascii preserves
        # \x00 (its range is [^\x00-\x7F]), so without this an input carrying a
        # NUL would be reported as censored even with no blocklist match -- the
        # same in-band-marker failure this sentinel replaced. Removing rather
        # than substituting keeps the stricter reading: "n\x00ike" fuses back to
        # a blocked word instead of being split into two harmless tokens.
        input_prompt = input_prompt.replace(CENSOR_SENTINEL, "")
        input_prompt = self.normalize_for_matching(input_prompt)
        # The whitelist is handed to load_censor_words(), so the matcher already
        # leaves whitelisted words alone. Restoring them a second time here is
        # what introduced the token misalignment described in the commit message.
        censored_prompt = self.profanity.censor(input_prompt, censor_char=CENSOR_SENTINEL)
        if CENSOR_SENTINEL in censored_prompt:
            display_prompt = censored_prompt.replace(CENSOR_SENTINEL, CENSOR)
            return True, f"Prompt blocked by censorship: Censored Prompt: {display_prompt}"
        return False, ""

    @staticmethod
    def check_partial_match(
        normalized_prompt: str, normalized_word: str, guardrail_partial_match_letter_count: float
    ) -> tuple[bool, str]:
        """
        Check robustly if normalized word and the matching target have a difference of up to guardrail_partial_match_letter_count characters.

        Args:
            normalized_prompt: a string with many words
            normalized_word: a string with one or multiple words, its length is smaller than normalized_prompt
            guardrail_partial_match_letter_count: maximum allowed difference in characters (float to allow partial characters)

        Returns:
            bool: True if a match is found, False otherwise
            str: A message indicating why the prompt was blocked
        """
        prompt_words = normalized_prompt.split()
        word_length = len(normalized_word.split())
        max_similarity_ratio = (len(normalized_word) - float(guardrail_partial_match_letter_count)) / float(
            len(normalized_word)
        )

        seq_matcher = SequenceMatcher(None)
        seq_matcher.set_seq2(normalized_word)

        for i in range(len(prompt_words) - word_length + 1):
            # Extract a substring from the prompt with the same number of words as the normalized_word
            substring = " ".join(prompt_words[i : i + word_length])
            seq_matcher.set_seq1(substring)

            # real_quick_ratio and quick_ratio are faster than ratio and both serve as upper bound for similarity ratio.
            # If they are less than max_similarity_ratio, it means that also the ratio will be less than max_similarity_ratio and we can skip the expensive ratio computation.
            # This saves a lot of time because in practice the tested words are usually dissimilar.
            # For details see: https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
            if (
                seq_matcher.real_quick_ratio() < max_similarity_ratio
                or seq_matcher.quick_ratio() < max_similarity_ratio
            ):
                continue

            similarity_ratio = seq_matcher.ratio()
            if similarity_ratio >= max_similarity_ratio:
                return (
                    True,
                    f"Prompt blocked by partial match blocklist: Prompt: {normalized_prompt}, Partial Match Word: {normalized_word}",
                )

        return False, ""

    @staticmethod
    def check_against_whole_word_blocklist(
        prompt: str,
        blocklist: list[str],
        guardrail_partial_match_min_chars: int = 6,
        guardrail_partial_match_letter_count: float = 0.4,
    ) -> tuple[bool, str]:
        """
        Check if the prompt contains any whole words from the blocklist.
        The match is case insensitive and robust to multiple spaces between words.

        Args:
            prompt: input prompt to check
            blocklist: list of words to check against
            guardrail_partial_match_min_chars: minimum number of characters in a word to check for partial match
            guardrail_partial_match_letter_count: maximum allowed difference in characters for partial match

        Returns:
            tuple[bool, str]: (True if a match is found, False otherwise), message indicating why the prompt was blocked
        """
        # Normalize spaces and convert to lowercase
        normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()

        normalized_words_cache = set()

        for word in blocklist:
            # Normalize spaces and convert to lowercase for each blocklist word
            normalized_word = re.sub(r"\s+", " ", word).strip().lower()

            if normalized_word in normalized_words_cache:
                continue

            normalized_words_cache.add(normalized_word)

            # Use word boundaries to ensure whole word match
            if re.search(r"\b" + re.escape(normalized_word) + r"\b", normalized_prompt):
                return True, f"Prompt blocked by exact match blocklist: Prompt: {prompt}, Exact Match Word: {word}"

        # Roughly 3/4 of the time this function requires is spent on partial matching.
        # We could use just one for loop to check both exact and partial matches but doing it in two loops is faster in practice
        # because it delays the partial matching as long as possible with a chance of early exit due to exact match.
        # Above we cache the normalized words and here we reuse them in the second loop for partial matching.

        for normalized_word in normalized_words_cache:
            # Check for partial match if the word is long enough
            if len(normalized_word) >= guardrail_partial_match_min_chars:
                match, message = Blocklist.check_partial_match(
                    normalized_prompt, normalized_word, guardrail_partial_match_letter_count
                )
                if match:
                    return True, message

        return False, ""

    def is_safe(self, input_prompt: str = "") -> tuple[bool, str]:
        """Check if the input prompt is safe using the blocklist."""
        # Check if the input is empty
        if not input_prompt:
            return False, "Input is empty"

        safe, message = self.check_passes(input_prompt)
        if not safe or input_prompt.isascii():
            return safe, message

        # The passes run once more on each other way the prompt can be read.
        # Like the raw pass, these only add checks: a prompt blocked as written
        # is still blocked.
        for spelling in self.alternate_spellings(input_prompt):
            safe, message = self.check_passes(spelling)
            if not safe:
                return safe, message
        return safe, message

    @classmethod
    def alternate_spellings(cls, prompt: str) -> list[str]:
        """Other readings of a non-ASCII prompt that the passes must also see.

        * The code points folded (NFKC, combining marks removed). censor_prompt
          folds them itself, but the lemmatized pass and the exact-match check
          only see the prompt after to_ascii, which deletes a fullwidth or
          accented letter rather than folding it. Without this reading an entry
          that is only on the exact-match list is missed when spelled that way.
        * Invisible characters deleted rather than turned into a space. The
          space reading catches "snow<U+200B>white" as the entry "snow white",
          but splits "sn<U+200B>ow white" into "sn ow white", which matches
          nothing. Deleting them reads it as the entry.
        * Letters from other scripts drawn like Latin ones ("sn<U+043E>w", with
          a Cyrillic o) folded to Latin. They survive NFKC, and to_ascii deletes
          them, so every pass reads the word with a letter missing. NFKC runs
          first, so a word mixing fullwidth and lookalike letters is recognized
          as one disguised word.

        Each reading is returned only if it differs from the prompt.
        """
        normalized = cls.normalize_code_points(prompt)
        joined = " ".join(INVISIBLE_CHARS.sub("", normalized).split())
        spellings: list[str] = []
        for spelling in (normalized, joined, fold_lookalikes(normalized), fold_lookalikes(joined)):
            if spelling != prompt and spelling not in spellings:
                spellings.append(spelling)
        return spellings

    def check_passes(self, input_prompt: str) -> tuple[bool, str]:
        """Run every blocklist pass on one spelling of the prompt."""
        # An extra first pass sees the prompt as written. to_ascii rewrites every
        # non-ASCII run to a space, which erases exactly the spellings
        # censor_prompt was taught to fold: a fullwidth or accented letter is
        # gone rather than normalized, so the word it spells never reaches the
        # matcher and the prompt reads as clean. censor_prompt does its own
        # code-point folding, so it needs the original text to fold.
        censored, message = self.censor_prompt(input_prompt)
        if censored:
            return False, message

        # The ASCII pass is kept rather than replaced. Some separators survive the
        # folding -- "boston·dynamics" stays one token on the raw pass -- and
        # only to_ascii turns them into the space the entry needs. Adding a pass
        # means nothing that was blocked before can pass now.
        input_prompt = to_ascii(input_prompt)
        censored, message = self.censor_prompt(input_prompt)
        if censored:
            return False, message

        # Check lemmatized words for censored words
        tokens = nltk.word_tokenize(input_prompt)
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_prompt = " ".join(lemmas)
        censored, message = self.censor_prompt(lemmatized_prompt)
        if censored:
            return False, message

        # Check for exact match blocklist words
        censored, message = self.check_against_whole_word_blocklist(
            input_prompt,
            self.exact_match_words,
            self.guardrail_partial_match_min_chars,
            self.guardrail_partial_match_letter_count,
        )
        if censored:
            return False, message

        # If all these checks pass, the input is safe
        return True, "Input is safe"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    return parser.parse_args()


def main(args):
    blocklist = Blocklist()
    runner = GuardrailRunner(safety_models=[blocklist])
    with misc.timer("blocklist safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
