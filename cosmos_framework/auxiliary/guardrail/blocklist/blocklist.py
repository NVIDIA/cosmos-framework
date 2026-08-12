# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import os
import re
import string
from difflib import SequenceMatcher

import nltk
from better_profanity import profanity
from better_profanity.constants import ALLOWED_CHARACTERS
from better_profanity.varying_string import VaryingString
from nltk.corpus import wordnet

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

# Punctuation that better_profanity does not treat as part of a word. The
# library's ALLOWED_CHARACTERS overlaps string.punctuation on " $ ' * @, which
# are exactly the characters it substitutes for letters (a->@, s->$, * for
# several vowels). Stripping those would erase a token the library still
# matches on, so only the remainder is stripped here.
_STRIP_CHARS = "".join(sorted(set(string.punctuation) - ALLOWED_CHARACTERS))

# Environment override for guardrail_exempt_fused_prose, so the stricter
# behaviour is selectable without editing code -- presets.py constructs
# Blocklist() with no arguments. Set to "0" to restore strict fused matching.
_EXEMPT_FUSED_PROSE_ENV = "COSMOS_GUARDRAIL_EXEMPT_FUSED_PROSE"


def exempt_fused_prose_default() -> bool:
    """Resolve the default for guardrail_exempt_fused_prose from the environment."""
    return os.environ.get(_EXEMPT_FUSED_PROSE_ENV, "1") != "0"


class Blocklist(ContentSafetyGuardrail):
    def __init__(
        self,
        guardrail_partial_match_min_chars: int = 6,
        guardrail_partial_match_letter_count: float = 0.4,
        guardrail_exempt_fused_prose: bool | None = None,
    ) -> None:
        """Blocklist model for text filtering safety check.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            guardrail_partial_match_min_chars (int, optional): Minimum number of characters in a word to check for partial match. Defaults to 6.
            guardrail_partial_match_letter_count (float, optional): Maximum allowed difference in characters for partial match. Defaults to 0.4.
            guardrail_exempt_fused_prose (bool, optional): When True, a match that
                exists only because the spaces between ordinary English words were
                deleted is treated as coincidence rather than evasion, so
                "a desk in the background" does not match the entry "deskin".
                Set False to restore the stricter previous behaviour, which blocks
                that sentence but never lets a fused match through. See
                `_has_legitimate_match` for the escape class this trades away.
                Defaults to the COSMOS_GUARDRAIL_EXEMPT_FUSED_PROSE environment
                variable, which itself defaults to True; set that variable to "0"
                to select strict matching without editing code.
        """
        if guardrail_exempt_fused_prose is None:
            guardrail_exempt_fused_prose = exempt_fused_prose_default()
        self.checkpoint_dir = os.path.join(GUARDRAIL1_CHECKPOINT.download(), "blocklist")
        nltk.data.path.append(os.path.join(self.checkpoint_dir, "nltk_data"))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.profanity = profanity
        self.guardrail_partial_match_min_chars = guardrail_partial_match_min_chars
        self.guardrail_partial_match_letter_count = guardrail_partial_match_letter_count
        self.guardrail_exempt_fused_prose = guardrail_exempt_fused_prose

        # Load blocklist and whitelist keywords
        self.blocklist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "custom"))
        self.whitelist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "whitelist"))
        self.exact_match_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "exact_match"))

        self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=self.whitelist_words)
        self._configure_join_bookkeeping()
        log.debug(f"Loaded {len(self.blocklist_words)} words/phrases from blocklist")
        log.debug(f"Whitelisted {len(self.whitelist_words)} words/phrases from whitelist")
        log.debug(f"Loaded {len(self.exact_match_words)} exact match words/phrases from blocklist")

    def _configure_join_bookkeeping(self) -> None:
        """Derive the fused-join state from the loaded blocklist and matcher.

        Called from __init__ and from the test helper, so tests exercise the
        values production actually runs with rather than a reimplementation.
        Requires self.profanity to be loaded and self.blocklist_words to be set.
        """
        self._dictionary_cache: dict[str, bool] = {}
        self._max_blocklist_words = max((len(w.split()) for w in self.blocklist_words), default=1)
        # Entries that genuinely contain spaces, so a spaced match can be told
        # apart from a match that only worked because the spaces were deleted.
        # Stored the way the library stores its own entries, so comparison
        # dispatches to VaryingString.__eq__ and leet spellings of a phrase
        # ("b0ston dynamics") still match.
        self._blocklist_phrases = [
            VaryingString(" ".join(w.lower().split()), char_map=self.profanity.CHARS_MAPPING)
            for w in self.blocklist_words
            if " " in w
        ]

    def uncensor_whitelist(self, input_prompt: str, censored_prompt: str) -> str:
        """Explicitly uncensor words that are in the whitelist."""
        input_words = input_prompt.split()
        censored_words = censored_prompt.split()
        whitelist_words = set(self.whitelist_words)
        for i, token in enumerate(input_words):
            if token.strip(string.punctuation).lower() in whitelist_words:
                censored_words[i] = token
        censored_prompt = " ".join(censored_words)
        return censored_prompt

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
        censored_prompt = self.profanity.censor(input_prompt, censor_char=CENSOR_SENTINEL)
        # Uncensor whitelisted words that were censored from blocklist fuzzy matching
        censored_prompt = self.uncensor_whitelist(input_prompt, censored_prompt)
        if CENSOR_SENTINEL not in censored_prompt:
            return False, ""
        # better_profanity also matches across word boundaries with the separators
        # deleted, so confirm a real match exists before blocking. Only reached
        # when something already matched, so the common path costs nothing.
        if not self._has_legitimate_match(input_prompt):
            return False, ""
        display_prompt = censored_prompt.replace(CENSOR_SENTINEL, CENSOR)
        return True, f"Prompt blocked by censorship: Censored Prompt: {display_prompt}"

    def _is_dictionary_word(self, word: str) -> bool:
        """True if the word is ordinary English, per the WordNet corpus.

        Used to tell an evasion apart from a coincidence. If the corpus is
        unavailable, report False so every join stays suspicious and the filter
        remains at least as strict as before.
        """
        if word not in self._dictionary_cache:
            try:
                self._dictionary_cache[word] = bool(wordnet.synsets(word))
            except LookupError:
                self._dictionary_cache[word] = False
        return self._dictionary_cache[word]

    def _is_prose_word(self, word: str) -> bool:
        """True if the word could plausibly stand alone in ordinary prose.

        Single letters are excluded even though WordNet knows them: a lone "f" or
        "n" beside another word is the shape of an evasion ("n ike"), not
        of normal writing.
        """
        return len(word) >= 2 and self._is_dictionary_word(word)

    def _fires(self, text: str) -> bool:
        """True if the matcher censors anything in `text`."""
        return self.profanity.censor(text, censor_char=CENSOR_SENTINEL) != text

    def _has_legitimate_match(self, input_prompt: str) -> bool:
        """Re-check a flagged prompt with word boundaries respected.

        better_profanity concatenates adjacent words with their separators removed
        (`any_next_words_form_swear_word`) so that a blocked word written with a space
        inserted mid-word is still caught -- "n ike" for the entry "nike". The
        side effect is that ordinary prose collides with short blocklist entries:
        "a desk in the background" forms "deskin".

        A match is legitimate when any of these holds:

        * a single whitespace-delimited token matches on its own -- this keeps
          matches that join across punctuation inside a token, such as "desk-in";
        * a run of tokens matches a blocklist entry that genuinely contains
          spaces, such as "Boston Dynamics";
        * a run of tokens matches only once the spaces are deleted AND at least
          one part is not an ordinary English word -- which is what an evasion
          looks like ("n ike", "to yota").

        A space-deleted join whose every part is a dictionary word is rejected as
        coincidence. Matching is delegated to the library throughout, so leet
        substitutions and punctuation are handled exactly as before.

        Known limits
        ------------
        This is a heuristic on top of a heuristic, and it is worth stating plainly
        what it does and does not buy.

        * **Escape class introduced here.** A banned word split into pieces that
          are *all* multi-letter dictionary words is no longer treated as an
          evasion. "assassin" written as "ass ass in" would pass. Set
          `guardrail_exempt_fused_prose=False` to give this up and go back to
          blocking every fused match, at the cost of blocking ordinary prose.
        * **The library's reach is wide.** With the entry "nike", stock code
          blocks "n ike", "ni ke", "nik e" and "n i k e" alike once a following
          word closes the window ("nik e shoe"). An earlier version of this note
          claimed the last two escaped; that was measured on fragments with no
          trailing token and is wrong. The exemption below therefore gives up
          more than first documented.
        * **WordNet is the trust boundary.** "Ordinary English" means "WordNet has
          a synset", which over-includes: it counts single letters such as "n" and
          "f" as words, which is why `_is_prose_word` requires two characters. A
          fragment that WordNet happens to know ("ike") is treated as ordinary.
        * **Layering matters.** The blocklist is a coarse pre-filter with model
          based guardrails (llamaGuard3, qwen3guard, video content safety) running
          alongside it. It is not the last line of defence, which is what makes
          this trade reasonable rather than reckless.
        """
        if not self.guardrail_exempt_fused_prose:
            return True

        whitelist = {w.lower() for w in self.whitelist_words}
        raw_tokens = input_prompt.split()
        # Strip only punctuation the library does not count as part of a word.
        # Stripping its leet characters would empty a token it still matches on,
        # and an empty token aborts the window scan below.
        tokens = [t.strip(_STRIP_CHARS).lower() for t in raw_tokens]

        for raw, token in zip(raw_tokens, tokens):
            if token and token not in whitelist and self._fires(raw):
                return True

        # Phrase matches are bounded by the longest blocklist entry, but a
        # separator-stripped join fuses several tokens into ONE entry word, so it
        # needs its own bound. Take that bound from the library rather than a
        # local constant: MAX_NUMBER_COMBINATIONS is monotonic across
        # load_censor_words calls, so the matcher's reach can exceed anything
        # derived from the custom list alone.
        max_window = max(self._max_blocklist_words, self.profanity.MAX_NUMBER_COMBINATIONS + 1)
        for i in range(len(tokens)):
            upper = min(max_window, len(tokens) - i)
            for size in range(2, upper + 1):
                window = tokens[i : i + size]
                if not all(window) or any(w in whitelist for w in window):
                    break
                if " ".join(window) in self._blocklist_phrases:
                    return True
                if self._fires("".join(window)) and not all(
                    self._is_prose_word(w) for w in window
                ):
                    return True
        return False

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
        input_prompt = to_ascii(input_prompt)

        # Check full sentence for censored words
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
