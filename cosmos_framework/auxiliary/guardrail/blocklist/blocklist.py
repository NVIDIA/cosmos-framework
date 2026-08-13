# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import os
import re
import unicodedata
from difflib import SequenceMatcher

import nltk
from better_profanity import profanity

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

# Placeholder a whitelisted phrase is replaced with before matching. It has to
# be a single token that no blocklist entry can fuse with or fuzzy-match, and it
# has to survive to_ascii, so it is plain lowercase ASCII with no separators.
WHITELIST_MASK_PREFIX = "zzwhitelisted"
WHITELIST_MASK_SUFFIX = "zz"


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

        # Load blocklist and whitelist keywords
        self.blocklist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "custom"))
        self.whitelist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "whitelist"))
        self.exact_match_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "exact_match"))

        # The matcher's whitelist is a set of single tokens, so a whitelisted
        # phrase never reaches it. Phrases are handled here instead, by masking
        # them out of the prompt before matching -- see mask_whitelisted_phrases.
        self.whitelist_phrases = sorted((w for w in self.whitelist_words if " " in w.strip()), key=len, reverse=True)
        whitelist_tokens = [w for w in self.whitelist_words if " " not in w.strip()]

        self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=whitelist_tokens)
        log.debug(f"Loaded {len(self.blocklist_words)} words/phrases from blocklist")
        log.debug(f"Whitelisted {len(whitelist_tokens)} words and {len(self.whitelist_phrases)} phrases")
        log.debug(f"Loaded {len(self.exact_match_words)} exact match words/phrases from blocklist")

        for token in self.unprotected_whitelist_tokens(self.profanity, whitelist_tokens):
            # better_profanity drops a whitelisted word before it expands the
            # leetspeak variants of the other entries, so a whitelisted word can
            # still be censored as some *other* entry's variant -- "flat" is what
            # "fiat" becomes under the i/l mapping. Whitelisting it then silently
            # does nothing, which is worth saying out loud at load time.
            log.warning(f"Whitelisted word is still censored as a variant of another blocklist entry: {token!r}")

    @staticmethod
    def unprotected_whitelist_tokens(matcher, whitelist_tokens: list[str]) -> list[str]:
        """Whitelisted words the loaded matcher censors anyway."""
        return [token for token in whitelist_tokens if matcher.censor(token, censor_char=CENSOR_SENTINEL) != token]

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

        The two foldings are separate methods because a whitelisted phrase has to
        be matched between them: see censor_prompt.
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

    def mask_whitelisted_phrases(self, prompt: str) -> tuple[str, dict[str, str]]:
        """Replace whitelisted phrases with placeholders, before matching.

        The matcher fuses adjacent words and tests the fused string against the
        blocklist, which is how "n ike" is caught. The same fusing means a
        blocklist entry spelled out of two ordinary words also matches the prose
        that contains them: "a desk in the background" fuses into an entry.

        Whitelisting the innocent halves cannot fix that -- the fusing does not
        consult the whitelist, and the halves are ordinary words that should not
        be exempt on their own. Whitelisting the *phrase* can: the placeholder
        does not fuse into anything, so exactly one spelling stops matching, and
        every other spelling of the entry ("d eskin", "de sk in", leetspeak) is
        still caught.

        Placeholders are single tokens, so the token count is unchanged and the
        censored output lines up with the input.

        This runs before invisible characters are folded to spaces, and that
        order matters: "desk<U+200B>in" renders as the entry with no space in it,
        and folding first would turn it into the whitelisted phrase and exempt an
        evasion that a reader cannot see. Matching the phrase while the invisible
        character is still there leaves that spelling blocked.
        """
        replacements = {}
        for index, phrase in enumerate(self.whitelist_phrases):
            placeholder = f"{WHITELIST_MASK_PREFIX}{index}{WHITELIST_MASK_SUFFIX}"
            pattern = re.compile(r"\b" + re.escape(phrase.strip()) + r"\b", re.IGNORECASE)
            prompt, count = pattern.subn(placeholder, prompt)
            if count:
                replacements[placeholder] = phrase
        return prompt, replacements

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
        # Whitelisted single words are handed to load_censor_words(), so the
        # matcher already leaves them alone; restoring them a second time after
        # censoring is what introduced the token misalignment removed earlier.
        # Whitelisted phrases never reach the matcher, so they are masked here --
        # between the two foldings, for the reason given in that method.
        input_prompt = self.normalize_code_points(input_prompt)
        input_prompt, masked_phrases = self.mask_whitelisted_phrases(input_prompt)
        input_prompt = self.fold_invisible_characters(input_prompt)
        censored_prompt = self.profanity.censor(input_prompt, censor_char=CENSOR_SENTINEL)
        if CENSOR_SENTINEL in censored_prompt:
            display_prompt = censored_prompt.replace(CENSOR_SENTINEL, CENSOR)
            for placeholder, phrase in masked_phrases.items():
                display_prompt = display_prompt.replace(placeholder, phrase)
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
