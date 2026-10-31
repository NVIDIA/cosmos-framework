# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Measure what a blocklist change stops catching.

A corpus of ordinary model output answers one question: does a change block
things it did not block before. Any change that *unblocks* -- a whitelist entry,
a data move, a new exemption mechanism -- carries the opposite risk, and a
corpus cannot show it: the phrasings that would newly slip through are exactly
the ones ordinary output does not contain.

This builds that missing side. For every entry it writes five prompts, one per
way the entry can be spelled while still reading as itself:

    verbatim    the entry as written
    spaced      split in the middle, the fusing evasion ("de skin")
    invisible   split with a zero-width space
    fullwidth   spelled in fullwidth letters
    lookalike   every letter that has one swapped for a Cyrillic or Greek twin

Freeze the probes once, run both versions over that same frozen set, and
compare: any prompt that was blocked and now is not is a spelling the change
gave up. A probe is identified by a digest of its entry plus the spelling, not
by the entry's position, so reordering the list does not change what is
compared, and an entry removed from the list keeps its probes in the frozen set
-- its spellings then show up as lost. Two result files that were not run over
the same probes are reported as incomparable, never as a clean comparison.
Entry text is never printed, so output can be pasted into a review. A frozen
probe file does hold the probe prompts, and so the entries, in plain text: keep
it with the lists, not in a ticket.

    coverage.py <blocklist-dir> --write-probes probes.json
    coverage.py <blocklist-dir> --probes probes.json --whitelist-dir <dir> --json before.json
    coverage.py <new-blocklist-dir> --probes probes.json --whitelist-dir <new-dir> --json after.json
    coverage.py --compare before.json after.json

The exact-match list and nltk data beside <blocklist-dir> are picked up on their
own; the whitelist is not, so pass --whitelist-dir to measure the production
configuration. Only blocklist entries are probed, not exact-match entries.
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

# Lowercase ASCII letter -> its fullwidth presentation form.
FULLWIDTH = {chr(code): chr(code - 0x61 + 0xFF41) for code in range(0x61, 0x7B)}

ZERO_WIDTH_SPACE = "​"


def _lookalike_spelling() -> dict[str, str]:
    """Latin letter -> a Cyrillic twin where one exists, else a Greek one."""
    from cosmos_framework.auxiliary.guardrail.blocklist.lookalikes import LOOKALIKES

    twins: dict[str, str] = {}
    for script in ("CYRILLIC SMALL LETTER", "GREEK SMALL LETTER"):
        for char, letter in sorted(LOOKALIKES.items()):
            if letter not in twins and unicodedata.name(char, "").startswith(script):
                twins[letter] = char
    return twins


LOOKALIKE = _lookalike_spelling()

PROMPT_TEMPLATE = "a photo of {body} in the scene"


def spellings(entry: str) -> dict[str, str]:
    """The five prompts probing one entry.

    Splitting at the midpoint is arbitrary but stable: the point is to produce a
    spelling the matcher has to fuse back together, not to find the split a
    person would choose.
    """
    entry = entry.strip()
    middle = max(1, len(entry) // 2)
    return {
        "verbatim": PROMPT_TEMPLATE.format(body=entry),
        "spaced": PROMPT_TEMPLATE.format(body=f"{entry[:middle]} {entry[middle:]}"),
        "invisible": PROMPT_TEMPLATE.format(body=f"{entry[:middle]}{ZERO_WIDTH_SPACE}{entry[middle:]}"),
        "fullwidth": PROMPT_TEMPLATE.format(body="".join(FULLWIDTH.get(c, c) for c in entry)),
        "lookalike": PROMPT_TEMPLATE.format(body="".join(LOOKALIKE.get(c, c) for c in entry.lower())),
    }


def probe_id(entry: str, spelling: str) -> str:
    """A stable, opaque name for one probe: a digest of the entry, and the spelling.

    The digest stands in for the entry text, which must not be printed, and does
    not depend on where the entry sits in the list.
    """
    digest = hashlib.sha256(entry.strip().encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{spelling}"


def build_probes(entries: Iterable[str]) -> dict[str, str]:
    """Every probe prompt for the given entries, keyed by probe_id."""
    probes: dict[str, str] = {}
    for entry in entries:
        for spelling, prompt in spellings(entry).items():
            probes[probe_id(entry, spelling)] = prompt
    return probes


def corpus_id(probes: dict[str, str]) -> str:
    """A digest of the whole probe set, recorded with each run's results."""
    return hashlib.sha256(json.dumps(probes, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def measure(probes: dict[str, str], is_blocked: Callable[[str], bool]) -> dict[str, bool]:
    """Whether each probe prompt is blocked, keyed by probe id."""
    return {key: bool(is_blocked(prompt)) for key, prompt in probes.items()}


def summarize(results: dict[str, bool]) -> dict[str, dict[str, int]]:
    """Blocked / passed counts per spelling, plus a total."""
    summary: dict[str, dict[str, int]] = {}
    for key, blocked in results.items():
        bucket = summary.setdefault(key.split(":", 1)[1], {"blocked": 0, "passed": 0})
        bucket["blocked" if blocked else "passed"] += 1
    summary["total"] = {
        "blocked": sum(1 for v in results.values() if v),
        "passed": sum(1 for v in results.values() if not v),
    }
    return summary


def compare(before: dict, after: dict) -> dict:
    """What the change gave up, and what it started catching.

    Takes two result files as written by a run. `lost` is the one that matters
    for an unblocking change: prompts that were blocked and are not any more.
    The diff is only meaningful over the same probes, so `comparable` is False
    when the two runs used different probe sets or report different probe ids,
    and `lost` must then not be read as a complete answer.
    """
    before_results, after_results = before["results"], after["results"]
    shared = set(before_results) & set(after_results)
    only_in_before = sorted(set(before_results) - set(after_results))
    only_in_after = sorted(set(after_results) - set(before_results))
    return {
        "comparable": before.get("corpus") == after.get("corpus") and not only_in_before and not only_in_after,
        "corpus": [before.get("corpus"), after.get("corpus")],
        "lost": sorted(key for key in shared if before_results[key] and not after_results[key]),
        "gained": sorted(key for key in shared if not before_results[key] and after_results[key]),
        "only_in_before": only_in_before,
        "only_in_after": only_in_after,
    }


def _build_is_blocked(blocklist_dir: str, whitelist_dir: str | None, add_phrase: str | None):
    """A predicate running the deployed entry point, is_safe, over the given lists.

    is_safe and not censor_prompt: is_safe is what callers run, and it does work
    of its own before the matcher -- to_ascii, a lemmatized pass, the exact-match
    list -- so a sweep through censor_prompt can report a spelling as caught that
    production lets through. When the lists sit in a checkpoint's blocklist
    directory, the sibling exact_match list and nltk_data are used as well, so
    the sweep sees what production sees.
    """
    from cosmos_framework.auxiliary.guardrail.blocklist.blocklist import Blocklist
    from cosmos_framework.auxiliary.guardrail.blocklist.utils import read_keyword_list_from_dir

    entries = read_keyword_list_from_dir(blocklist_dir)
    whitelist = read_keyword_list_from_dir(whitelist_dir) if whitelist_dir else []
    if add_phrase:
        whitelist = whitelist + [add_phrase]

    root = Path(blocklist_dir).parent
    exact_match_dir = root / "exact_match"
    exact_match = read_keyword_list_from_dir(str(exact_match_dir)) if exact_match_dir.is_dir() else []

    lemmatizer = None
    nltk_dir = root / "nltk_data"
    if nltk_dir.is_dir():
        import nltk

        nltk.data.path.append(str(nltk_dir))
        lemmatizer = nltk.WordNetLemmatizer()
    else:
        print(f"no nltk_data next to {blocklist_dir}; the lemmatized pass sees words unchanged", file=sys.stderr)

    probe = Blocklist.from_lists(entries, whitelist, exact_match, lemmatizer=lemmatizer)
    setup = {
        "blocklist": _digest(entries),
        "whitelist": _digest(whitelist),
        "exact_match": _digest(exact_match),
        "lemmatizer": "wordnet" if lemmatizer is not None else "none",
    }
    return entries, (lambda text: not probe.is_safe(text)[0]), setup


def _digest(words: list[str]) -> str:
    """A digest of a keyword list, recorded so a result says what it measured."""
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocklist_dir", nargs="?", help="directory of blocklist keyword files")
    parser.add_argument("--whitelist-dir", help="directory of whitelist keyword files")
    parser.add_argument("--add-phrase", help="whitelist this phrase in addition, to model a proposed change")
    parser.add_argument("--write-probes", help="write the probes for blocklist_dir here and stop, to freeze them")
    parser.add_argument("--probes", help="run this frozen probe file instead of probes built from blocklist_dir")
    parser.add_argument("--json", help="write the per-prompt results here")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two result files")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Exit 0 when nothing was lost, 1 when something was, 2 when the runs cannot be compared."""
    args = parse_args(argv)

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())
        after = json.loads(Path(args.compare[1]).read_text())
        for path, run in zip(args.compare, (before, after)):
            if not isinstance(run, dict) or not run.get("results") or "corpus" not in run:
                print(f"incomparable: {path} is not a result file written by a run, or holds no probes")
                return 2
        diff = compare(before, after)
        if not diff["comparable"]:
            print("incomparable: the two runs did not evaluate the same probes")
            print(f"   probe sets: {diff['corpus'][0]} vs {diff['corpus'][1]}")
            print(f"   probes only in before: {len(diff['only_in_before'])}")
            print(f"   probes only in after : {len(diff['only_in_after'])}")
            print("   freeze the probes with --write-probes and run both versions with --probes")
            return 2
        for name in sorted(set(before.get("setup", {})) | set(after.get("setup", {}))):
            was, now = before.get("setup", {}).get(name), after.get("setup", {}).get(name)
            if was != now:
                print(f"setup changed: {name} {was} -> {now}")
        print(f"lost   (blocked before, not after): {len(diff['lost'])}")
        for key in diff["lost"]:
            print(f"   {key}")
        print(f"gained (not blocked before, now)  : {len(diff['gained'])}")
        return 1 if diff["lost"] else 0

    if not args.blocklist_dir:
        print("a blocklist directory is required unless --compare is used", file=sys.stderr)
        return 2

    entries, is_blocked, setup = _build_is_blocked(args.blocklist_dir, args.whitelist_dir, args.add_phrase)
    if args.write_probes:
        probes = build_probes(entries)
        Path(args.write_probes).write_text(json.dumps({"corpus": corpus_id(probes), "probes": probes}, indent=0))
        print(f"probes: {len(probes)} from {len(entries)} entries, set {corpus_id(probes)}")
        print(f"written: {args.write_probes}")
        return 0

    if args.probes:
        frozen = json.loads(Path(args.probes).read_text())
        probes = frozen["probes"]
        if corpus_id(probes) != frozen["corpus"]:
            print(f"{args.probes} was edited after it was written; refusing to run it", file=sys.stderr)
            return 2
    else:
        probes = build_probes(entries)

    results = measure(probes, is_blocked)
    summary = summarize(results)
    print(f"entries: {len(entries)}   prompts: {len(results)}   probe set: {corpus_id(probes)}")
    for spelling in ("verbatim", "spaced", "invisible", "fullwidth", "lookalike", "total"):
        if spelling in summary:
            counts = summary[spelling]
            print(f"  {spelling:10s} blocked={counts['blocked']:5d} passed={counts['passed']:5d}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"corpus": corpus_id(probes), "setup": setup, "summary": summary, "results": results}, indent=0)
        )
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
