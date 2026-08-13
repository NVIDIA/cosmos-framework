# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Measure what a blocklist change stops catching.

A corpus of ordinary model output answers one question: does a change block
things it did not block before. Any change that *unblocks* -- a whitelist entry,
a data move, a new exemption mechanism -- carries the opposite risk, and a
corpus cannot show it: the phrasings that would newly slip through are exactly
the ones ordinary output does not contain.

This builds that missing side. For every entry it writes four prompts, one per
way the entry can be spelled while still reading as itself:

    verbatim    the entry as written
    spaced      split in the middle, the fusing evasion ("de skin")
    invisible   split with a zero-width space
    fullwidth   spelled in fullwidth letters

Run it before and after a change and compare: any prompt that was blocked and
now is not is a spelling the change gave up. Entry text is never printed --
results are keyed by index and spelling -- so output can be pasted into a
review.

    coverage.py <blocklist-dir> --whitelist-dir <dir> --json before.json
    coverage.py <blocklist-dir> --whitelist-dir <dir> --add-phrase "desk in" --json after.json
    coverage.py --compare before.json after.json
"""

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

# Lowercase ASCII letter -> its fullwidth presentation form.
FULLWIDTH = {chr(code): chr(code - 0x61 + 0xFF41) for code in range(0x61, 0x7B)}

ZERO_WIDTH_SPACE = "​"

PROMPT_TEMPLATE = "a photo of {body} in the scene"


def spellings(entry: str) -> dict[str, str]:
    """The four prompts probing one entry.

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
    }


def measure(entries: Iterable[str], is_blocked: Callable[[str], bool]) -> dict[str, bool]:
    """Whether each probe prompt is blocked, keyed "<index>:<spelling>"."""
    results: dict[str, bool] = {}
    for index, entry in enumerate(entries):
        for spelling, prompt in spellings(entry).items():
            results[f"{index}:{spelling}"] = bool(is_blocked(prompt))
    return results


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


def compare(before: dict[str, bool], after: dict[str, bool]) -> dict[str, list[str]]:
    """What the change gave up, and what it started catching.

    `lost` is the one that matters for an unblocking change: prompts that were
    blocked and are not any more.
    """
    shared = set(before) & set(after)
    return {
        "lost": sorted(key for key in shared if before[key] and not after[key]),
        "gained": sorted(key for key in shared if not before[key] and after[key]),
        "only_in_before": sorted(set(before) - set(after)),
        "only_in_after": sorted(set(after) - set(before)),
    }


def _build_is_blocked(blocklist_dir: str, whitelist_dir: str | None, add_phrase: str | None):
    """A predicate running the deployed censor path over the given lists."""
    from better_profanity.better_profanity import Profanity

    from cosmos_framework.auxiliary.guardrail.blocklist.blocklist import Blocklist
    from cosmos_framework.auxiliary.guardrail.blocklist.utils import read_keyword_list_from_dir

    entries = read_keyword_list_from_dir(blocklist_dir)
    whitelist = read_keyword_list_from_dir(whitelist_dir) if whitelist_dir else []
    if add_phrase:
        whitelist = whitelist + [add_phrase]

    probe = Blocklist.__new__(Blocklist)
    probe.profanity = Profanity()
    probe.profanity.load_censor_words(
        custom_words=entries,
        whitelist_words=[w for w in whitelist if " " not in w.strip()],
    )
    probe.whitelist_words = whitelist
    probe.whitelist_phrases = sorted((w for w in whitelist if " " in w.strip()), key=len, reverse=True)
    return entries, (lambda text: probe.censor_prompt(text)[0])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocklist_dir", nargs="?", help="directory of blocklist keyword files")
    parser.add_argument("--whitelist-dir", help="directory of whitelist keyword files")
    parser.add_argument("--add-phrase", help="whitelist this phrase in addition, to model a proposed change")
    parser.add_argument("--json", help="write the per-prompt results here")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two result files")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())["results"]
        after = json.loads(Path(args.compare[1]).read_text())["results"]
        diff = compare(before, after)
        print(f"lost   (blocked before, not after): {len(diff['lost'])}")
        for key in diff["lost"]:
            print(f"   {key}")
        print(f"gained (not blocked before, now)  : {len(diff['gained'])}")
        return 1 if diff["lost"] else 0

    if not args.blocklist_dir:
        print("a blocklist directory is required unless --compare is used", file=sys.stderr)
        return 2

    entries, is_blocked = _build_is_blocked(args.blocklist_dir, args.whitelist_dir, args.add_phrase)
    results = measure(entries, is_blocked)
    summary = summarize(results)
    print(f"entries: {len(entries)}   prompts: {len(results)}")
    for spelling in ("verbatim", "spaced", "invisible", "fullwidth", "total"):
        if spelling in summary:
            counts = summary[spelling]
            print(f"  {spelling:10s} blocked={counts['blocked']:5d} passed={counts['passed']:5d}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "results": results}, indent=0))
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
