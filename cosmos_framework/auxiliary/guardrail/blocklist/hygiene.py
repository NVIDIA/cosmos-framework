# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hygiene check for blocklist entries that ordinary prose can trip.

`better_profanity` catches evasions such as "n ike" by fusing adjacent tokens
and testing the fused string against the blocklist. The side effect is that a
single-word entry which is itself two ordinary English words glued together
also matches ordinary prose: the entry "deskin" blocks "a robot sitting at a
desk in the background", because "desk" + "in" fuses back into the entry.

The check is purely syntactic -- it asks whether an entry has a two-way split
whose halves are both dictionary words -- so it flags entries whose split is
grammatical but not idiomatic ("deskinning" -> "desk" + "inning") alongside the
ones prose really produces. It is therefore meant to be read by a list owner,
not to auto-delete entries. Multi-word entries are never flagged: they cannot
be fused out of prose.

The remedy for a flagged entry is to whitelist the phrase it collides with:
"desk in" on the whitelist exempts that one spelling while every other spelling
of the entry stays blocked. Moving the entry to the exact-match list also works
but gives up more -- it stops the fused and leetspeak spellings from matching at
all. Either way the decision belongs to the list owners.

Two modes:

    gate    check candidate entries before they are added to the list; exits
            non-zero if any candidate is flagged, so it can run in CI.

    audit   report which entries already on a list have the shape.

The audit prints entry text only with --show-entries, so its default output can
be pasted into a ticket without reproducing content-safety terms.
"""

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

# A single letter is a word in WordNet ("a", "I"), which would make every entry
# splittable. Two characters is the shortest half worth reporting.
DEFAULT_MIN_PART_LEN = 2

# Sentence used to ask the live matcher whether a split really is blocked. The
# surrounding words are inert; only the fused pair matters.
PROBE_TEMPLATE = "a {left} {right} nearby"

# Entries safe to name outside NVIDIA when illustrating the problem: trademarks
# and product names, no content-safety terms.
PUBLIC_EXAMPLES = frozenset({"deskin", "chromebook", "gameboy", "batman", "shotgun", "playstation"})


def wordnet_vocabulary(nltk_data_dir: str | None = None) -> Callable[[str], bool]:
    """Return an `is_word` predicate backed by WordNet.

    The blocklist checkpoint ships its own `nltk_data`; pass that directory to
    avoid depending on a system-wide corpus.
    """
    import nltk

    if nltk_data_dir:
        nltk.data.path.append(str(nltk_data_dir))
    from nltk.corpus import wordnet

    wordnet.synsets("test")  # fail here, not part-way through a scan
    cache: dict[str, bool] = {}

    def is_word(word: str) -> bool:
        if word not in cache:
            cache[word] = bool(wordnet.synsets(word))
        return cache[word]

    return is_word


def colliding_splits(
    entry: str,
    is_word: Callable[[str], bool],
    min_part_len: int = DEFAULT_MIN_PART_LEN,
) -> list[tuple[str, str]]:
    """Every two-way split of `entry` whose halves are both dictionary words.

    Ordered longest-left-half first, which puts the split prose actually
    produces ("desk in") ahead of the incidental ones ("de skin").
    """
    entry = entry.strip().lower()
    if " " in entry:
        return []
    splits = [
        (entry[:i], entry[i:])
        for i in range(min_part_len, len(entry) - min_part_len + 1)
        if is_word(entry[:i]) and is_word(entry[i:])
    ]
    return sorted(splits, key=lambda split: -len(split[0]))


def read_keyword_files(directory: str | Path) -> list[str]:
    """Read every keyword file in a directory, one entry per line."""
    entries = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_dir():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def build_reachability_probe(
    blocklist_words: Iterable[str], whitelist_words: Iterable[str] = ()
) -> Callable[[str, str], bool]:
    """Return a predicate telling whether the matcher really blocks a split.

    A split is only a problem if the deployed matcher fires on the spaced form;
    this drops splits that are theoretical for the list being checked.
    """
    from better_profanity.better_profanity import Profanity

    matcher = Profanity()
    matcher.load_censor_words(custom_words=list(blocklist_words), whitelist_words=list(whitelist_words))

    def is_reachable(left: str, right: str) -> bool:
        probe = PROBE_TEMPLATE.format(left=left, right=right)
        return matcher.censor(probe, censor_char="\x00") != probe

    return is_reachable


def check_entry(
    entry: str,
    is_word: Callable[[str], bool],
    min_part_len: int = DEFAULT_MIN_PART_LEN,
    is_reachable: Callable[[str, str], bool] | None = None,
) -> list[tuple[str, str]]:
    """Colliding splits for one entry, optionally confirmed against a matcher."""
    splits = colliding_splits(entry, is_word, min_part_len)
    if is_reachable is not None:
        splits = [split for split in splits if is_reachable(*split)]
    return splits


def audit_entries(
    entries: Iterable[str],
    is_word: Callable[[str], bool],
    min_part_len: int = DEFAULT_MIN_PART_LEN,
    is_reachable: Callable[[str, str], bool] | None = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Flagged entries, deduplicated and lowercased, in list order."""
    findings = []
    for entry in sorted({e.strip().lower() for e in entries}):
        splits = check_entry(entry, is_word, min_part_len, is_reachable)
        if splits:
            findings.append((entry, splits))
    return findings


def _format_splits(splits: list[tuple[str, str]]) -> str:
    return ", ".join(f"{left}|{right}" for left, right in splits)


def _cmd_gate(args, is_word, is_reachable) -> int:
    flagged = 0
    for candidate in args.entry:
        splits = check_entry(candidate, is_word, args.min_part_len, is_reachable)
        if not splits:
            print(f"ok      {candidate}")
            continue
        flagged += 1
        left, right = splits[0]
        print(f"FLAGGED {candidate}  splits into dictionary words: {_format_splits(splits)}")
        print(f"        prose such as {PROBE_TEMPLATE.format(left=left, right=right)!r} would be blocked")
        print(f"        whitelist the phrase '{left} {right}' to exempt that one spelling,")
        print("        or waive with a note if the split is not English anyone writes")
    if flagged and not args.warn_only:
        print(f"\n{flagged} of {len(args.entry)} candidate(s) flagged", file=sys.stderr)
        return 1
    return 0


def _cmd_audit(args, is_word, is_reachable) -> int:
    entries = read_keyword_files(args.blocklist_dir)
    unique = sorted({entry.strip().lower() for entry in entries})
    single = [entry for entry in unique if " " not in entry]
    findings = audit_entries(single, is_word, args.min_part_len, is_reachable)

    print(f"blocklist directory       : {args.blocklist_dir}")
    print(f"entries (lines)           : {len(entries)}")
    print(f"entries (unique, lowered) : {len(unique)}")
    print(f"  multi-word              : {len(unique) - len(single)}")
    print(f"  single-word             : {len(single)}")
    share = f" ({len(findings) / len(single):.0%} of single-word)" if single else ""
    print(f"flagged single-word entries: {len(findings)}{share}")
    if is_reachable is not None:
        print("  confirmed against the matcher: the spaced form really is blocked")

    public = [(entry, splits) for entry, splits in findings if entry in PUBLIC_EXAMPLES]
    if public:
        print("\nexamples that are safe to quote outside NVIDIA:")
        for entry, splits in public:
            left, right = splits[0]
            print(f"  {entry:<14} -> '{left} {right}'")
        print(f"  ... and {len(findings) - len(public)} more; pass --show-entries to list them")

    if args.show_entries:
        print("\nall flagged entries:")
        for entry, splits in findings:
            print(f"  {entry:<24} {_format_splits(splits)}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nltk-data", help="directory holding the WordNet corpus, e.g. the checkpoint's nltk_data")
    parser.add_argument(
        "--min-part-len",
        type=int,
        default=DEFAULT_MIN_PART_LEN,
        help="shortest half that counts as a word (default: %(default)s)",
    )
    parser.add_argument(
        "--confirm-reachable",
        metavar="BLOCKLIST_DIR",
        help="keep only splits the matcher built from this directory really blocks",
    )
    parser.add_argument("--whitelist-dir", help="whitelist directory, used when confirming reachability")

    sub = parser.add_subparsers(dest="mode", required=True)
    gate = sub.add_parser("gate", help="check candidate entries before adding them")
    gate.add_argument("entry", nargs="+")
    gate.add_argument("--warn-only", action="store_true", help="report flagged candidates but exit 0")

    audit = sub.add_parser("audit", help="report collisions in an existing list")
    audit.add_argument("blocklist_dir")
    audit.add_argument("--show-entries", action="store_true", help="print entry text, which may include unsafe terms")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    is_word = wordnet_vocabulary(args.nltk_data)

    source = args.confirm_reachable or (args.blocklist_dir if args.mode == "audit" else None)
    is_reachable = None
    if source:
        whitelist = read_keyword_files(args.whitelist_dir) if args.whitelist_dir else []
        is_reachable = build_reachability_probe(read_keyword_files(source), whitelist)

    return _cmd_gate(args, is_word, is_reachable) if args.mode == "gate" else _cmd_audit(args, is_word, is_reachable)


if __name__ == "__main__":
    sys.exit(main())
