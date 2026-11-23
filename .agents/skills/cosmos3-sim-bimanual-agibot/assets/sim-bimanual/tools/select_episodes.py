#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Select held-out AgiBotWorld-Beta episodes and the tar files needed to fetch them.

Input: a directory of task_info/task_<task_id>.json files from the Hugging Face dataset
agibot-world/AgiBotWorld-Beta. Each file is a JSON list of episodes with episode_id, task_id, task_name,
init_scene_text and label_info (sometimes spelled lable_info) whose action_config is a list of
{start_frame, end_frame, action_text, skill}. Episode length = max(end_frame) + 1 over action_config (0 if missing).

Tar layout on the Hub (sizes come from the repo tree listing, which works without gated access):
  observations/<task_id>/<first>-<last>.tar   videos + depth of episodes first..last of that task (~48 GB each)
  proprio_stats/<first>-<last>.tar            proprio_stats.h5 of episodes first..last (ranges span tasks)
A tar contains an episode iff first <= episode_id <= last. Ranges DO overlap in the real listing
(e.g. observations/327/648642-685046.tar and 684757-685393.tar); an episode inside several ranges is
"ambiguous" and handled by --ambiguous: all (default) = every candidate tar must be downloaded (always correct);
narrowest = heuristic, the episode is assigned to the candidate tar with the narrowest id range (wide catch-all tars are
the least likely home; verify after download); any = optimistic, one downloaded candidate is enough; drop = skip it.
Real listings are very ambiguous (observations/352 has 6 tars whose ranges all overlap), see stats.*_candidates_histogram.

Selection
  eligible = num_frames >= --min-frames and not excluded (--exclude-text, --exclude-episode-ids)
  mapped   = eligible with >= 1 observation tar (within its own task) and >= 1 proprio tar
  plan     = tars to download: greedy over tar bundles - every (observation tar, proprio tar) pair plus, under
             --ambiguous all, the full requirement set of each ambiguous episode (tars that are only useful
             together) - adding the bundle with the most newly available episodes (then fewest bytes) until >= --n
             are available, at most --max-obs-tars observation tars, preferring >= --min-tasks distinct tasks when
             possible. --planner auto (default) also runs an exhaustive search over the (small) tar subsets and
             keeps the better plan (reaches --n, covers --min-tasks, fewer bytes); stats says which one was used.
  pick     = --n episodes among those available in the plan: round-robin over tasks, then over
             init_scene_text values, seeded shuffle inside each (task, scene) group; output sorted by episode_id.

Output JSON: {"episodes": [...], "tars": [...], "total_download_bytes": int, "stats": {...}, "excluded": {...}}
  episodes[]: episode_id, task_id, task_name, init_scene_text, num_frames, first_action_text, source_file,
              obs_tar / proprio_tar (the tar to extract from; always a string for a fetchable episode: exact when
              *_tar_exact is true, otherwise the tar assigned by --ambiguous narrowest or the narrowest-range candidate
              in the plan), obs_tars_in_plan / proprio_tars_in_plan (every candidate the plan downloads),
              *_tar_candidates (every tar whose range contains the id), ambiguous.
  tars[]:     path, size_bytes, kind, task_id, n_selected_episodes (selected episodes the tar may hold).
Exit status: 0 ok; 3 when fewer than --n episodes could be selected (the output is still written); 1 fatal.

Examples
  select_episodes.py --task-info-dir task_info --n 20 --fetch-index --out held_out.json
  select_episodes.py --task-info-dir task_info --n 20 --obs-index held_out.hub_index.json --out held_out.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO_ID = "agibot-world/AgiBotWorld-Beta"
TAR_RANGE_RE = re.compile(r"(\d+)-(\d+)\.tar$")
OBS_PATH_RE = re.compile(r"(?:^|/)observations/(\d+)/")
TASK_FILE_RE = re.compile(r"task_(\d+)\.json$")
EXHAUSTIVE_LIMIT = 300_000  # max (observation subset x proprio subset) evaluations for --planner auto/exhaustive
EXIT_SHORTFALL = 3


# ----------------------------------------------------------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tar:
    path: str
    size: int
    kind: str  # "observations" | "proprio_stats"
    task_id: int | None
    first: int
    last: int

    def contains(self, episode_id: int) -> bool:
        return self.first <= episode_id <= self.last


@dataclass
class Episode:
    episode_id: int
    task_id: int
    task_name: str
    init_scene_text: str
    num_frames: int
    first_action_text: str
    action_texts: list[str]
    source: str
    obs_candidates: list[str] = field(default_factory=list)
    proprio_candidates: list[str] = field(default_factory=list)
    obs_set: frozenset[str] = frozenset()
    pro_set: frozenset[str] = frozenset()

    @property
    def ambiguous(self) -> bool:
        return len(self.obs_candidates) > 1 or len(self.proprio_candidates) > 1


@dataclass
class Plan:
    name: str
    obs: tuple[str, ...]
    proprio: tuple[str, ...]
    available: list[Episode]
    total_bytes: int
    steps: list[dict] = field(default_factory=list)
    diversity_first: bool = False

    def tasks(self) -> set[int]:
        return {e.task_id for e in self.available}

    def summary(self) -> dict:
        return {"obs_tars": list(self.obs), "proprio_tars": list(self.proprio), "total_bytes": self.total_bytes,
                "available_episodes": len(self.available), "tasks_available": sorted(self.tasks()), "steps": self.steps}


# ----------------------------------------------------------------------------------------------------------------------
# task_info parsing
# ----------------------------------------------------------------------------------------------------------------------
def _as_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    return None


def _as_text(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def parse_entry(entry, file_task_id: int | None) -> tuple[Episode | None, str | None, list[str]]:
    """Return (episode, malformed_reason, flags). malformed_reason is set only when the entry is unusable."""
    flags: list[str] = []
    if not isinstance(entry, dict):
        return None, "entry_not_a_dict", flags
    episode_id = _as_int(entry.get("episode_id"))
    if episode_id is None:
        return None, "missing_episode_id", flags
    task_id = _as_int(entry.get("task_id"))
    if task_id is None:
        if file_task_id is None:
            return None, "missing_task_id", flags
        task_id = file_task_id
        flags.append("task_id_from_filename")
    elif file_task_id is not None and task_id != file_task_id:
        flags.append("task_id_mismatch")  # the file placement decides where the observations live on the Hub
        task_id = file_task_id
    label = entry.get("label_info")
    if not isinstance(label, dict):
        label = entry.get("lable_info")
    action_config = label.get("action_config") if isinstance(label, dict) else None
    if not isinstance(action_config, list) or not action_config:
        flags.append("no_action_config")
        action_config = []
    ends: list[int] = []
    texts: list[str] = []
    ordered: list[tuple[int, int, str]] = []
    for i, seg in enumerate(action_config):
        if not isinstance(seg, dict):
            flags.append("bad_action_segment")
            continue
        end = _as_int(seg.get("end_frame"))
        if end is None:
            flags.append("bad_action_segment")
        else:
            ends.append(end)
        text = _as_text(seg.get("action_text"))
        if text:
            texts.append(text)
        start = _as_int(seg.get("start_frame"))
        ordered.append((start if start is not None else 1 << 62, i, text))
    ordered.sort()
    first_text = next((t for _, _, t in ordered if t), "")
    num_frames = (max(ends) + 1) if ends else 0
    return (Episode(episode_id=episode_id, task_id=task_id, task_name=_as_text(entry.get("task_name")),
                    init_scene_text=_as_text(entry.get("init_scene_text")), num_frames=max(0, num_frames),
                    first_action_text=first_text, action_texts=texts, source=""), None, flags)


def load_task_info_dir(task_info_dir: Path) -> tuple[list[Episode], dict]:
    files = sorted(p for p in task_info_dir.glob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"error: no *.json task_info files in {task_info_dir}")
    episodes: list[Episode] = []
    seen: set[int] = set()
    counters: dict = {"task_info_files": 0, "entries_total": 0, "malformed": Counter(), "flags": Counter(),
                      "duplicate_episode_id": 0, "unreadable_files": []}
    for path in files:
        m = TASK_FILE_RE.search(path.name)
        file_task_id = int(m.group(1)) if m else None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError and UnicodeDecodeError
            counters["unreadable_files"].append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        counters["task_info_files"] += 1
        if isinstance(data, dict):  # tolerate {"episodes": [...]} style wrappers
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            counters["unreadable_files"].append(f"{path.name}: top-level JSON is not a list")
            continue
        for entry in data:
            counters["entries_total"] += 1
            ep, reason, flags = parse_entry(entry, file_task_id)
            for f in flags:
                counters["flags"][f] += 1
            if ep is None:
                counters["malformed"][reason] += 1
                continue
            if ep.episode_id in seen:
                counters["duplicate_episode_id"] += 1
                continue
            seen.add(ep.episode_id)
            ep.source = path.name
            episodes.append(ep)
    episodes.sort(key=lambda e: (e.task_id, e.episode_id))
    return episodes, counters


# ----------------------------------------------------------------------------------------------------------------------
# tar index: load from JSON or fetch from the Hub
# ----------------------------------------------------------------------------------------------------------------------
def _tar_from_record(rec, kind_hint: str | None, task_hint: int | None) -> Tar | None:
    if isinstance(rec, str):
        rec = {"path": rec, "size": 0}
    if not isinstance(rec, dict) or not isinstance(rec.get("path"), str):
        return None
    path = rec["path"].strip("/")
    m = TAR_RANGE_RE.search(path)
    if not m:
        return None
    first, last = int(m.group(1)), int(m.group(2))
    if first > last:
        first, last = last, first
    size = _as_int(rec.get("size", rec.get("size_bytes"))) or 0
    obs_m = OBS_PATH_RE.search(path)
    if obs_m:
        return Tar(path, size, "observations", int(obs_m.group(1)), first, last)
    if "proprio_stats/" in path or kind_hint == "proprio_stats":
        return Tar(path, size, "proprio_stats", None, first, last)
    if kind_hint == "observations" and task_hint is not None:
        return Tar(path, size, "observations", task_hint, first, last)
    return None


def empty_index() -> dict:
    return {"observations": defaultdict(list), "proprio_stats": []}


def read_json_file(path: Path, what: str):
    """Read a JSON file; a missing/unreadable file or invalid JSON/UTF-8 is a clean fatal error (no traceback)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError and UnicodeDecodeError
        raise SystemExit(f"error: cannot read {what} {path}: {type(exc).__name__}: {exc}") from None


def merge_index_file(index: dict, path: Path, want_obs: bool, want_proprio: bool) -> dict:
    """Accepts the cache format written by --fetch-index, a {task_id: [records]} mapping or a flat list of records."""
    data = read_json_file(path, "tar index")
    got = {"observations": 0, "proprio_stats": 0}

    def add(rec, kind_hint=None, task_hint=None):
        tar = _tar_from_record(rec, kind_hint, task_hint)
        if tar is None:
            return
        if tar.kind == "observations" and want_obs:
            if tar.path not in {t.path for t in index["observations"][tar.task_id]}:
                index["observations"][tar.task_id].append(tar)
                got["observations"] += 1
        elif tar.kind == "proprio_stats" and want_proprio:
            if tar.path not in {t.path for t in index["proprio_stats"]}:
                index["proprio_stats"].append(tar)
                got["proprio_stats"] += 1

    if isinstance(data, dict) and ("observations" in data or "proprio_stats" in data):
        obs = data.get("observations") or {}
        if isinstance(obs, dict):
            for k, recs in obs.items():
                for rec in recs or []:
                    add(rec, "observations", _as_int(k))
        elif isinstance(obs, list):
            for rec in obs:
                add(rec, "observations")
        for rec in data.get("proprio_stats") or []:
            add(rec, "proprio_stats")
    elif isinstance(data, dict):
        for k, recs in data.items():
            for rec in recs or []:
                add(rec, "observations", _as_int(k))
    elif isinstance(data, list):
        for rec in data:
            add(rec)
    else:
        raise SystemExit(f"error: unrecognised index format in {path}")
    return got


def fetch_index(task_ids: list[int], repo_id: str) -> dict:
    """List observations/<task_id> for every task and proprio_stats once. Never prints credentials."""
    from huggingface_hub import HfApi  # local import: only needed with --fetch-index

    api = HfApi()
    listing = {"repo_id": repo_id, "fetched_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "observations": {}, "proprio_stats": [], "errors": []}

    def list_tars(path_in_repo: str) -> list[dict]:
        try:
            entries = list(api.list_repo_tree(repo_id, path_in_repo=path_in_repo, repo_type="dataset"))
        except Exception as exc:  # 404 for a task without observations, network errors, ...
            first_line = (str(exc).splitlines() or [""])[0][:200]  # str(exc) may be empty, e.g. TimeoutError()
            msg = f"{path_in_repo}: {type(exc).__name__}: {first_line}"
            listing["errors"].append(msg)
            print(f"[fetch-index] WARNING {msg}", file=sys.stderr)
            return []
        recs = [{"path": e.path, "size": int(e.size)} for e in entries
                if getattr(e, "size", None) is not None and TAR_RANGE_RE.search(e.path)]
        recs.sort(key=lambda r: r["path"])
        ranges = [tuple(map(int, TAR_RANGE_RE.search(r["path"]).groups())) for r in recs]
        overlaps = sum(1 for a, b in itertools.combinations(ranges, 2) if a[0] <= b[1] and b[0] <= a[1])
        print(f"[fetch-index] {path_in_repo}: {len(recs)} tar(s), {fmt_gb(sum(r['size'] for r in recs))} GB"
              + (f", {overlaps} overlapping range pair(s)" if overlaps else ""), file=sys.stderr)
        for r in recs:
            print(f"[fetch-index]   {r['path']}  {r['size']:>14,d} bytes", file=sys.stderr)
        return recs

    for tid in sorted(task_ids):
        listing["observations"][str(tid)] = list_tars(f"observations/{tid}")
    listing["proprio_stats"] = list_tars("proprio_stats")
    return listing


def index_from_listing(listing: dict) -> dict:
    index = empty_index()
    for k, recs in (listing.get("observations") or {}).items():
        for rec in recs:
            tar = _tar_from_record(rec, "observations", _as_int(k))
            if tar is not None:
                index["observations"][tar.task_id].append(tar)
    for rec in listing.get("proprio_stats") or []:
        tar = _tar_from_record(rec, "proprio_stats", None)
        if tar is not None:
            index["proprio_stats"].append(tar)
    return index


def map_episodes(episodes: list[Episode], index: dict, tars: dict[str, Tar], policy: str = "all") -> None:
    """Fill *_candidates (every tar whose range contains the id) and the planning sets obs_set / pro_set:
    all|any|drop -> all candidates (any uses one-of semantics when planning); narrowest -> the single candidate with the
    narrowest id range (ties: smaller tar, then path)."""
    def planning_set(paths: list[str]) -> frozenset[str]:
        if policy == "narrowest" and len(paths) > 1:
            return frozenset({min(paths, key=lambda q: (tars[q].last - tars[q].first, tars[q].size, q))})
        return frozenset(paths)

    for e in episodes:
        e.obs_candidates = sorted(t.path for t in index["observations"].get(e.task_id, []) if t.contains(e.episode_id))
        e.proprio_candidates = sorted(t.path for t in index["proprio_stats"] if t.contains(e.episode_id))
        e.obs_set, e.pro_set = planning_set(e.obs_candidates), planning_set(e.proprio_candidates)


# ----------------------------------------------------------------------------------------------------------------------
# eligibility
# ----------------------------------------------------------------------------------------------------------------------
def load_exclude_text(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = read_json_file(path, "--exclude-text file")
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        raise SystemExit(f"error: {path} must be a JSON list of strings")
    terms = sorted({s.strip() for s in data if s.strip()}, key=str.lower)
    return terms


def text_matches(ep: Episode, term: str) -> bool:
    t = term.lower()
    if t == ep.task_name.strip().lower():
        return True
    if t in ep.init_scene_text.lower():
        return True
    return any(t in a.lower() for a in ep.action_texts)


def filter_episodes(episodes: list[Episode], min_frames: int, exclude_terms: list[str], exclude_ids: set[int],
                    ambiguous_policy: str) -> tuple[list[Episode], dict]:
    excluded = {"too_short": 0, "by_episode_id": 0, "by_text_episodes": 0, "by_text": {t: 0 for t in exclude_terms},
                "no_obs_tar": 0, "no_proprio_tar": 0, "ambiguous_dropped": 0}
    excluded_ids: dict[str, list[int]] = defaultdict(list)
    mapped: list[Episode] = []
    for ep in episodes:
        if ep.num_frames < min_frames:
            excluded["too_short"] += 1
            continue
        if ep.episode_id in exclude_ids:
            excluded["by_episode_id"] += 1
            excluded_ids["by_episode_id"].append(ep.episode_id)
            continue
        hits = [t for t in exclude_terms if text_matches(ep, t)]
        if hits:
            excluded["by_text_episodes"] += 1
            for t in hits:
                excluded["by_text"][t] += 1
            excluded_ids["by_text"].append(ep.episode_id)
            continue
        if not ep.obs_candidates:
            excluded["no_obs_tar"] += 1
            excluded_ids["no_obs_tar"].append(ep.episode_id)
            continue
        if not ep.proprio_candidates:
            excluded["no_proprio_tar"] += 1
            excluded_ids["no_proprio_tar"].append(ep.episode_id)
            continue
        if ep.ambiguous and ambiguous_policy == "drop":
            excluded["ambiguous_dropped"] += 1
            excluded_ids["ambiguous_dropped"].append(ep.episode_id)
            continue
        mapped.append(ep)
    excluded["episode_ids"] = {k: v for k, v in excluded_ids.items()}
    return mapped, excluded


# ----------------------------------------------------------------------------------------------------------------------
# planning: which tars to download
# ----------------------------------------------------------------------------------------------------------------------
def _fetchable(e: Episode, obs: frozenset[str] | set[str], pro: frozenset[str] | set[str], policy: str) -> bool:
    """all: every candidate tar of the episode is chosen; any: at least one candidate of each kind is chosen."""
    if policy == "any":
        return bool(e.obs_set & obs) and bool(e.pro_set & pro)
    return e.obs_set <= obs and e.pro_set <= pro


def _available(eps: list[Episode], obs, pro, policy: str) -> list[Episode]:
    return [e for e in eps if _fetchable(e, obs, pro, policy)]


Move = tuple[frozenset[str], frozenset[str]]  # (observation tars, proprio tars) added together by one greedy step


def greedy_moves(eps: list[Episode], policy: str) -> list[Move]:
    """Candidate greedy moves: every (observation tar, proprio tar) pair plus, unless the policy is 'any' (one-of
    semantics, a pair always suffices), the full requirement set of every episode that needs more than one tar of a
    kind (all its candidate observation tars + all its candidate proprio tars). Without these bundles an episode whose
    id falls in two overlapping proprio ranges (e.g. proprio_stats/866961-940165.tar and 902109-928319.tar on the real
    Hub) can never become available through single-pair steps, and the greedy search stalls with an empty plan."""
    obs_cands = sorted({o for e in eps for o in e.obs_set})
    pro_cands = sorted({p for e in eps for p in e.pro_set})
    moves: set[Move] = {(frozenset({o}), frozenset({p})) for o in obs_cands for p in pro_cands}
    if policy != "any":
        moves |= {(e.obs_set, e.pro_set) for e in eps if len(e.obs_set) > 1 or len(e.pro_set) > 1}
    return sorted(moves, key=lambda m: (len(m[0]) + len(m[1]), sorted(m[0]), sorted(m[1])))


def plan_greedy(eps: list[Episode], tars: dict[str, Tar], n: int, max_obs: int, min_tasks: int,
                diversity_first: bool, policy: str = "all") -> Plan:
    """Greedy over tar bundles (see greedy_moves): at every step add the bundle with the most newly available
    episodes, then the fewest new bytes (ties: path order). A bundle is an (observation tar, proprio tar) pair or the
    full requirement set of an ambiguous episode, so tars that are only useful together are added in one step.
    With diversity_first, while fewer than min_tasks tasks are covered only bundles that add an observation tar of an
    uncovered task are considered (even after n episodes are available, as long as --max-obs-tars allows another
    observation tar); the restriction is relaxed as soon as it cannot make progress."""
    moves = greedy_moves(eps, policy)
    chosen_o: set[str] = set()
    chosen_p: set[str] = set()
    steps: list[dict] = []
    avail = _available(eps, chosen_o, chosen_p, policy)
    want_diversity = diversity_first
    while True:
        covered = {tars[o].task_id for o in chosen_o}
        # keep adding tars while fewer than n episodes are available, or (diversity phase) while fewer than
        # min_tasks tasks are covered and another observation tar may still be added
        want_new_task = want_diversity and len(covered) < min_tasks and len(chosen_o) < max_obs
        if len(avail) >= n and not want_new_task:
            break
        best = None
        for move_o, move_p in moves:
            new_o, new_p = move_o - chosen_o, move_p - chosen_p
            if not new_o and not new_p:
                continue
            if len(chosen_o) + len(new_o) > max_obs:
                continue
            if want_new_task and all(tars[o].task_id in covered for o in new_o):  # also skips moves with no new obs tar
                continue
            gain = len(_available(eps, chosen_o | move_o, chosen_p | move_p, policy)) - len(avail)
            cost = sum(tars[t].size for t in new_o | new_p)
            key = (-gain, cost, tuple(sorted(move_o)), tuple(sorted(move_p)))
            if best is None or key < best[0]:
                best = (key, new_o, new_p, gain, cost, len(move_o) + len(move_p) > 2)
        if best is None or best[3] <= 0:
            if want_new_task:  # diversity restriction cannot make progress -> relax it
                want_diversity = False
                continue
            break
        _, new_o, new_p, gain, cost, is_bundle = best
        steps.append({"obs_tars": sorted(new_o), "proprio_tars": sorted(new_p), "bundle": is_bundle,
                      "new_episodes": gain, "new_bytes": cost, "diversity_phase": want_new_task})
        chosen_o |= new_o
        chosen_p |= new_p
        avail = _available(eps, chosen_o, chosen_p, policy)
    total = sum(tars[t].size for t in chosen_o | chosen_p)
    return Plan("greedy-diverse" if diversity_first else "greedy", tuple(sorted(chosen_o)), tuple(sorted(chosen_p)),
                avail, total, steps, diversity_first)


def plan_exhaustive(eps: list[Episode], tars: dict[str, Tar], n: int, max_obs: int, min_tasks: int,
                    policy: str = "all", limit: int = EXHAUSTIVE_LIMIT) -> tuple[Plan | None, str, int]:
    """Enumerate observation-tar subsets (size <= max_obs) x proprio-tar subsets; keep the plan that reaches n
    with (diversity satisfied, fewest bytes, most available episodes). Returns (plan|None, note, evaluations)."""
    obs_cands = sorted({o for e in eps for o in e.obs_set})
    pro_cands = sorted({p for e in eps for p in e.pro_set})
    if not obs_cands or not pro_cands:
        return None, "no mapped episodes", 0
    k_max = min(max_obs, len(obs_cands))
    n_obs_subsets = sum(math.comb(len(obs_cands), k) for k in range(1, k_max + 1))
    estimate = n_obs_subsets * (2 ** len(pro_cands))
    if estimate > limit:
        return None, f"search space too large ({estimate:,} > {limit:,} evaluations); greedy plan used", 0
    best = None
    evaluated = 0
    for k in range(1, k_max + 1):
        for obs in itertools.combinations(obs_cands, k):
            obs_set = frozenset(obs)
            cand = [e for e in eps if (e.obs_set & obs_set if policy == "any" else e.obs_set <= obs_set)]
            if len(cand) < n:
                continue
            pro_rel = sorted({p for e in cand for p in e.pro_set})
            obs_bytes = sum(tars[t].size for t in obs)
            for m in range(1, len(pro_rel) + 1):
                for pro in itertools.combinations(pro_rel, m):
                    evaluated += 1
                    pro_set = frozenset(pro)
                    avail = [e for e in cand if (e.pro_set & pro_set if policy == "any" else e.pro_set <= pro_set)]
                    if len(avail) < n:
                        continue
                    total = obs_bytes + sum(tars[t].size for t in pro)
                    div_ok = len({e.task_id for e in avail}) >= min_tasks
                    key = (not div_ok, total, -len(avail), obs, pro)
                    if best is None or key < best[0]:
                        best = (key, obs, pro, avail, total)
    if best is None:
        return None, f"no tar subset within --max-obs-tars {max_obs} makes {n} episodes available", evaluated
    _, obs, pro, avail, total = best
    return Plan("exhaustive", tuple(obs), tuple(pro), avail, total), "ok", evaluated


def plan_key(plan: Plan, n: int, min_tasks: int) -> tuple:
    """Smaller is better: reaches n, satisfies diversity, fewer bytes, more episodes."""
    return (len(plan.available) < n, len(plan.tasks()) < min_tasks, plan.total_bytes, -len(plan.available))


def why_better(winner: Plan, loser: Plan, n: int, min_tasks: int) -> str:
    """Explain in plan_key order why winner ranks before loser (a byte comparison is only claimed when it applies)."""
    w_gb, l_gb = f"{fmt_gb(winner.total_bytes)} GB", f"{fmt_gb(loser.total_bytes)} GB"
    if len(loser.available) < n <= len(winner.available):
        return (f"{winner.name} plan reaches --n {n} ({len(winner.available)} available, {w_gb}) where the {loser.name} "
                f"plan did not ({len(loser.available)} available, {l_gb})")
    if len(loser.tasks()) < min_tasks <= len(winner.tasks()):
        return (f"{winner.name} plan covers >= --min-tasks {min_tasks} distinct tasks {sorted(winner.tasks())} ({w_gb}) "
                f"where the {loser.name} plan did not ({sorted(loser.tasks())}, {l_gb})")
    if winner.total_bytes < loser.total_bytes:
        return (f"{winner.name} plan is cheaper than the {loser.name} one: {w_gb} vs {l_gb} "
                f"(saving {fmt_gb(loser.total_bytes - winner.total_bytes)} GB)")
    return (f"{winner.name} plan makes more episodes available than the {loser.name} one at the same cost ({w_gb}): "
            f"{len(winner.available)} vs {len(loser.available)}")


# ----------------------------------------------------------------------------------------------------------------------
# picking episodes inside the plan
# ----------------------------------------------------------------------------------------------------------------------
def pick_episodes(available: list[Episode], n: int, seed: int) -> list[Episode]:
    """Round-robin over tasks, inside a task round-robin over init_scene_text values; seeded shuffle inside each
    (task, scene) group so the choice is deterministic for a given seed. Returned sorted by (task_id, episode_id)."""
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[Episode]] = defaultdict(list)
    for e in sorted(available, key=lambda e: (e.task_id, e.init_scene_text, e.episode_id)):
        groups[(e.task_id, e.init_scene_text)].append(e)
    for g in groups.values():
        rng.shuffle(g)
    tasks = sorted({t for t, _ in groups})
    rng.shuffle(tasks)
    scenes_by_task: dict[int, list[tuple[int, str]]] = {}
    for t in tasks:
        keys = sorted(k for k in groups if k[0] == t)
        rng.shuffle(keys)
        scenes_by_task[t] = keys
    picked: list[Episode] = []
    while len(picked) < n and tasks:
        for t in list(tasks):
            if len(picked) >= n:
                break
            keys = scenes_by_task[t]
            while keys and not groups[keys[0]]:
                keys.pop(0)
            if not keys:
                tasks.remove(t)
                continue
            key = keys.pop(0)
            picked.append(groups[key].pop())
            keys.append(key)  # rotate to the back so the next pick in this task comes from another scene
    picked.sort(key=lambda e: (e.task_id, e.episode_id))
    return picked


# ----------------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------------
def fmt_gb(b: int | float) -> str:
    return f"{b / 1e9:,.1f}"


def _trunc(s: str, width: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= width else s[: max(1, width - 3)] + "..."


def print_table(headers: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt_row = lambda r: "  ".join((c.rjust(w) if i in right else c.ljust(w)) for i, (c, w) in enumerate(zip(r, widths)))
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def narrowest_tar(paths: list[str], tars: dict[str, Tar]) -> str | None:
    """The candidate with the narrowest id range (ties: smaller tar, then path); None for an empty list."""
    return min(paths, key=lambda q: (tars[q].last - tars[q].first, tars[q].size, q)) if paths else None


def episode_record(e: Episode, plan_paths: set[str], tars: dict[str, Tar], policy: str) -> dict:
    """obs_tar / proprio_tar = the tar to extract the episode from; always a string when a candidate is in the plan.
    It is exact (obs_tar_exact / proprio_tar_exact true) when the episode id falls in exactly one tar range. For an
    ambiguous episode it is the tar assigned by --ambiguous narrowest, or under all/any the narrowest-range candidate
    that is in the plan (heuristic: the episode is in exactly one of *_tars_in_plan, verify after download)."""
    def resolve(candidates: list[str], planning_set: frozenset[str]) -> tuple[str | None, list[str]]:
        in_plan = [c for c in candidates if c in plan_paths]
        assigned = [c for c in planning_set if c in plan_paths] if policy == "narrowest" else in_plan
        return (assigned[0] if len(assigned) == 1 else narrowest_tar(in_plan, tars)), in_plan

    obs_tar, obs_in_plan = resolve(e.obs_candidates, e.obs_set)
    pro_tar, pro_in_plan = resolve(e.proprio_candidates, e.pro_set)
    return {"episode_id": e.episode_id, "task_id": e.task_id, "task_name": e.task_name,
            "init_scene_text": e.init_scene_text, "num_frames": e.num_frames,
            "obs_tar": obs_tar, "proprio_tar": pro_tar,
            "obs_tar_exact": len(e.obs_candidates) == 1, "proprio_tar_exact": len(e.proprio_candidates) == 1,
            "obs_tars_in_plan": obs_in_plan, "proprio_tars_in_plan": pro_in_plan,
            "obs_tar_candidates": list(e.obs_candidates), "proprio_tar_candidates": list(e.proprio_candidates),
            "ambiguous": e.ambiguous, "first_action_text": e.first_action_text, "source_file": e.source}


# ----------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-info-dir", required=True, type=Path, help="directory of task_<task_id>.json files")
    ap.add_argument("--n", required=True, type=int, help="number of episodes to select")
    ap.add_argument("--min-frames", type=int, default=901, help="minimum episode length in frames (default 901)")
    ap.add_argument("--exclude-text", type=Path, metavar="FILE.json",
                    help="JSON list of strings; an episode is excluded if any string is a case-insensitive substring "
                         "of its init_scene_text or of any action_text, or equals its task_name")
    ap.add_argument("--exclude-episode-ids", type=int, nargs="*", default=[], metavar="ID")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--obs-index", type=Path, metavar="FILE.json",
                     help="observation tar listing: the cache written by --fetch-index, a {task_id: [{path,size}]} "
                          "mapping or a flat list of {path,size}. If it also has a proprio_stats section it is used "
                          "unless --proprio-index is given")
    src.add_argument("--fetch-index", action="store_true",
                     help="list observations/<task_id> for every task in --task-info-dir plus proprio_stats once "
                          "from the Hub and cache the listing as JSON next to --out")
    ap.add_argument("--proprio-index", type=Path, metavar="FILE.json", help="proprio tar listing (same formats)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--index-cache", type=Path, metavar="FILE.json",
                    help="where --fetch-index writes the listing (default <out stem>.hub_index.json next to --out)")
    ap.add_argument("--max-obs-tars", type=int, default=2, help="maximum number of observation tars (default 2)")
    ap.add_argument("--min-tasks", type=int, default=2, help="preferred minimum of distinct tasks (default 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", choices=["auto", "greedy", "exhaustive"], default="auto",
                    help="auto: greedy, then keep an exhaustive plan when it is better (reaches --n, covers --min-tasks, "
                         "fewer bytes) (default); greedy: only the greedy bundle selection; exhaustive: exhaustive "
                         "search (falls back to greedy if too large)")
    ap.add_argument("--ambiguous", choices=["all", "narrowest", "any", "drop"], default="all",
                    help="episode whose id falls in several tar ranges: all = every candidate tar must be downloaded "
                         "(default, always correct); narrowest = assign it to the candidate with the narrowest id range "
                         "(heuristic, verify after download); any = one downloaded candidate is enough (optimistic); "
                         "drop = skip such episodes")
    ap.add_argument("--out", required=True, type=Path, metavar="FILE.json")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n < 1 or args.max_obs_tars < 1 or args.min_tasks < 1 or args.min_frames < 0:
        raise SystemExit("error: --n, --max-obs-tars and --min-tasks must be >= 1 and --min-frames >= 0")
    if not args.task_info_dir.is_dir():
        raise SystemExit(f"error: --task-info-dir {args.task_info_dir} is not a directory")
    if not args.fetch_index and args.obs_index is None:
        raise SystemExit("error: give --obs-index FILE.json (and optionally --proprio-index) or --fetch-index")
    for opt, path in (("--obs-index", args.obs_index), ("--proprio-index", args.proprio_index),
                      ("--exclude-text", args.exclude_text)):
        if path is not None and not path.is_file():
            raise SystemExit(f"error: {opt} {path} is not a readable file")
    notes: list[str] = []

    # 1. task_info ---------------------------------------------------------------------------------------------------
    episodes, counters = load_task_info_dir(args.task_info_dir)
    tasks_in_input = sorted({e.task_id for e in episodes})
    for msg in counters["unreadable_files"]:
        print(f"WARNING unreadable task_info file: {msg}", file=sys.stderr)

    # 2. tar index -----------------------------------------------------------------------------------------------------
    if args.fetch_index:
        listing = fetch_index(tasks_in_input, args.repo_id)
        index_cache = args.index_cache or args.out.with_name(args.out.stem + ".hub_index.json")
        try:
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            index_cache.write_text(json.dumps(listing, indent=1))
        except OSError as exc:
            raise SystemExit(f"error: cannot write index cache {index_cache}: {exc}") from None
        print(f"[fetch-index] cached listing -> {index_cache}  (reuse with --obs-index {index_cache})", file=sys.stderr)
        index = index_from_listing(listing)
        index_source = {"fetched_from": args.repo_id, "cache": str(index_cache), "errors": listing["errors"]}
    else:
        index = empty_index()
        got_obs = merge_index_file(index, args.obs_index, want_obs=True, want_proprio=args.proprio_index is None)
        if args.proprio_index is not None:
            got_pro = merge_index_file(index, args.proprio_index, want_obs=False, want_proprio=True)
            got_obs["proprio_stats"] += got_pro["proprio_stats"]
        index_source = {"obs_index": str(args.obs_index),
                        "proprio_index": str(args.proprio_index or args.obs_index), "tars_loaded": got_obs}
    tars: dict[str, Tar] = {t.path: t for ts in index["observations"].values() for t in ts}
    tars.update({t.path: t for t in index["proprio_stats"]})
    if not index["proprio_stats"]:
        notes.append("index has no proprio_stats tars: every episode is dropped as no_proprio_tar")
    for tid in tasks_in_input:
        if not index["observations"].get(tid):
            notes.append(f"task {tid}: no observation tars in the index -> all its episodes dropped as no_obs_tar")
    map_episodes(episodes, index, tars, args.ambiguous)

    # 3. eligibility ---------------------------------------------------------------------------------------------------
    exclude_terms = load_exclude_text(args.exclude_text)
    exclude_ids = set(args.exclude_episode_ids)
    mapped, excluded = filter_episodes(episodes, args.min_frames, exclude_terms, exclude_ids, args.ambiguous)
    n_eligible = len(episodes) - excluded["too_short"] - excluded["by_episode_id"] - excluded["by_text_episodes"]
    eligible_eps = [e for e in episodes if e.num_frames >= args.min_frames and e.episode_id not in exclude_ids
                    and not any(text_matches(e, t) for t in exclude_terms)]
    obs_hist = dict(sorted(Counter(len(e.obs_candidates) for e in eligible_eps).items()))
    pro_hist = dict(sorted(Counter(len(e.proprio_candidates) for e in eligible_eps).items()))
    n_ambiguous = sum(1 for e in eligible_eps if e.ambiguous)
    if n_ambiguous:
        notes.append(f"{n_ambiguous} of {len(eligible_eps)} eligible episodes fall in several tar ranges "
                     f"(observation tars per episode {obs_hist}, proprio tars per episode {pro_hist}); "
                     f"--ambiguous {args.ambiguous}: " + {"all": "every candidate tar must be downloaded",
                                                          "narrowest": "each was assigned to the candidate tar with the "
                                                                       "narrowest id range (heuristic, verify after download)",
                                                          "any": "one downloaded candidate is assumed to be enough (optimistic)",
                                                          "drop": "such episodes were dropped"}[args.ambiguous])

    # 4. plan --------------------------------------------------------------------------------------------------------
    plans: dict[str, Plan] = {}
    if mapped:
        g_plain = plan_greedy(mapped, tars, args.n, args.max_obs_tars, args.min_tasks, diversity_first=False,
                              policy=args.ambiguous)
        plans["greedy"] = g_plain
        greedy = g_plain
        if args.min_tasks > 1:
            g_div = plan_greedy(mapped, tars, args.n, args.max_obs_tars, args.min_tasks, diversity_first=True,
                                policy=args.ambiguous)
            plans["greedy-diverse"] = g_div
            if plan_key(g_div, args.n, args.min_tasks) < plan_key(g_plain, args.n, args.min_tasks):
                greedy = g_div
        chosen = greedy
        exhaustive_note = "not run (--planner greedy)"
        evaluations = 0
        if args.planner in ("auto", "exhaustive"):
            ex, exhaustive_note, evaluations = plan_exhaustive(mapped, tars, args.n, args.max_obs_tars, args.min_tasks,
                                                               policy=args.ambiguous)
            if ex is not None:
                plans["exhaustive"] = ex
                if plan_key(ex, args.n, args.min_tasks) < plan_key(greedy, args.n, args.min_tasks):
                    chosen = ex
                    notes.append(why_better(ex, greedy, args.n, args.min_tasks))
                elif args.planner == "exhaustive":
                    chosen = ex
            elif args.planner == "exhaustive":
                notes.append(f"exhaustive planner unavailable: {exhaustive_note}")
        if not chosen.available:
            notes.append(f"NO PLAN: no tar bundle within --max-obs-tars {args.max_obs_tars} makes any of the {len(mapped)} "
                         f"mapped episodes available" + (
                             "; under --ambiguous all an ambiguous episode needs every candidate tar of both kinds, "
                             "so consider a larger --max-obs-tars, --ambiguous narrowest or --ambiguous any"
                             if n_ambiguous and args.ambiguous == "all" else ""))
    else:
        chosen = Plan("none", (), (), [], 0)
        exhaustive_note, evaluations = "no mapped episodes", 0
        notes.append("no eligible episode could be mapped to both an observation tar and a proprio tar")

    # 5. pick --------------------------------------------------------------------------------------------------------
    selected = pick_episodes(chosen.available, args.n, args.seed)
    tasks_selected = sorted({e.task_id for e in selected})
    min_tasks_ok = len(tasks_selected) >= args.min_tasks
    if not min_tasks_ok:
        if len(tasks_in_input) < args.min_tasks:
            why = f"only {len(tasks_in_input)} task(s) present in {args.task_info_dir}"
        elif len({e.task_id for e in mapped}) < args.min_tasks:
            why = f"only {len({e.task_id for e in mapped})} task(s) have eligible, mapped episodes"
        elif len(selected) < args.n:
            why = "no tar plan reaches --n at all"
        else:
            why = (f"no plan with <= {args.max_obs_tars} observation tar(s) reaches --n {args.n} while covering "
                   f">= {args.min_tasks} tasks; the cheapest single-task plan was used")
        notes.append(f"MIN-TASKS NOT SATISFIED: selected episodes cover {len(tasks_selected)} distinct task(s) "
                     f"< --min-tasks {args.min_tasks} ({why})")
    if len(selected) < args.n:
        notes.append(f"SHORTFALL: only {len(selected)} of --n {args.n} episodes could be selected "
                     f"({len(chosen.available)} available in the chosen tars, {len(mapped)} mapped, "
                     f"{n_eligible} eligible)")

    chosen_paths = sorted(set(chosen.obs) | set(chosen.proprio), key=lambda p: (tars[p].kind, p))
    tar_records = [{"path": p, "size_bytes": tars[p].size, "kind": tars[p].kind, "task_id": tars[p].task_id,
                    "n_selected_episodes": sum(1 for e in selected if p in e.obs_set or p in e.pro_set)}
                   for p in chosen_paths]
    total_bytes = sum(r["size_bytes"] for r in tar_records)

    stats = {
        "requested": args.n, "selected": len(selected), "shortfall": max(0, args.n - len(selected)),
        "task_info_files": counters["task_info_files"], "entries_total": counters["entries_total"],
        "episodes_parsed": len(episodes), "eligible": n_eligible, "mapped": len(mapped),
        "available_in_plan": len(chosen.available),
        "tasks_in_input": tasks_in_input, "tasks_selected": tasks_selected,
        "min_tasks": args.min_tasks, "min_tasks_satisfied": min_tasks_ok,
        "scenes_selected": len({(e.task_id, e.init_scene_text) for e in selected}),
        "ambiguous_selected": sum(1 for e in selected if e.ambiguous), "ambiguous_eligible": n_ambiguous,
        "obs_candidates_histogram": obs_hist, "proprio_candidates_histogram": pro_hist,
        "num_frames_selected": {"min": min((e.num_frames for e in selected), default=0),
                                "max": max((e.num_frames for e in selected), default=0)},
        "planner_requested": args.planner, "planner_used": chosen.name,
        "plans": {k: v.summary() for k, v in plans.items()},
        "exhaustive_note": exhaustive_note, "exhaustive_evaluations": evaluations,
        "obs_tars_chosen": len(chosen.obs), "proprio_tars_chosen": len(chosen.proprio),
        "max_obs_tars": args.max_obs_tars, "min_frames": args.min_frames, "seed": args.seed,
        "ambiguous_policy": args.ambiguous, "index_source": index_source,
        "notes": notes,
    }
    excluded_out = {"malformed_entries": dict(counters["malformed"]), "malformed_total": sum(counters["malformed"].values()),
                    "entry_flags": dict(counters["flags"]), "duplicate_episode_id": counters["duplicate_episode_id"],
                    "unreadable_files": counters["unreadable_files"], **excluded,
                    "exclude_terms": exclude_terms, "exclude_episode_ids": sorted(exclude_ids)}
    plan_paths = {r["path"] for r in tar_records}
    out = {"episodes": [episode_record(e, plan_paths, tars, args.ambiguous) for e in selected], "tars": tar_records,
           "total_download_bytes": total_bytes,
           "stats": stats, "excluded": excluded_out}
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    except OSError as exc:
        raise SystemExit(f"error: cannot write --out {args.out}: {exc}") from None

    # 6. report --------------------------------------------------------------------------------------------------------
    print(f"select_episodes: {counters['task_info_files']} task_info file(s), {counters['entries_total']} entries -> "
          f"{len(episodes)} episodes (malformed {excluded_out['malformed_total']}, duplicates {counters['duplicate_episode_id']}), "
          f"tasks {tasks_in_input}")
    print(f"eligible {n_eligible} (too_short {excluded['too_short']}, by_episode_id {excluded['by_episode_id']}, "
          f"by_text {excluded['by_text_episodes']}) -> mapped {len(mapped)} (no_obs_tar {excluded['no_obs_tar']}, "
          f"no_proprio_tar {excluded['no_proprio_tar']}, ambiguous_dropped {excluded['ambiguous_dropped']})")
    print(f"plan: {chosen.name} | {len(chosen.obs)} observation tar(s) + {len(chosen.proprio)} proprio tar(s) = "
          f"{fmt_gb(total_bytes)} GB | available {len(chosen.available)} -> selected {len(selected)} of {args.n} | "
          f"tasks {tasks_selected} | min_tasks {'satisfied' if min_tasks_ok else 'NOT SATISFIED'}")
    for name, p in plans.items():
        print(f"  candidate {name:<15} {fmt_gb(p.total_bytes):>8} GB  obs {len(p.obs)}  proprio {len(p.proprio)}  "
              f"available {len(p.available):>3}  tasks {sorted(p.tasks())}")
    print()
    def in_plan(cands: list[str]) -> str:  # tars of the plan that may hold the episode; * = id in several ranges
        return "|".join(Path(c).name for c in cands if c in plan_paths)

    rows = [[str(e.episode_id), str(e.task_id), _trunc(e.task_name, 30), str(e.num_frames), _trunc(e.init_scene_text, 48),
             in_plan(e.obs_candidates) + ("*" if len(e.obs_candidates) > 1 else ""),
             in_plan(e.proprio_candidates) + ("*" if len(e.proprio_candidates) > 1 else ""),
             _trunc(e.first_action_text, 40)] for e in selected]
    print_table(["episode_id", "task", "task_name", "frames", "init_scene_text", "obs_tar", "proprio_tar", "first_action_text"],
                rows, right={0, 1, 3})
    if any(e.ambiguous for e in selected):
        print(f"* = episode id falls in several tar ranges (--ambiguous {args.ambiguous}); the table lists every "
              "candidate in the plan, the JSON obs_tar/proprio_tar name the assigned or narrowest one (*_tar_exact false)")
    print()
    print_table(["kind", "task", "size_GB", "episodes", "path"],
                [[r["kind"], str(r["task_id"] if r["task_id"] is not None else "-"), fmt_gb(r["size_bytes"]),
                  str(r["n_selected_episodes"]), r["path"]] for r in tar_records], right={2, 3})
    print(f"TOTAL download: {fmt_gb(total_bytes)} GB ({total_bytes:,d} bytes) in {len(tar_records)} tar(s)")
    for note in notes:
        print(f"NOTE: {note}")
    print(f"wrote {args.out}")
    return EXIT_SHORTFALL if len(selected) < args.n else 0


if __name__ == "__main__":
    sys.exit(main())
