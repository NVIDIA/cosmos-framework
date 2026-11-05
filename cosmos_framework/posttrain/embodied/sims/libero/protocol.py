# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit LIBERO benchmark membership, independent of simulator dependencies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiberoProtocol:
    """Environment-owned execution, reset and reward facts shared across role boundaries."""

    execution_horizon: int
    max_episode_steps: int
    reset_steps: int
    reset_gripper: float
    stop_on_success: bool
    reward: str
    reset_action: str
    seed_derivation: str
    action_transform: str = "binary-inverted-gripper"
    fixed_episode: bool = False
    parallel_lanes: bool = False
    last_control_render: bool = False

    def identity(self) -> dict:
        value = asdict(self)
        if not self.fixed_episode:
            # Existing GR00T checkpoints bind this exact protocol representation.
            value.pop("action_transform")
            value.pop("fixed_episode")
            value.pop("parallel_lanes")
            value.pop("last_control_render")
        return value


LIBERO_PROTOCOLS = {
    "libero_spatial": LiberoProtocol(16, 256, 15, -1.0, False, "relative", "open-gripper-zero", "legacy"),
    "libero_10": LiberoProtocol(
        8, 720, 5, 0.0, True, "terminal-success0-timeout-1-v1", "simulator-zero-seven", "libero-sha256-coordinates-v1"
    ),
}
LIBERO_SUITES = tuple(LIBERO_PROTOCOLS)
DEFAULT_EXECUTION_PROFILE = "default"
PI05_SPATIAL_EXECUTION_PROFILE = "rlinf_pi05_spatial_v2"
HASH_POLICY_REPLICA_ENV_SEEDS = "libero-sha256-policy-rlinf-replica-env-v1"
_PI05_SPATIAL_PROTOCOL = LiberoProtocol(
    5,
    240,
    15,
    -1.0,
    False,
    "relative",
    "open-gripper-zero",
    HASH_POLICY_REPLICA_ENV_SEEDS,
    "identity",
    True,
    True,
    True,
)


def execution_protocol(suite: str, profile: str = DEFAULT_EXECUTION_PROFILE) -> LiberoProtocol:
    if suite not in LIBERO_SUITES:
        raise ValueError(f"unsupported LIBERO suite {suite!r}")
    if profile == DEFAULT_EXECUTION_PROFILE:
        return LIBERO_PROTOCOLS[suite]
    if profile == PI05_SPATIAL_EXECUTION_PROFILE and suite == "libero_spatial":
        return _PI05_SPATIAL_PROTOCOL
    raise ValueError(f"unsupported LIBERO execution_profile {profile!r} for suite {suite!r}")


ALIGNED_EVAL_PROFILE = "aligned_200"
LEGACY_EVAL_PROFILE = "legacy_100"
SHARED_PROFILE = "shared_500"
SINGLE_TASK_PROFILE = "single_task_50"


@dataclass(frozen=True)
class LiberoRuntime:
    """The single simulator dependency set that recipes, probes, identities and test tools bind.

    ``libero`` identifies the RLinf distribution; its source digest binds the frozen package files.
    Suite protocols (chunks, limits, reset, seeds, rewards) stay in ``LIBERO_PROTOCOLS``.
    """

    mujoco: str
    robosuite: str
    numpy: str
    libero: str
    isaac_gr00t: str = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"

    def identity(self) -> dict[str, str]:
        return {
            **asdict(self),
            "requirements_sha256": hashlib.sha256(_requirements_path().read_bytes()).hexdigest(),
            "libero_source_sha256": hashlib.sha256(
                Path(__file__).with_name("rlinf-source.json").read_bytes()
            ).hexdigest(),
        }


def _requirements_path() -> Path:
    return Path(__file__).with_name("requirements.lock")


def runtime_packages() -> dict[str, str]:
    """Read the exact package pins exported from the historical RLinf simulator dependency closure."""
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", _requirements_path().read_text(), re.MULTILINE))


#: Qualified RLinf runtime shared by both production suites.
LIBERO_RUNTIME = LiberoRuntime("3.3.7", "1.4.1", "1.26.4", "rlinf-libero==0.1.3")


def evaluation_state_ids(profile: str) -> range:
    if profile == LEGACY_EVAL_PROFILE:
        return range(10)
    if profile == ALIGNED_EVAL_PROFILE:
        return range(20)
    if profile in (SHARED_PROFILE, SINGLE_TASK_PROFILE):
        return range(50)
    raise ValueError(f"unsupported LIBERO evaluation profile {profile!r}")


def _manifest_state_ids(split: str, suite: str, profile: str) -> range:
    evaluation_states = evaluation_state_ids(profile)
    if split == "train":
        return range(0 if suite == "libero_10" or profile == SHARED_PROFILE else 10, 50)
    return evaluation_states


def _manifest_task_ids(suite: str, profile: str, task_id: int | None) -> range:
    if profile == SINGLE_TASK_PROFILE:
        if suite != "libero_10":
            raise ValueError("single_task_50 requires the libero_10 suite")
        if task_id is None or type(task_id) is not int or not 0 <= task_id < 10:
            raise ValueError("single_task_50 manifest task_id must be an integer in [0,10)")
        return range(task_id, task_id + 1)
    if task_id is not None:
        raise ValueError("manifest task_id requires single_task_50")
    return range(10)


@dataclass(frozen=True, order=True)
class LiberoCase:
    task_id: int
    init_state_id: int
    suite: str = "libero_spatial"

    def __post_init__(self) -> None:
        if self.suite not in LIBERO_SUITES:
            raise ValueError(f"unsupported LIBERO suite {self.suite!r}")
        for name, upper in (("task_id", 10), ("init_state_id", 50)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < upper:
                raise ValueError(f"LIBERO {name} must be an integer in [0,{upper}), got {value!r}")

    @property
    def id(self) -> str:
        return f"{self.suite}/{self.task_id}/{self.init_state_id}"


@dataclass(frozen=True)
class LiberoManifest:
    split: str
    cases: tuple[LiberoCase, ...]
    suite: str = "libero_spatial"
    profile: str = LEGACY_EVAL_PROFILE
    task_id: int | None = None

    def __post_init__(self) -> None:
        if self.suite not in LIBERO_SUITES:
            raise ValueError(f"unsupported LIBERO suite {self.suite!r}")
        if self.split not in ("train", "eval"):
            raise ValueError("LIBERO split must be train or eval")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError(f"LIBERO {self.split} manifest contains duplicate cases")
        if self.profile == ALIGNED_EVAL_PROFILE and (self.suite != "libero_10" or self.split != "eval"):
            raise ValueError("aligned_200 is a LIBERO-10 evaluation profile")
        tasks = _manifest_task_ids(self.suite, self.profile, self.task_id)
        states = _manifest_state_ids(self.split, self.suite, self.profile)
        expected = {LiberoCase(task, state, self.suite) for task in tasks for state in states}
        if set(self.cases) != expected:
            raise ValueError(f"LIBERO {self.split} manifest must contain exactly its fixed task/state partition")

    def payload(self) -> dict:
        value = {
            "schema_version": 1 if self.suite == "libero_spatial" else 2,
            "suite": self.suite,
            "split": self.split,
            "cases": [{"task_id": case.task_id, "init_state_id": case.init_state_id} for case in self.cases],
        }
        if self.profile == ALIGNED_EVAL_PROFILE:
            value.update(schema_version=3, profile=self.profile)
        elif self.profile == SHARED_PROFILE:
            value.update(schema_version=4, profile=self.profile)
        elif self.profile == SINGLE_TASK_PROFILE:
            value.update(schema_version=5, profile=self.profile, task_id=self.task_id)
        return value

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.payload(), indent=4) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> LiberoManifest:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError(f"LIBERO manifest {path} must be an object")
        expected_keys = {"schema_version", "suite", "split", "cases"}
        schema = value.get("schema_version")
        if schema in (3, 4, 5):
            expected_keys.add("profile")
        if schema == 5:
            expected_keys.add("task_id")
        if set(value) != expected_keys:
            raise ValueError(f"LIBERO manifest {path} has unexpected or missing keys")
        if (
            type(schema) is not int
            or value["suite"] not in LIBERO_SUITES
            or schema not in ((1, 4) if value["suite"] == "libero_spatial" else (2, 3, 4, 5))
        ):
            raise ValueError(f"LIBERO manifest {path} has an unsupported schema or suite")
        profiles = {3: ALIGNED_EVAL_PROFILE, 4: SHARED_PROFILE, 5: SINGLE_TASK_PROFILE}
        if schema in profiles and value["profile"] != profiles[schema]:
            raise ValueError(f"LIBERO manifest {path} schema {schema} requires {profiles[schema]}")
        if not isinstance(value["cases"], list):
            raise ValueError(f"LIBERO manifest {path} cases must be a list")
        cases = []
        for index, case in enumerate(value["cases"]):
            try:
                cases.append(LiberoCase(**case, suite=value["suite"]))
            except (TypeError, ValueError) as error:
                raise type(error)(f"LIBERO manifest {path} cases[{index}]: {error}") from error
        return cls(
            value["split"],
            tuple(cases),
            value["suite"],
            value.get("profile", LEGACY_EVAL_PROFILE),
            value["task_id"] if schema == 5 else None,
        )


def fixed_manifest(
    split: str, suite: str = "libero_spatial", profile: str = LEGACY_EVAL_PROFILE, *, task_id: int | None = None
) -> LiberoManifest:
    states = _manifest_state_ids(split, suite, profile)
    tasks = _manifest_task_ids(suite, profile, task_id)
    return LiberoManifest(
        split, tuple(LiberoCase(task, state, suite) for task in tasks for state in states), suite, profile, task_id
    )


def load_partition(train_path: str, eval_path: str) -> tuple[LiberoManifest, LiberoManifest]:
    train, evaluation = LiberoManifest.read(train_path), LiberoManifest.read(eval_path)
    if train.split != "train" or evaluation.split != "eval" or train.suite != evaluation.suite:
        raise ValueError("LIBERO train/eval manifests must declare matching suites and their respective splits")
    for profile in (SHARED_PROFILE, SINGLE_TASK_PROFILE):
        if profile in (train.profile, evaluation.profile) and train.profile != evaluation.profile:
            raise ValueError(f"{profile} requires matching training and evaluation manifest profiles")
    if train.task_id != evaluation.task_id:
        raise ValueError("single_task_50 training and evaluation manifests must declare matching task_id")
    if (
        train.suite == "libero_spatial"
        and train.profile != SHARED_PROFILE
        and set(train.cases).intersection(evaluation.cases)
    ):
        raise ValueError("LIBERO-Spatial train/eval manifests must remain disjoint")
    return train, evaluation


@dataclass(frozen=True, order=True)
class LiberoTrial:
    """One evaluation trial; different seeds never alias the same initial state."""

    case: LiberoCase
    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("LIBERO trial seed must be a non-negative integer")

    @property
    def id(self) -> str:
        return f"{self.case.id}/seed:{self.seed}"


@dataclass(frozen=True)
class LiberoTrialManifest:
    """Explicit immutable evaluation inventory; training partitions keep their own contract."""

    suite: str
    trials: tuple[LiberoTrial, ...]

    def __post_init__(self) -> None:
        if self.suite not in LIBERO_SUITES or not self.trials:
            raise ValueError("LIBERO trial manifest requires a supported suite and nonempty trials")
        if any(trial.case.suite != self.suite for trial in self.trials):
            raise ValueError("LIBERO trial manifest contains another suite")
        if len(set(self.trials)) != len(self.trials):
            raise ValueError("LIBERO trial manifest contains duplicate trials")

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "suite": self.suite,
            "trials": [
                {"task_id": trial.case.task_id, "init_state_id": trial.case.init_state_id, "seed": trial.seed}
                for trial in self.trials
            ],
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.payload(), indent=4) + "\n")

    @classmethod
    def from_payload(cls, value: dict) -> LiberoTrialManifest:
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "suite", "trials"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("LIBERO trial manifest has unsupported keys or schema_version")
        if not isinstance(value["trials"], list):
            raise ValueError("LIBERO trial manifest trials must be a list")
        trials = []
        for index, row in enumerate(value["trials"]):
            if not isinstance(row, dict) or set(row) != {"task_id", "init_state_id", "seed"}:
                raise ValueError(f"LIBERO trials[{index}] requires task_id, init_state_id and seed")
            trials.append(LiberoTrial(LiberoCase(row["task_id"], row["init_state_id"], value["suite"]), row["seed"]))
        return cls(value["suite"], tuple(trials))

    @classmethod
    def read(cls, path: str | Path) -> LiberoTrialManifest:
        return cls.from_payload(json.loads(Path(path).read_text()))


def libero_seed(suite: str, phase: str, seed: int, *coordinates: int) -> int:
    """Stable request seeds, independent of process identities and execution chunk length."""
    payload = json.dumps([suite, phase, seed, *coordinates], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
