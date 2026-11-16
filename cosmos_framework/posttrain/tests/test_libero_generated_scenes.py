# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU protocol/reset fixtures; these do not generate or qualify real LIBERO scenes."""

import copy
import json
import random
import runpy
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.sims.libero import generated as g
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
    LIBERO_RUNTIME,
    GeneratedLiberoCase,
    LiberoCase,
    execution_protocol,
)
from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession, LiberoSimulator
from cosmos_framework.posttrain.entrypoints.gr00t_rollout import load_cases

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Controller:
    name = "OSC_POSE"

    def __init__(self):
        self.interpolator_pos = self.interpolator_ori = None
        self.goal_pos = self.initial_ee_pos = np.zeros(3)
        self.goal_ori = self.initial_ee_ori_mat = np.eye(3)
        self.initial_joint = np.zeros(7)
        self.new_update = True
        self.use_delta = self.use_ori = True
        self.impedance_mode = "fixed"
        self.input_min, self.input_max = -np.ones(6), np.ones(6)
        self.output_min, self.output_max = -np.ones(6), np.ones(6)
        self.kp, self.kd = np.full(6, 150.0), np.full(6, 24.0)
        self.sim = object()  # deliberately not a serializable controller field


class Physics:
    """API-shaped fixture with fixture placement outside qpos and hidden controller state."""

    def __init__(self, trace, fail=None):
        self.trace, self.fail, self.closed = trace, fail, False
        trace.append(("construct", float(np.random.random()), random.random()))
        self.model = SimpleNamespace(
            **{name: np.zeros((2, 4 if name.endswith("quat") else 3)) for name in g.MODEL_FIELDS}
        )
        self.model.get_xml = lambda: "<model><fixture/></model>"
        self.model.body_names = ("world", "fixture")
        self.model.opt = SimpleNamespace(timestep=0.002, gravity=np.array([0.0, 0.0, -9.81]))
        self.data = SimpleNamespace(**{name: np.zeros(7) for name in g.PHYSICS_FIELDS})
        self.sim = SimpleNamespace(data=self.data, model=self.model)
        robot = SimpleNamespace(controller=Controller(), gripper=SimpleNamespace(current_action=np.zeros(1)))
        for name in g.BUFFER_FIELDS:
            setattr(robot, name, SimpleNamespace(last=np.zeros(7), current=np.zeros(7)))
        self.env = SimpleNamespace(
            robots=[robot],
            control_freq=20,
            control_timestep=0.05,
            hard_reset=True,
            deterministic_reset=False,
            done=False,
            timestep=0,
            cur_time=0.0,
            obj_body_id={"fixture": 1},
            action_spec=(-np.ones(7), np.ones(7)),
            reset=self.native_reset,
        )

    def seed(self, seed):
        self.trace.append(("seed", seed))
        np.random.seed(seed)

    def reset(self):
        pytest.fail("wrapper reset must not hide retries")

    def set_init_state(self, state):
        pytest.fail("generated scenes must not restore only a flat MuJoCo state")

    def native_reset(self):
        self.trace.append(("reset",))
        if self.fail == "reset":
            raise RuntimeError("native reset failed")
        self.model.body_pos[1] = np.random.uniform(-0.02, 0.02, 3)
        self.data.qpos[:] = np.random.uniform(-0.01, 0.01, 7)
        return self.observe()

    def observe(self):
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": self.data.qpos[:3].copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def step(self, action):
        self.trace.append(("step", action.tolist()))
        self.env.timestep += 1
        self.env.cur_time += 0.05
        if self.fail == "step" and self.env.timestep == 3:
            raise RuntimeError("native step failed")
        self.data.qvel[:] = self.env.timestep / 1000
        self.env.robots[0].controller.goal_pos = self.data.qpos[:3].copy()
        return self.observe(), 0.0, False, {}

    def check_success(self):
        return False

    def get_sim_state(self):
        return np.concatenate(([self.env.cur_time], self.data.qpos, self.data.qvel))

    def close(self):
        self.closed = True
        self.trace.append(("close",))


@pytest.fixture
def manifest():
    source = {
        "runtime": LIBERO_RUNTIME.identity(),
        "native_source": g.native_source(),
        "runtime_files": {name: {"sha256": "f" * 64, "files": 10} for name in ("robosuite", "mujoco", "numpy")},
        "assets_sha256": "a" * 64,
        "asset_files": 10,
        "renderer": "egl",
        "tasks": [
            {"task_id": i, "name": f"task_{i}", "language": f"do task {i}", "bddl_sha256": "b" * 64} for i in range(10)
        ],
    }
    request = {"task_id": 2, "construction_seed": 123, "reset_seed": 456}
    env, _, evidence, seconds = g.reset_scene(request, lambda: Physics([]))
    env.close()
    plan = {
        "format": g.PLAN_FORMAT,
        "source": source,
        "policies": [
            {"kind": "artifact", "manifest_sha256": "c" * 64},
            {"kind": "artifact", "manifest_sha256": "d" * 64},
        ],
        "requests": [request],
    }
    record = {
        "request": request,
        "scene_id": g.scene_id(source, request),
        "status": "prepared",
        "evidence": evidence,
        "seconds": seconds,
        "stderr": "",
    }
    return {
        "format": g.FORMAT,
        "plan": plan,
        "reset_order": g.RESET_ORDER,
        "policy_noise": g.POLICY_NOISE,
        "records": [record],
    }


def test_seed_before_constructor_native_reset_once_and_exact_settling(manifest):
    trace = []
    request = manifest["plan"]["requests"][0]
    env, raw, evidence, seconds = g.reset_scene(
        request, lambda: Physics(trace), expected=manifest["records"][0]["evidence"]
    )
    assert [row[0] for row in trace] == ["construct", "seed", "reset"] + ["step"] * 15
    assert all(row[1] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] for row in trace if row[0] == "step")
    assert seconds >= 0 and not env.closed
    assert evidence["after_settle"]["observations"]["robot0_eef_pos"] == g._value(raw["robot0_eef_pos"])
    assert evidence["constructed"]["numpy_rng"] != evidence["before_settle"]["numpy_rng"]
    assert evidence["before_settle"]["flat_state"] != evidence["after_settle"]["flat_state"]
    env.close()


@pytest.mark.parametrize("failure", ["reset", "step", "evidence", "opaque", "nonfinite"])
def test_failed_reset_settle_snapshot_or_verification_closes_and_never_retries(manifest, failure):
    env = None

    def factory():
        nonlocal env
        env = Physics([], failure)
        if failure == "opaque":
            env.env.robots[0].controller.unexpected = object()
        if failure == "nonfinite":
            env.data.qpos[0] = np.nan
        return env

    expected = copy.deepcopy(manifest["records"][0]["evidence"])
    if failure == "evidence":
        expected["after_settle"]["controller"]["new_update"] = False
    with pytest.raises((RuntimeError, ValueError)):
        g.reset_scene(manifest["plan"]["requests"][0], factory, expected=expected)
    assert env.closed and sum(row[0] == "reset" for row in env.trace) <= 1


@pytest.mark.parametrize("field", ["model", "controller", "physics", "observations", "numpy_rng", "robot_buffers"])
def test_verification_covers_more_than_flat_qpos_qvel(manifest, field):
    expected = copy.deepcopy(manifest["records"][0]["evidence"])
    expected["after_settle"][field] = {}
    env = Physics([])
    with pytest.raises(ValueError, match="regeneration differs"):
        g.reset_scene(manifest["plan"]["requests"][0], lambda: env, expected=expected)
    assert env.closed


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "missing_snapshot",
        "missing_controller",
        "missing_observation",
        "missing_physics",
        "physical_hash",
        "missing_seed",
        "bad_seed",
        "bool_seed",
        "duplicate_request",
        "wrong_order",
        "failed",
        "nonfinite_clock",
        "public_id",
    ],
)
def test_incomplete_tampered_or_ambiguous_manifest_refuses(manifest, mutation):
    value = copy.deepcopy(manifest)
    row = value["records"][0]
    if mutation == "source":
        value["plan"]["source"]["native_source"]["embodied/sims/libero/runtime.py"] = "0" * 64
    elif mutation == "missing_snapshot":
        row["evidence"].pop("constructed")
    elif mutation == "missing_controller":
        row["evidence"]["after_settle"]["controller"] = {}
    elif mutation == "missing_observation":
        row["evidence"]["after_settle"]["observations"] = {}
    elif mutation == "missing_physics":
        row["evidence"]["after_settle"]["physics"].pop("ctrl")
    elif mutation == "physical_hash":
        row["evidence"]["after_settle"]["physical_sha256"] = "0" * 64
    elif mutation == "missing_seed":
        value["plan"]["requests"][0].pop("reset_seed")
    elif mutation == "bad_seed":
        value["plan"]["requests"][0]["reset_seed"] = -1
    elif mutation == "bool_seed":
        value["plan"]["requests"][0]["reset_seed"] = True
    elif mutation == "duplicate_request":
        value["plan"]["requests"] *= 2
    elif mutation == "wrong_order":
        value["reset_order"] = "restore-flat-state"
    elif mutation == "failed":
        row["status"] = "error"
    elif mutation == "nonfinite_clock":
        row["seconds"] = float("nan")
    elif mutation == "public_id":
        value["plan"]["requests"][0]["init_state_id"] = 1
    with pytest.raises(ValueError):
        g.validate_manifest(value)


def test_pairing_identity_excludes_policy_set_but_rejects_duplicate_physics(manifest):
    original = g.cases(g.validate_manifest(manifest))[0]
    changed = copy.deepcopy(manifest)
    changed["plan"]["policies"] = [{"kind": "artifact", "manifest_sha256": "e" * 64}]
    assert g.cases(g.validate_manifest(changed))[0] == original
    request = dict(changed["plan"]["requests"][0], construction_seed=789)
    changed["plan"]["requests"].append(request)
    duplicate = copy.deepcopy(changed["records"][0])
    duplicate.update(request=request, scene_id=g.scene_id(changed["plan"]["source"], request))
    changed["records"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate generated physical"):
        g.validate_manifest(changed)


def test_digest_and_paired_frozen_artifacts_before_model_imports(tmp_path, manifest):
    path = tmp_path / "scenes.json"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    marker = artifact / ".posttrain-checkpoint-artifact.json"
    marker.write_text("candidate")
    manifest["plan"]["policies"] = [
        {"kind": "artifact", "manifest_sha256": g.file_sha(marker)},
        {"kind": "artifact", "manifest_sha256": g.digest("control")},
    ]
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        cases=None,
        generated_scenes=str(path),
        generated_scenes_sha256=g.file_sha(path),
        checkpoint=str(artifact),
        base=False,
        dense=False,
        unmerged=False,
    )
    selected, loaded = load_cases(args)
    assert asdict(selected[0]) == {
        "task_id": 2,
        "scene_id": manifest["records"][0]["scene_id"],
        "suite": "libero_spatial",
    }
    assert loaded == manifest
    marker.write_text("unfrozen")
    with pytest.raises(ValueError, match="different frozen policy"):
        load_cases(args)
    args.cases = "public.json"
    with pytest.raises(ValueError, match="exclusively"):
        load_cases(args)
    args.cases = None
    args.generated_scenes_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest differs"):
        load_cases(args)


def test_published_cases_serialization_unchanged_and_generated_ids_cannot_impersonate_them(tmp_path):
    path = tmp_path / "public.json"
    path.write_text(json.dumps([{"task_id": 2, "init_state_id": 3}]))
    args = SimpleNamespace(cases=str(path), generated_scenes=None, generated_scenes_sha256=None)
    selected, prepared = load_cases(args)
    assert selected == [LiberoCase(2, 3)] and prepared is None
    assert asdict(selected[0]) == {"task_id": 2, "init_state_id": 3, "suite": "libero_spatial"}
    with pytest.raises(TypeError):
        GeneratedLiberoCase(task_id=2, scene_id="a" * 64, init_state_id=3)
    with pytest.raises(TypeError):
        LiberoCase(task_id=2, scene_id="a" * 64)


def test_actual_module_main_uses_canonical_fresh_worker_and_retains_failed_seed(tmp_path, manifest, monkeypatch):
    plan_path, output = tmp_path / "plan.json", tmp_path / "result.json"
    manifest["plan"]["requests"].append({"task_id": 3, "construction_seed": 9, "reset_seed": 10})
    plan_path.write_text(json.dumps(manifest["plan"]))
    commands = []

    def child(command, **kw):
        payload = json.loads(kw["input"])
        commands.append((command, payload))
        if payload["index"] == 0:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({key: value for key, value in manifest["records"][0].items() if key != "stderr"}),
                stderr="fixture constructor log",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="RandomizationError: invalid placement")

    monkeypatch.setattr(subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", [g.MODULE, "--plan", str(plan_path), "--output", str(output)])
    with pytest.raises(ValueError):
        runpy.run_module(g.MODULE, run_name="__main__")
    result = json.loads(output.read_text())
    assert len(commands) == 2 and all(
        command == [sys.executable, "-m", g.MODULE, "--worker"] for command, _ in commands
    )
    assert result["records"][1]["request"] == manifest["plan"]["requests"][1]
    assert result["records"][1]["status"] == "error" and "RandomizationError" in result["records"][1]["error"]
    with pytest.raises(ValueError):
        g.validate_manifest(result)
    with pytest.raises(SystemExit):
        runpy.run_module(g.MODULE, run_name="__main__")
    assert len(commands) == 2  # create-only refusal, no replacement attempts


def test_generated_native_session_checks_manifest_before_observation_and_forbids_training(manifest, monkeypatch):
    # Existing runtime dispatch with a controlled upstream module; no simulator construction.
    from types import ModuleType

    module = ModuleType("libero.libero.utils")
    module.get_libero_path = lambda _: "/fixture"
    monkeypatch.setitem(sys.modules, "libero.libero.utils", module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", SimpleNamespace(OffScreenRenderEnv=Physics))
    session = object.__new__(LiberoChunkSession)
    session._generated_scene = manifest["records"][0]
    session._envs, session._tasks = [None], [-1]
    session._suite = SimpleNamespace(
        get_task=lambda _: SimpleNamespace(problem_folder="spatial", bddl_file="task.bddl", language="task")
    )
    monkeypatch.setattr(session, "_open_env", lambda *args: Physics([]))
    observed = []
    monkeypatch.setattr(session, "_reset_statistics", lambda lane, raw: observed.append(raw))
    session._reset(0)
    assert len(observed) == 1 and session._tasks == [2]
    with pytest.raises(ValueError, match="cannot reset"):
        session._reset(0)
    sim = object.__new__(LiberoSimulator)
    sim._generated = manifest
    with pytest.raises(ValueError, match="evaluation-only"):
        sim.open_window(0)
    with pytest.raises(TypeError):
        sim.open_evaluation({"task_id": 2, "init_state_id": 3}, 17)


def test_generated_session_failure_never_publishes_observation(manifest, monkeypatch):
    from types import ModuleType

    module = ModuleType("libero.libero.utils")
    module.get_libero_path = lambda _: "/fixture"
    monkeypatch.setitem(sys.modules, "libero.libero.utils", module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", SimpleNamespace(OffScreenRenderEnv=Physics))
    session = object.__new__(LiberoChunkSession)
    session._generated_scene = copy.deepcopy(manifest["records"][0])
    session._generated_scene["evidence"]["after_settle"]["controller"]["new_update"] = False
    session._envs, session._tasks = [None], [-1]
    session._suite = SimpleNamespace(
        get_task=lambda _: SimpleNamespace(problem_folder="spatial", bddl_file="task.bddl", language="task")
    )
    physics = []

    def factory(*args):
        env = Physics([])
        physics.append(env)
        return env

    monkeypatch.setattr(session, "_open_env", factory)
    monkeypatch.setattr(session, "_reset_statistics", lambda *args: pytest.fail("unverified observations escaped"))
    with pytest.raises(ValueError, match="regeneration differs"):
        session._reset(0)
    assert physics[0].closed and session._envs == [None]


def test_profile_control_math_unchanged():
    protocol = execution_protocol("libero_spatial", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)
    assert (protocol.execution_horizon, protocol.max_episode_steps, protocol.reset_steps, protocol.reset_gripper) == (
        8,
        256,
        15,
        -1.0,
    )


@pytest.mark.parametrize("mode", ["bf16_frozen", "nf4_frozen"])
def test_native_base_controls_share_manifest_and_bind_full_source_config_identity(tmp_path, manifest, mode):
    from cosmos_framework.posttrain.entrypoints.gr00t_rollout import base_policy_binding, require_generated_policy
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    identity = {
        "format": "cosmos.family-backbone.v1",
        "base": recorded_source(),
        "adapter": {"mode": mode},
        "quantization": None if mode == "bf16_frozen" else {"format": "nf4", "nested": True},
        "task_contract": {"config": {"num_inference_timesteps": 4, "execution_horizon": 8}},
        "frozen_tensors": {"native_weight": {"sha256": "a" * 64}},
    }
    manifest["plan"]["policies"].append(base_policy_binding(identity))
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        cases=None,
        generated_scenes=str(path),
        generated_scenes_sha256=g.file_sha(path),
        checkpoint=None,
        base=True,
        dense=False,
        unmerged=False,
    )
    selected, loaded = load_cases(args)
    assert selected == g.cases(manifest) and loaded == manifest
    policy = SimpleNamespace(net=SimpleNamespace(_cosmos_adapter_checkpoint_codec=SimpleNamespace(identity=identity)))
    require_generated_policy(args, policy, manifest)
    identity["task_contract"]["config"]["num_inference_timesteps"] = 5
    with pytest.raises(ValueError, match="frozen generated-scene policy set"):
        require_generated_policy(args, policy, manifest)
    identity["training_contract"] = {}
    with pytest.raises(ValueError, match="untrained native"):
        base_policy_binding(identity)


def test_describe_hashes_actual_robot_controller_physics_and_assets_without_env(tmp_path, monkeypatch):
    import importlib.metadata
    import importlib.util
    from pathlib import PurePosixPath
    from types import ModuleType

    assets, bddl, installed = tmp_path / "assets", tmp_path / "bddl", tmp_path / "installed"
    assets.mkdir()
    bddl.mkdir()
    installed.mkdir()
    (assets / "bowl.obj").write_bytes(b"libero asset")
    for i in range(10):
        (bddl / f"{i}.bddl").write_text(f"native task {i}")
    for name in ("robot.xml", "osc_pose.json", "reset.py", "physics.so", "numpy.so"):
        (installed / name).write_text(name)
    paths = {"robosuite": ["robot.xml", "osc_pose.json", "reset.py"], "mujoco": ["physics.so"], "numpy": ["numpy.so"]}
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            files=[PurePosixPath(p) for p in paths[name]], locate_file=lambda p: installed / p
        ),
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(installed / name / "__init__.py"))
    )
    libero = ModuleType("libero.libero")
    libero.get_libero_path = lambda name: str(assets if name == "assets" else bddl)
    suite = SimpleNamespace(
        n_tasks=10,
        get_task=lambda i: SimpleNamespace(
            name=f"task_{i}", language=f"do task {i}", problem_folder="", bddl_file=f"{i}.bddl"
        ),
    )
    libero.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
    monkeypatch.setitem(sys.modules, "libero.libero", libero)
    monkeypatch.setattr(LiberoSimulator, "probe", lambda self: None)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    baseline = g.describe_source()
    for package, file in (
        ("robosuite", "robot.xml"),
        ("robosuite", "osc_pose.json"),
        ("robosuite", "reset.py"),
        ("mujoco", "physics.so"),
        ("numpy", "numpy.so"),
    ):
        original = (installed / file).read_bytes()
        (installed / file).write_bytes(original + b"changed")
        assert g.describe_source()["runtime_files"][package] != baseline["runtime_files"][package]
        (installed / file).write_bytes(original)
    (assets / "bowl.obj").write_bytes(b"changed mesh")
    assert g.describe_source()["assets_sha256"] != baseline["assets_sha256"]
    (bddl / "0.bddl").write_text("different initialization region")
    assert g.describe_source()["tasks"][0] != baseline["tasks"][0]


def test_timeout_retains_original_seed_and_runs_next_requested_seed_once(manifest, monkeypatch):
    manifest["plan"]["requests"].append({"task_id": 0, "construction_seed": 1, "reset_seed": 2})
    calls = []

    def timeout(command, **kw):
        calls.append(json.loads(kw["input"])["index"])
        raise subprocess.TimeoutExpired(command, kw["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    result = g.prepare(manifest["plan"])
    assert calls == [0, 1]
    assert [row["request"] for row in result["records"]] == manifest["plan"]["requests"]
    assert all(row["status"] == "error" for row in result["records"])


def test_generated_policy_noise_is_stable_paired_and_separate_from_public(monkeypatch):
    import torch
    from torch import nn

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.model.gr00t import rollout

    policy = SimpleNamespace(config=make_config().model, net=nn.Linear(2, 2))
    monkeypatch.setattr(rollout, "prepare_observations", lambda policy, obs: ({}, None))
    monkeypatch.setattr(rollout, "predict_processed", lambda *a, **kw: (None, torch.randn(1, 16, 7).numpy()))
    wrapper = rollout.NativeArtifactRollout(policy)
    rng = torch.get_rng_state().clone()
    context = {"case": {"suite": "libero_spatial", "task_id": 2, "scene_id": "a" * 64}, "seed": 17, "decision": 0}
    first = wrapper.act([{}], context=context)
    np.testing.assert_array_equal(first, wrapper.act([{}], context=context))
    changed = copy.deepcopy(context)
    changed["case"]["scene_id"] = "b" * 64
    assert not np.array_equal(first, wrapper.act([{}], context=changed))
    published = {**context, "case": {"suite": "libero_spatial", "task_id": 2, "init_state_id": 3}}
    assert not np.array_equal(first, wrapper.act([{}], context=published))
    assert torch.equal(torch.get_rng_state(), rng)


def test_independent_cpu_fixture_processes_reproduce_all_evidence(manifest):
    # Real process isolation; fixture physics only, not actual LIBERO/MuJoCo generation.
    code = """
import json, sys
from cosmos_framework.posttrain.tests.test_libero_generated_scenes import Physics
from cosmos_framework.posttrain.embodied.sims.libero import generated as g
request = json.loads(sys.stdin.read())
env, raw, evidence, seconds = g.reset_scene(request, lambda: Physics([]))
env.close()
print(json.dumps(evidence))
"""
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(manifest["plan"]["requests"][0]),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        results.append(json.loads(child.stdout))
    assert results[0] == results[1] == manifest["records"][0]["evidence"]


def test_cleanup_does_not_mask_original_reset_failure():
    env = Physics([], "reset")

    def close():
        raise RuntimeError("cleanup failed")

    env.close = close
    with pytest.raises(RuntimeError, match="native reset failed") as error:
        g.reset_scene({"task_id": 0, "construction_seed": 1, "reset_seed": 2}, lambda: env)
    assert str(error.value.__cause__) == "cleanup failed"


def test_wrong_worker_identity_is_retained_as_failure(manifest, monkeypatch):
    wrong = {key: value for key, value in manifest["records"][0].items() if key != "stderr"}
    wrong["scene_id"] = "f" * 64
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(wrong), stderr="")
    )
    result = g.prepare(manifest["plan"])
    assert result["records"][0]["status"] == "error"
    assert "different generated request" in result["records"][0]["error"]
    assert result["records"][0]["request"] == manifest["plan"]["requests"][0]
