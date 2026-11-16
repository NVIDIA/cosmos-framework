# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-free preparation and exact regeneration of prospectively declared Spatial scenes.

This records seeded native construction/reset, not a partial MuJoCo state checkpoint.
No preparation attempt is replaced, filtered by a policy, or silently retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_RUNTIME,
    GeneratedLiberoCase,
)

FORMAT = "libero-generated-scenes-v1"
PLAN_FORMAT = "libero-generated-plan-v1"
RESET_ORDER = (
    "python-numpy-construction-seed/construct/python-numpy-reset-seed/native-reset-once/15-open-gripper-controls-v1"
)
POLICY_NOISE = "libero-generated-scene-policy-v1"
MODULE = "cosmos_framework.posttrain.embodied.sims.libero.generated"
PHYSICS_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "qfrc_applied",
    "xfrc_applied",
)
MODEL_FIELDS = ("body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos", "site_quat", "cam_pos", "cam_quat")
BUFFER_FIELDS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)
SNAPSHOT_FIELDS = (
    "model",
    "physics",
    "flat_state",
    "controller",
    "observations",
    "numpy_rng",
    "python_rng",
    "native_success",
    "native_clock",
    "gripper_action",
    "robot_buffers",
    "physical_sha256",
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"incomplete or unexpected {name} fields")


def validate_request(request):
    _keys(request, ("task_id", "construction_seed", "reset_seed"), "generated request")
    for name, upper in (("task_id", 10), ("construction_seed", 2**32), ("reset_seed", 2**32)):
        if type(request[name]) is not int or not 0 <= request[name] < upper:
            raise ValueError(f"invalid generated {name}")


def validate_source(source):
    _keys(
        source,
        ("runtime", "runtime_files", "native_source", "assets_sha256", "asset_files", "tasks", "renderer"),
        "source",
    )
    if source["runtime"] != LIBERO_RUNTIME.identity():
        raise ValueError("generated runtime pin differs")
    if source["native_source"] != native_source():
        raise ValueError("generated native source differs")
    _keys(source["runtime_files"], ("robosuite", "mujoco", "numpy"), "runtime file closure")
    for entry in source["runtime_files"].values():
        _keys(entry, ("sha256", "files"), "installed runtime inventory")
        if not _sha(entry["sha256"]) or type(entry["files"]) is not int or entry["files"] < 1:
            raise ValueError("generated runtime files are not bound")
    if not _sha(source["assets_sha256"]) or type(source["asset_files"]) is not int or source["asset_files"] < 1:
        raise ValueError("generated assets are not bound")
    if source["renderer"] not in ("egl", "osmesa", "glfw"):
        raise ValueError("generated scenes require an explicit supported MUJOCO_GL backend")
    if not isinstance(source["tasks"], list) or len(source["tasks"]) != 10:
        raise ValueError("generated source requires all ten tasks")
    for index, task in enumerate(source["tasks"]):
        _keys(task, ("task_id", "name", "language", "bddl_sha256"), "task source")
        if task["task_id"] != index or not _sha(task["bddl_sha256"]) or not task["name"] or not task["language"]:
            raise ValueError("generated task source differs")


def native_source():
    root = Path(__file__).parent
    posttrain = root.parents[2]
    return {
        name: file_sha(posttrain / name)
        for name in (
            "embodied/sims/libero/generated.py",
            "embodied/sims/libero/protocol.py",
            "embodied/sims/libero/runtime.py",
            "entrypoints/gr00t_rollout.py",
            "model/gr00t/rollout.py",
            "checkpoint/io.py",
        )
    }


def validate_plan(plan):
    _keys(plan, ("format", "source", "policies", "requests"), "generation plan")
    if plan["format"] != PLAN_FORMAT or not isinstance(plan["policies"], list) or not plan["policies"]:
        raise ValueError("generation plan requires its format and frozen policy set")
    for policy in plan["policies"]:
        if not isinstance(policy, dict):
            raise ValueError("invalid frozen policy binding")
        if policy.get("kind") == "artifact":
            _keys(policy, ("kind", "manifest_sha256"), "policy artifact")
            if not _sha(policy["manifest_sha256"]):
                raise ValueError("invalid policy artifact digest")
        elif policy.get("kind") == "native-base":
            _keys(
                policy,
                ("kind", "source", "adapter", "quantization", "task_config", "identity_sha256"),
                "native source policy",
            )
            if (
                not _sha(policy["identity_sha256"])
                or not isinstance(policy["source"], dict)
                or policy["source"].get("format") != "cosmos.gr00t-native-source.v1"
                or not isinstance(policy["adapter"], dict)
                or policy["adapter"].get("mode") not in ("bf16_frozen", "nf4_frozen")
                or not isinstance(policy["task_config"], dict)
                or not policy["task_config"]
            ):
                raise ValueError("native source policy requires its exact source, frozen mode and configuration")
        else:
            raise ValueError("unsupported frozen policy kind")
    if len({digest(policy) for policy in plan["policies"]}) != len(plan["policies"]):
        raise ValueError("duplicate frozen policy bindings")
    validate_source(plan["source"])
    if not isinstance(plan["requests"], list) or not plan["requests"]:
        raise ValueError("generation plan requires explicit requests")
    for request in plan["requests"]:
        validate_request(request)
    if len({digest(request) for request in plan["requests"]}) != len(plan["requests"]):
        raise ValueError("duplicate generated requests")


def scene_id(source, request):
    return digest({"source": source, "request": request, "reset_order": RESET_ORDER})


def describe_source():
    """Probe installed code/dependencies and hash assets/BDDL; never construct an environment."""
    from importlib.metadata import distribution
    from importlib.util import find_spec

    from libero.libero import benchmark, get_libero_path

    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator

    LiberoSimulator(embodiment="libero_sim").close()  # existing complete dependency/source probe
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if suite.n_tasks != 10:
        raise ValueError("generated source task count differs")
    assets = Path(get_libero_path("assets")).resolve(strict=True)
    files = {}
    for path in sorted(assets.rglob("*")):
        path.resolve(strict=True).relative_to(assets)
        if path.is_file():
            files[path.relative_to(assets).as_posix()] = file_sha(path)
    tasks = []
    for index in range(10):
        task = suite.get_task(index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        tasks.append({"task_id": index, "name": task.name, "language": task.language, "bddl_sha256": file_sha(bddl)})
    runtime_files = {}
    for name in ("robosuite", "mujoco", "numpy"):
        package = distribution(name)
        spec = find_spec(name)
        if (
            spec is None
            or not spec.origin
            or Path(spec.origin).resolve() != Path(package.locate_file(f"{name}/__init__.py")).resolve()
        ):
            raise ValueError(f"generated {name} import differs from its installed distribution")
        files_in_distribution = package.files
        if not files_in_distribution:
            raise ValueError(f"installed {name} has no file inventory")
        hashes = {
            str(path): file_sha(package.locate_file(path))
            for path in files_in_distribution
            if "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        runtime_files[name] = {"sha256": digest(hashes), "files": len(hashes)}
    result = {
        "runtime": LIBERO_RUNTIME.identity(),
        "runtime_files": runtime_files,
        "native_source": native_source(),
        "assets_sha256": digest(files),
        "asset_files": len(files),
        "tasks": tasks,
        "renderer": os.environ.get("MUJOCO_GL"),
    }
    validate_source(result)
    return result


def _value(value):
    """Canonical scalar/array evidence, rejecting opaque objects and nonfinite state."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("nonfinite or opaque generated state")
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _value(value.item())
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _value(item) for key, item in sorted(value.items())}
    raise ValueError(f"unsupported generated state type {type(value).__name__}")


def snapshot(env, observations):
    """Evidence for regeneration, not a serialization claimed to restore controller state."""
    import numpy as np

    native = env.env
    controller = native.robots[0].controller
    if len(native.robots) != 1 or controller.name != "OSC_POSE" or native.control_freq != 20:
        raise ValueError("generated scene requires the native one-Panda OSC_POSE/20Hz contract")
    if controller.interpolator_pos is not None or controller.interpolator_ori is not None:
        raise ValueError("unexpected generated controller interpolation")
    controller_state = {key: _value(value) for key, value in vars(controller).items() if key != "sim"}
    physics = {name: _value(np.asarray(getattr(env.sim.data, name))) for name in PHYSICS_FIELDS}
    model = {
        "xml_sha256": hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest(),
        "body_names": list(env.sim.model.body_names),
        **{name: _value(getattr(env.sim.model, name)) for name in MODEL_FIELDS},
        "object_body_ids": _value(native.obj_body_id),
        "action_spec": _value(native.action_spec),
        "timestep": _value(env.sim.model.opt.timestep),
        "gravity": _value(env.sim.model.opt.gravity),
    }
    result = {
        "model": model,
        "physics": physics,
        "flat_state": _value(env.get_sim_state()),
        "controller": controller_state,
        "observations": _value(observations),
        "numpy_rng": _value(np.random.get_state()),
        "python_rng": _value(random.getstate()),
        "native_success": bool(env.check_success()),
        "native_clock": _value(
            {
                name: getattr(native, name)
                for name in (
                    "timestep",
                    "cur_time",
                    "done",
                    "control_freq",
                    "control_timestep",
                    "hard_reset",
                    "deterministic_reset",
                )
            }
        ),
        "robot_buffers": {name: _value(vars(getattr(native.robots[0], name))) for name in BUFFER_FIELDS},
        "gripper_action": _value(native.robots[0].gripper.current_action),
    }
    # Excludes clocks/seeds: repeated physical scenes are not independent observations.
    result["physical_sha256"] = digest({"model": model, "qpos": physics["qpos"], "qvel": physics["qvel"]})
    return result


def _validate_snapshot(state, *, observations):
    _keys(state, SNAPSHOT_FIELDS, "scene snapshot")
    _keys(state["physics"], PHYSICS_FIELDS, "physics snapshot")
    _keys(
        state["model"],
        (*MODEL_FIELDS, "xml_sha256", "body_names", "object_body_ids", "action_spec", "timestep", "gravity"),
        "model snapshot",
    )
    _keys(state["robot_buffers"], BUFFER_FIELDS, "robot buffers")
    _keys(
        state["native_clock"],
        ("timestep", "cur_time", "done", "control_freq", "control_timestep", "hard_reset", "deterministic_reset"),
        "native clock",
    )
    if state["native_clock"]["control_freq"] != 20 or state["native_clock"]["deterministic_reset"] is not False:
        raise ValueError("generated native control/reset contract differs")
    required = {
        "goal_pos",
        "goal_ori",
        "initial_joint",
        "initial_ee_pos",
        "initial_ee_ori_mat",
        "new_update",
        "use_delta",
        "use_ori",
        "impedance_mode",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "kp",
        "kd",
    }
    if not isinstance(state["controller"], dict) or not required <= state["controller"].keys():
        raise ValueError("incomplete controller snapshot")
    if not isinstance(state["observations"], dict) or (
        observations
        and not {
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        }
        <= state["observations"].keys()
    ):
        raise ValueError("incomplete generated observations")

    def validate(value):
        if isinstance(value, dict):
            if "sha256" in value:
                _keys(value, ("dtype", "shape", "sha256"), "array evidence")
                if (
                    not _sha(value["sha256"])
                    or not isinstance(value["dtype"], str)
                    or not value["dtype"]
                    or not isinstance(value["shape"], list)
                    or any(type(n) is not int or n < 0 for n in value["shape"])
                ):
                    raise ValueError("invalid array evidence")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif (
            value is not None
            and type(value) not in (bool, int, str)
            and not (type(value) is float and math.isfinite(value))
        ):
            raise ValueError("invalid generated snapshot value")

    validate(state)
    if not _sha(state["model"]["xml_sha256"]) or type(state["native_success"]) is not bool:
        raise ValueError("invalid native model/success record")
    physical = digest({"model": state["model"], "qpos": state["physics"]["qpos"], "qvel": state["physics"]["qvel"]})
    if state["physical_sha256"] != physical:
        raise ValueError("generated physical identity differs")


def reset_scene(request, factory, *, expected=None):
    """Construct once, reset once, settle exactly 15 controls, verify before returning a live env."""
    import numpy as np

    validate_request(request)
    started = time.monotonic()
    env = None
    try:
        random.seed(request["construction_seed"])
        np.random.seed(request["construction_seed"])
        env = factory()
        constructed = snapshot(env, {})
        random.seed(request["reset_seed"])
        env.seed(request["reset_seed"])
        # ControlEnv.reset retries silently; invoke the same underlying native reset exactly once.
        raw = env.env.reset()
        before = snapshot(env, raw)
        zero = np.zeros(7, dtype=np.float32)
        zero[-1] = -1.0
        for _ in range(15):
            raw, _, _, _ = env.step(zero)
        evidence = {"constructed": constructed, "before_settle": before, "after_settle": snapshot(env, raw)}
        if expected is not None and evidence != expected:
            raise ValueError("generated scene regeneration differs from its exact manifest evidence")
        return env, raw, evidence, time.monotonic() - started
    except BaseException as failure:
        if env is not None:
            try:
                env.close()
            except BaseException as cleanup:
                raise failure from cleanup
        raise


def validate_manifest(value):
    _keys(value, ("format", "plan", "reset_order", "policy_noise", "records"), "generated manifest")
    if (value["format"], value["reset_order"], value["policy_noise"]) != (FORMAT, RESET_ORDER, POLICY_NOISE):
        raise ValueError("generated scene contract differs")
    plan = value["plan"]
    validate_plan(plan)
    if not isinstance(value["records"], list) or len(value["records"]) != len(plan["requests"]):
        raise ValueError("generated records do not cover every requested seed")
    seen = set()
    for request, record in zip(plan["requests"], value["records"], strict=True):
        _keys(record, ("request", "scene_id", "status", "evidence", "seconds", "stderr"), "prepared scene")
        if (
            record["status"] != "prepared"
            or record["request"] != request
            or record["scene_id"] != scene_id(plan["source"], request)
        ):
            raise ValueError("incomplete or mismatched generated attempt")
        GeneratedLiberoCase(request["task_id"], record["scene_id"])
        _keys(record["evidence"], ("constructed", "before_settle", "after_settle"), "scene evidence")
        if (
            type(record["seconds"]) not in (int, float)
            or not math.isfinite(record["seconds"])
            or record["seconds"] < 0
            or not isinstance(record["stderr"], str)
        ):
            raise ValueError("invalid generated attempt clock/log")
        for name, state in record["evidence"].items():
            _validate_snapshot(state, observations=name != "constructed")
        physical = record["evidence"]["after_settle"]["physical_sha256"]
        if physical in seen:
            raise ValueError("duplicate generated physical scene; keep attempts but do not count independent episodes")
        seen.add(physical)
    return value


def read_manifest(path, sha256):
    if not _sha(sha256):
        raise ValueError("generated manifest requires an explicit expected SHA256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("generated manifest digest differs")
    return validate_manifest(json.loads(raw))


def cases(manifest):
    return [GeneratedLiberoCase(row["request"]["task_id"], row["scene_id"]) for row in manifest["records"]]


def _worker(payload):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    plan, index = payload["plan"], payload["index"]
    validate_plan(plan)
    if describe_source() != plan["source"]:
        raise ValueError("generated source files/runtime/assets differ")
    request = plan["requests"][index]
    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(request["task_id"])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    def factory():
        return OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=0,
            ignore_done=True,
            horizon=100000,
        )

    env, _, evidence, seconds = reset_scene(request, factory, expected=payload.get("expected"))
    try:
        return {
            "request": request,
            "scene_id": scene_id(plan["source"], request),
            "status": "prepared",
            "evidence": evidence,
            "seconds": seconds,
        }
    finally:
        env.close()


def _publish_json(path, value):
    """Publish fully serialized JSON durably, without replacing any existing name."""
    from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes, fsync_directory

    path = Path(path)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    # The existing atomic helper replaces its private staging name; only the final hard link
    # is visible to consumers. An existing final file (including a symlink) always refuses.
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as staging:
        temporary = Path(staging) / "record.json"
        atomic_write_bytes(str(temporary), payload)
        os.link(temporary, path)
        fsync_directory(str(path.parent))


def prepare(plan, *, journal, expected=None):
    from cosmos_framework.posttrain.checkpoint.io import fsync_directory

    validate_plan(plan)
    if expected is not None:
        validate_manifest(expected)
        if expected["plan"] != plan:
            raise ValueError("verification plan differs from the prepared manifest")
    journal = Path(journal)
    journal.mkdir()  # Never reopen or resume a previous attempt, even after interruption.
    fsync_directory(str(journal.parent))
    plan_sha256 = digest(plan)
    _publish_json(
        journal / "header.json",
        {
            "format": "libero-generated-attempts-v1",
            "plan": plan,
            "plan_sha256": plan_sha256,
            "verification_manifest_sha256": digest(expected) if expected is not None else None,
            "reset_order": RESET_ORDER,
            "policy_noise": POLICY_NOISE,
        },
    )
    records = []
    for index, request in enumerate(plan["requests"]):
        _publish_json(
            journal / f"{index:06d}.started.json",
            {"index": index, "plan_sha256": plan_sha256, "request": request, "status": "started"},
        )
        payload = {"plan": plan, "index": index}
        if expected is not None:
            payload["expected"] = expected["records"][index]["evidence"]
        try:
            child = subprocess.run(
                [sys.executable, "-m", MODULE, "--worker"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            record = json.loads(child.stdout) if child.returncode == 0 else {"status": "error", "error": child.stderr}
            if child.returncode == 0:
                _keys(record, ("request", "scene_id", "status", "evidence", "seconds"), "worker record")
                if (
                    record["request"] != request
                    or record["scene_id"] != scene_id(plan["source"], request)
                    or record["status"] != "prepared"
                ):
                    raise ValueError("worker returned a different generated request")
            record["stderr"] = child.stderr
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired can contain bytes even with text=True, including a
            # truncated UTF-8 sequence. Retain both streams in the durable failure.
            record = {
                "status": "error",
                "error": str(exc),
                **{
                    name: value.decode("utf-8", errors="backslashreplace") if isinstance(value, bytes) else value or ""
                    for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr))
                },
            }
        except (ValueError, TypeError, OSError) as exc:
            record = {"status": "error", "error": str(exc)}
        # Every original request remains visible, including a failed worker. Never replace or resample it.
        record = {**record, "request": request, "scene_id": scene_id(plan["source"], request)}
        _publish_json(
            journal / f"{index:06d}.result.json", {"index": index, "plan_sha256": plan_sha256, "record": record}
        )
        records.append(record)
    return {
        "format": FORMAT,
        "plan": plan,
        "reset_order": RESET_ORDER,
        "policy_noise": POLICY_NOISE,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true", help="Hash source/assets without generating scenes")
    action.add_argument("--plan", type=Path, help="Prospectively frozen source/policy/seeds JSON")
    action.add_argument("--verify", type=Path, help="Regenerate an existing manifest in fresh processes")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sha256", help="Required digest for --verify")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        import contextlib

        payload = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = _worker(payload)
        print(json.dumps(result, allow_nan=False))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    journal = args.output.with_name(args.output.name + ".attempts")
    if args.verify:
        previous = read_manifest(args.verify, args.sha256)
        result = prepare(previous["plan"], journal=journal, expected=previous)
    elif args.plan:
        result = prepare(json.loads(args.plan.read_text()), journal=journal)
    else:
        result = describe_source()
    if not args.describe:
        validate_manifest(result)  # failed/duplicate attempts remain in the journal, never a usable manifest
    _publish_json(args.output, result)


if __name__ == "__main__":
    main()
