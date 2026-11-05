# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native/Omni zero-update calibration; strict pilot gates the common 200-case evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    ALIGNED_EVAL_PROFILE,
    LiberoCase,
    fixed_manifest,
    libero_seed,
)
from cosmos_framework.posttrain.tests.helpers.gr00t_calibration import validate_baseline


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, allow_nan=False) + "\n")


def provenance(config) -> dict:
    """Hash the actual fixed model, processor, simulator source and assets before inference."""
    from concurrent.futures import ThreadPoolExecutor

    import gr00t
    import yaml
    from tqdm import tqdm

    from cosmos_framework.posttrain.model.gr00t.build import libero10_model_identity

    paths = set(Path(config.model.checkpoint_path).glob("*"))
    paths.update(Path(config.model.checkpoint_path).glob("processor/*"))
    source = Path(gr00t.__file__).parent
    paths.update(source.rglob("*.py"))
    libero_config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    assets = yaml.safe_load(libero_config.read_text())
    paths.add(libero_config)
    for key in ("assets", "bddl_files", "init_states"):
        paths.update(Path(assets[key]).rglob("*"))
    simulator = subprocess.run(
        [
            config.environment.config.sim_python,
            "-c",
            "import importlib.metadata as m,json; print(json.dumps({'robosuite_assets':str(m.distribution('robosuite').locate_file('robosuite/models/assets')), 'versions':{d.metadata['Name']:d.version for d in m.distributions()}}))",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    simulator = json.loads(simulator.stdout)
    paths.update(Path(simulator["robosuite_assets"]).rglob("*"))
    paths.update(Path(assets["bddl_files"]).parent.rglob("*.py"))
    posttrain = Path(__file__).resolve().parents[1]
    # Every posttrain source that shapes a calibration decision: model forwards, serving, the
    # simulator runtime and the test-owned calibration backend. A missing entry means the evidence
    # would not describe the code that ran, so refuse rather than hash a partial set.
    sources = (
        "tests/gr00t_baseline_cli.py",
        "tests/helpers/gr00t_calibration.py",
        "tests/helpers/gr00t_diagnostics.py",
        "tests/helpers/gr00t_omni_calibration.py",
        "tests/helpers/libero_calibration_sim.py",
        "model/gr00t/ppo.py",
        "model/gr00t/policy.py",
        "model/gr00t/serving/body.py",
        "model/gr00t/serving/omni_pipeline.py",
        "embodied/sims/libero/protocol.py",
        "embodied/sims/libero/config.py",
        "embodied/sims/libero/adapter.py",
        "embodied/sims/libero/runtime.py",
    )
    missing = [name for name in sources if not (posttrain / name).is_file()]
    if missing:
        raise FileNotFoundError(f"calibration provenance sources are missing: {missing}")
    paths.update(posttrain / name for name in sources)
    backbone = (
        Path(os.environ["HF_HOME"])
        / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
    )
    if not backbone.is_dir():
        raise ValueError(f"pinned backbone cache missing: {backbone}")
    paths.update(backbone.glob("*"))
    files = sorted(path for path in paths if path.is_file())

    def digest(path):
        with path.open("rb") as stream:
            return str(path), hashlib.file_digest(stream, "sha256").hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(tqdm(pool.map(digest, files), total=len(files), desc="Baseline source SHA256"))
    runtime = Path(os.environ["GR00T_RUNTIME_DIR"]) / "preflight"
    return {
        "model": libero10_model_identity(config.model.checkpoint_path),
        "simulator": simulator,
        "files_sha256": hashes,
        "runtime": {path.stem: json.loads(path.read_text()) for path in runtime.glob("*.json")},
    }


def native_observation(raw: dict) -> dict:
    # These native_* fields are produced independently by NVIDIA's real LiberoEnv.
    return {
        "video": {key: raw[f"native_{key}"][None, None] for key in ("image", "wrist_image")},
        "state": {key: raw[f"native_{key}"][None, None] for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")},
        "language": {"annotation.human.action.task_description": [[raw["prompt"]]]},
    }


def capture_differences(left: dict, right: dict, prefix: str) -> list[dict]:
    from cosmos_framework.posttrain.tests.helpers.parity_diff import compare, flatten_tensors

    def flatten(capture):
        result = flatten_tensors(capture)
        for kind in ("latents_before", "velocities"):
            if len(capture[kind]) != 4:
                raise AssertionError(f"{kind} must contain exactly four real ODE steps")
            result.update({f"{kind}.{index}": item for index, item in enumerate(capture[kind])})
        return result

    a, b = flatten(left), flatten(right)
    if set(a) != set(b):
        raise AssertionError(f"capture tensor inventories differ: {set(a) ^ set(b)}")
    return [compare(f"{prefix}/{name}", a[name], b[name]).as_json() for name in a]


class CalibrationWorker:
    def __init__(self, config, recipe, output: str, rank: int):
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.config.config_joint_ppo import resolve_joint_config
        from cosmos_framework.posttrain.tests.helpers.gr00t_parity_gate import isaac_auto_classes

        validate_baseline(config)
        self.config, self.recipe, self.rank = config, recipe, rank
        # The task adapter owns simulator launch data; the lazy config node has fields but no properties.
        self.environment = instantiate(resolve_joint_config(config).environment)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.native, self.omni, self.gateway = None, None, None
        from cosmos_framework.posttrain.control.termination import TerminationWatch

        self.watch = TerminationWatch()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from gr00t.model.modules.dit import _should_force_math_sdpa

        if not _should_force_math_sdpa():
            raise AssertionError("GR00T_DIT_SDPA_MODE=math did not take effect")
        torch.manual_seed(0)
        with isaac_auto_classes():
            self.native = Gr00tPolicy("libero_sim", config.model.checkpoint_path, device="cuda:0")
        self.initial_native = self.native_fingerprint()
        self.mode = "native"

    def native_fingerprint(self):
        import torch

        digest = hashlib.sha256()
        dtypes = set()
        for name, parameter in self.native.model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            dtypes.add(str(parameter.dtype))
        if dtypes != {"torch.bfloat16"}:
            raise AssertionError(f"native model is not entirely BF16: {dtypes}")
        for name, buffer in self.native.model.named_buffers():
            digest.update(name.encode())
            digest.update(buffer.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def native_smoke(self, tasks: list[int]) -> dict:
        """Run the unmodified NVIDIA client/program against its actual native policy server."""
        import torch
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
        from gr00t.policy.server_client import PolicyClient, PolicyServer

        port = 52000 + self.rank
        server = PolicyServer(Gr00tSimPolicyWrapper(self.native), host="127.0.0.1", port=port)

        def serve():
            torch.cuda.set_device(0)
            torch.manual_seed(0)
            server.run()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        results = []
        try:
            names = subprocess.run(
                [
                    self.config.environment.config.sim_python,
                    "-c",
                    "import json; from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_10'](); print(json.dumps([s.get_task(i).name for i in range(10)]))",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            names = json.loads(names.stdout.splitlines()[-1])
            for task in tasks:
                if self.watch.should_stop():
                    raise InterruptedError("native baseline stopped at an episode boundary")
                command = [
                    self.config.environment.config.sim_python,
                    "-m",
                    "gr00t.eval.rollout_policy",
                    "--policy-client-host",
                    "127.0.0.1",
                    "--policy-client-port",
                    str(port),
                    "--env-name",
                    f"libero_sim/{names[task]}",
                    "--n-envs",
                    "1",
                    "--n-episodes",
                    "1",
                    "--n-action-steps",
                    "8",
                    "--max-episode-steps",
                    "720",
                    "--seed",
                    "0",
                    "--video-dir",
                    str(self.output / f"native-videos/task-{task}"),
                ]
                with (self.output / f"native-task-{task}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                results.append({"task": task, "command": command})
                print(f"NATIVE_SMOKE task={task} complete", flush=True)
        finally:
            with PolicyClient("127.0.0.1", port) as client:
                client.kill_server()
            thread.join(timeout=30)
            server.close()
        write_json(self.output / "native-smoke.json", results)
        return {"tasks": tasks, "complete": True}

    def start_omni(self):
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.tests.helpers.gr00t_calibration import CalibrationGr00tBody

        self.omni = CalibrationGr00tBody(
            instantiate(self.config.model),
            ppo=True,
            ppo_save_cfg_dir=str(self.output / "model-config"),
            transfer_dtype="bfloat16",
        )
        self.initial_omni = self.omni.inference_fingerprint()
        if self.initial_omni["publication_version"] != 0:
            raise AssertionError("baseline started with a trained publication")
        if set(self.initial_omni["parameter_dtypes"]) != {"torch.bfloat16"} or not self.initial_omni["sdpa_math"]:
            raise AssertionError("Omni did not bind the actual BF16/math-SDPA baseline")
        write_json(
            self.output / "initial-models.json",
            {
                "native_sha256": self.initial_native,
                "native_buffers": {name: str(buffer.dtype) for name, buffer in self.native.model.named_buffers()},
                "omni": self.initial_omni,
            },
        )

    def open_gateway(self, tag: str):
        from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody

        if self.gateway is not None:
            self.gateway.close()
        self.gateway = SimGatewayBody(
            sim="libero_calibration",
            sink="null",
            embodiment="libero_sim",
            sim_python=self.environment.config.sim_python,
            sim_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            sim_ipc_dir=os.environ["B1K_SIM_IPC_DIR"],
            sim_socket_path=str(Path(os.environ["B1K_SIM_IPC_DIR"]) / f"cal-{self.rank}-{tag}.sock"),
            sim_options={**self.environment.config.simulator_options, "n_lanes": 1},
            # The calibration backend (native NVIDIA views beside the posttrain ones) is test-owned;
            # the child imports it before resolving `sim`, so production recipes never see it.
            preload=("cosmos_framework.posttrain.tests.helpers.libero_calibration_sim",),
        )

    def native_action(self, observation: dict, seed: int, *, capture: bool):
        from contextlib import nullcontext

        import torch

        from cosmos_framework.posttrain.tests.helpers.gr00t_diagnostics import Gr00tForwardCapture

        scope = (
            Gr00tForwardCapture(self.native.model, self.native, "collate_fn", self.native.processor)
            if capture
            else nullcontext()
        )
        with torch.random.fork_rng(devices=[0]), scope as recorded:
            torch.manual_seed(seed)
            actions, _ = self.native.get_action(native_observation(observation))
        action = np.concatenate([actions[key] for key in self.native.modality_configs["action"].modality_keys], axis=-1)
        if action.shape != (1, 16, 7):
            raise AssertionError(f"native decoded shape differs: {action.shape}")
        return action[0, :8].copy(), recorded.data if capture else None

    def act(self, observations, *, context):
        from cosmos_framework.posttrain.tests.helpers.parity_diff import compare

        (observation,) = observations
        case, decision = context["case"], context["decision"]
        seed = libero_seed(
            "libero_10", "evaluation-policy", context["seed"], case["task_id"], case["init_state_id"], decision
        )
        if self.mode == "pilot" and decision < 4:
            native, first = self.native_action(observation, seed, capture=True)
            repeated, second = self.native_action(observation, seed, capture=True)
            ours = self.omni.capture_ode(observation, seed=seed)
            again = self.omni.capture_ode(observation, seed=seed)
            diffs = [
                compare(f"observation/{key}", observation[key], observation[f"native_{key}"]).as_json()
                for key in ("image", "wrist_image", "x", "y", "z", "roll", "pitch", "yaw", "gripper")
            ]
            diffs += capture_differences(first, second, "native-repeat")
            diffs += capture_differences(ours["capture"], again["capture"], "omni-repeat")
            diffs += capture_differences(first, ours["capture"], "native-omni")
            diffs += [
                compare("executed-prefix", native, ours["action"]).as_json(),
                compare("native-repeat-action", native, repeated).as_json(),
            ]
            path = self.output / "pilot" / f"task-{case['task_id']}-decision-{decision}"
            write_json(path.with_suffix(".json"), {"case": case, "decision": decision, "seed": seed, "diffs": diffs})
            if any(not row["bitwise_equal"] for row in diffs):
                import torch

                torch.save({"observation": observation, "native": first, "omni": ours}, path.with_suffix(".pt"))
                mismatch = next(row for row in diffs if not row["bitwise_equal"])
                raise AssertionError(f"strict pilot mismatch: {mismatch}")
            if decision == 0:
                legacy = self.omni.capture_ode(observation, seed=seed, sampling_mode="ode")
                negative = capture_differences(first, legacy["capture"], "legacy-ode")
                write_json(path.with_name(path.name + "-negative.json"), negative)
                if not any(not row["bitwise_equal"] for row in negative):
                    raise AssertionError("legacy ODE negative control failed to detect the old difference")
            return [native]
        if self.mode in ("native", "pilot"):
            action, _ = self.native_action(observation, seed, capture=False)
        else:
            action = self.omni.infer_batch(observations, seed=seed, weight_version=0, sampling_mode="native_ode")[0][
                "action"
            ]
        self.decision_trace.append(
            {
                "decision": decision,
                "seed": seed,
                "observation_sha256": {
                    key: hashlib.sha256(observation[key].tobytes()).hexdigest()
                    for key in ("raw_image", "raw_wrist_image", "raw_eef_pos", "raw_eef_quat", "raw_gripper")
                },
                "action": action.tolist(),
            }
        )
        return [action]

    def run_cases(self, mode: str, cases: list[dict]) -> list[dict]:
        self.mode = mode
        self.open_gateway(mode)
        rows = []
        for case in cases:
            if self.watch.should_stop():
                raise InterruptedError("common baseline stopped at an episode boundary; completed rows retained")
            self.decision_trace = []
            result = self.gateway.run_evaluation(policy=self, cases=[case], version=0, seed=0, decisions=90)
            if result["policy_version"] != 0 or not result["training_session_unchanged"]:
                raise AssertionError("baseline evaluation changed simulator or publication state")
            (row,) = result["episodes"]
            row["decisions"] = self.decision_trace
            rows.append(row)
            write_json(self.output / mode / f"task-{case['task_id']}-state-{case['init_state_id']}.json", row)
            print(
                f"BASELINE {mode} rank={self.rank} case={case} success={row['success']} steps={row['steps']}",
                flush=True,
            )
        if self.native_fingerprint() != self.initial_native or self.omni.inference_fingerprint() != self.initial_omni:
            raise AssertionError("zero-update evaluation changed model weights or publication version")
        write_json(
            self.output / f"{mode}-unchanged.json",
            {"native": self.initial_native, "omni": self.initial_omni, "optimizer_step": 0, "publication_version": 0},
        )
        return rows

    def close(self):
        try:
            if self.gateway is not None:
                self.gateway.close()
        finally:
            if self.omni is not None:
                self.omni.close()


def export_results(output: Path, native: list[dict], omni: list[dict]):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from cosmos_framework.posttrain.tests.helpers.libero_analysis import episode_metrics, paired_comparison

    manifest = fixed_manifest("eval", "libero_10", ALIGNED_EVAL_PROFILE)
    report = {
        "native": episode_metrics(native, manifest),
        "omni": episode_metrics(omni, manifest),
        **paired_comparison(native, omni),
    }
    pairs = {(row["task_id"], row["init_state_id"]): row for row in native}
    report["first_divergences"] = []
    for row in omni:
        before = pairs[row["task_id"], row["init_state_id"]]
        first = next((index for index, (a, b) in enumerate(zip(before["decisions"], row["decisions"])) if a != b), None)
        if first is None and len(before["decisions"]) != len(row["decisions"]):
            first = min(len(before["decisions"]), len(row["decisions"]))
        if first is not None or before["control_rewards"] != row["control_rewards"]:
            report["first_divergences"].append(
                {"task_id": row["task_id"], "init_state_id": row["init_state_id"], "decision": first}
            )
    report["paired_gains"] = [
        row for row in omni if row["success"] and not pairs[row["task_id"], row["init_state_id"]]["success"]
    ]
    report["paired_losses"] = [
        row for row in omni if not row["success"] and pairs[row["task_id"], row["init_state_id"]]["success"]
    ]
    report.update(
        optimizer_step=0, publication_version=0, evaluation_distribution="training-pool", profile=ALIGNED_EVAL_PROFILE
    )
    write_json(output / "comparison.json", report)
    with (output / "episodes.jsonl").open("w") as stream, (output / "episodes.csv").open("w") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["policy", "task_id", "init_state_id", "success", "steps", "return"]
        )
        writer.writeheader()
        for name, rows in (("native", native), ("omni", omni)):
            for row in rows:
                stream.write(json.dumps({"policy": name, **row}) + "\n")
                writer.writerow({key: name if key == "policy" else row[key] for key in writer.fieldnames})
    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    for offset, name in ((-0.2, "native"), (0.2, "omni")):
        axis.bar(
            np.arange(10) + offset,
            [report[name][f"val/task_{task}/success_rate"] for task in range(10)],
            width=0.4,
            label=name,
        )
    axis.set(ylim=(0, 1.05), xlabel="LIBERO-10 task", ylabel="Success rate", xticks=range(10))
    axis.legend()
    fig.savefig(output / "per-task-success.png")
    (output / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>LIBERO-10 zero-update calibration</h1><p>200 identical cases per policy; training-distribution evaluation; no optimizer updates.</p><img src="per-task-success.png"><p><a href="comparison.json">Paired comparison</a> · <a href="episodes.csv">CSV</a> · <a href="episodes.jsonl">JSONL</a></p>'
    )
    return report


def main():
    import ray

    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-only", action="store_true", help="Check ten four-decision pilots without full episodes"
    )
    args, options = parser.parse_known_args()
    config = load_config(args.config, ["--", *[value for value in options if value != "--"]])
    recipe = instantiate(config.rl)
    validate_baseline(config)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "identity.json",
        {
            "recipe": recipe.identity(),
            "checkpoint": config.model.checkpoint_path,
            "batch_size": 1,
            "optimizer_step": 0,
            "publication_version": 0,
            "sources": provenance(config),
        },
    )
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    width = min(8, int(ray.cluster_resources()["GPU"]))
    worker = ray.remote(num_gpus=1, num_cpus=2)(CalibrationWorker)
    workers = [worker.remote(config, recipe, str(args.output / f"rank-{rank}"), rank) for rank in range(width)]
    try:
        if not args.pilot_only:
            ray.get([actor.native_smoke.remote(list(range(rank, 10, width))) for rank, actor in enumerate(workers)])
        ray.get([actor.start_omni.remote() for actor in workers])
        pilots = [asdict(LiberoCase(task, 0, "libero_10")) for task in range(10)]
        ray.get([actor.run_cases.remote("pilot", pilots[rank::width]) for rank, actor in enumerate(workers)])
        write_json(args.output / "pilot-passed.json", {"tasks": 10, "decisions_per_task": 4, "bitwise_equal": True})
        if args.pilot_only:
            return
        cases = [asdict(case) for case in fixed_manifest("eval", "libero_10", ALIGNED_EVAL_PROFILE).cases]
        results = {}
        for name in ("native", "omni"):
            shards = ray.get([actor.run_cases.remote(name, cases[rank::width]) for rank, actor in enumerate(workers)])
            results[name] = [row for shard in shards for row in shard]
        report = export_results(args.output, **results)
        from cosmos_framework.posttrain.utils.report import (
            emit_metric_frame,
            finish_best_effort,
            make_reporters,
            start_best_effort,
        )

        reporters = make_reporters(
            config,
            str(args.output),
        )
        failed = set()
        try:
            start_best_effort(reporters, failed_reporters=failed)
            emit_metric_frame(
                reporters,
                {
                    "val/native_success_rate": report["native"]["val/success_rate"],
                    "val/omni_success_rate": report["omni"]["val/success_rate"],
                    "val/success_rate_delta": report["success_rate_delta"],
                    "rollout/publication_version": 0,
                },
                0,
                failed_reporters=failed,
            )
        finally:
            finish_best_effort(reporters, failed_reporters=failed)
        print(json.dumps(report), flush=True)
    finally:
        try:
            ray.get([actor.close.remote() for actor in workers])
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
