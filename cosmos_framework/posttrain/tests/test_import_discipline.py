# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Gate G4 -- ``embodied/`` imports stdlib and numpy, and nothing else.

That property is what lets one rollout core load in the Isaac venv (py3.11 + Isaac 5.1, no
torch, no i4), the policy venv (torch + i4, no omnigibson) and the trainer venv alike. It is
also the whole answer to "why is ``embodied/`` a top-level package instead of a subtree of
``roles/``": ``roles/`` is the framework's role taxonomy and this is a library two venvs
share. The acute symptom was ``roles/env/__init__.py`` eagerly re-exporting ``wfm``, which
imports torch at module scope, so anything under ``roles.env`` dragged the training stack in.
That barrel is lazy now and ``test_the_roles_env_barrel_is_lazy`` holds it there -- but a
one-line regression can undo a lazy import, and cannot undo a directory boundary.

It costs nothing to maintain and everything to lose: the first convenient
``from ..roles.rollout.cosmos3 import ...`` in the core makes it unimportable in the venv that
runs the simulator, and the failure surfaces as a Ray actor that dies at bind time with an
ImportError from three packages away.

Measured as a **delta over a numpy-only interpreter**, not as an absolute module set. The
absolute set picks up whatever the ambient environment injects -- ``sitecustomize``,
``cython_runtime``, a conda shim -- none of which the core imported, and any of which would
make this gate fail for reasons unrelated to the code under test. The delta is the quantity
the gate cares about: *what does importing the core add?*

It runs in a subprocess. This test process has already imported pytest, numpy and much else,
so ``sys.modules`` here says nothing about what a fresh interpreter would pull in.

The gate WALKS ``embodied/`` rather than naming its modules, because a written-out list
passes forever the moment someone adds a file -- which is exactly when it should fire. The
walk itself lives in ``embodied/plugin_surface.py``, not here, so that the thing being
enforced and the thing doing the enforcing are two statements rather than one agreeing with
itself.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cosmos_framework.posttrain.embodied import plugin_surface

#: Repo root -- four levels up from cosmos_framework/posttrain/tests/.
_REPO = Path(__file__).resolve().parents[4]

_PKG = "cosmos_framework.posttrain"

_EMBODIED = _REPO / "cosmos_framework/posttrain/embodied"

#: WALKED, never written out -- and declared by the package under test rather than here, so
#: the thing being enforced and the thing doing the enforcing are two statements. A gate that
#: declares its own expectation agrees with itself by construction.
CORE = plugin_surface.core_modules()

_DELTA_PROBE = textwrap.dedent(
    """
    import json, sys

    import numpy  # the one third-party dependency the core is allowed


    def tops():
        stdlib = set(sys.stdlib_module_names)
        return {{m.split(".")[0] for m in sys.modules}} - stdlib


    before = tops()
    {body}
    print(json.dumps(sorted(tops() - before)))
    """
)


def _delta(body: str) -> list[str]:
    """Top-level non-stdlib packages that ``body`` adds to a numpy-only interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", _DELTA_PROBE.format(body=body)],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _backend_modules(root: Path) -> list[str]:
    modules = set()
    for sub in plugin_surface.SUBPACKAGES:
        for path in (root / sub).rglob("*.py"):
            relative = path.relative_to(root).with_suffix("")
            if relative.name == "__init__":
                relative = relative.parent
            modules.add(".".join(relative.parts))
    return sorted(modules)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_core_set_is_derived_from_the_directory_and_not_written_out():
    """NEGATIVE CONTROL ON THE WALK, and it is not redundant with the gate.

    Every test here consumes ``CORE``; none notices if it goes back to being a hand-written
    tuple. Reverting ``core_modules()`` to a literal leaves this whole file green, and the
    silent-escape defect returns intact -- a module added under ``embodied/`` escapes every
    gate in this file until someone happens to compare the list against ``ls``.

    The glob is recomputed here rather than imported, so the declaration and the check are two
    independent statements about the directory.
    """
    walked = set(CORE)
    on_disk = {p.stem for p in _EMBODIED.glob("*.py")} - {"__init__"}

    assert walked, f"no modules found under {_EMBODIED} -- the glob is wrong, not the tree"
    assert walked == on_disk, (
        f"the core set no longer matches the directory: missing {sorted(on_disk - walked)}, "
        f"stale {sorted(walked - on_disk)}. It must be WALKED -- a literal lets the next "
        f"module under embodied/ escape every gate in this file."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_core_adds_nothing_beyond_the_repo_itself():
    # Joined at column 0: the template is dedent()ed, so {body} lands unindented.
    body = "\n".join(f'__import__("cosmos_framework.posttrain.embodied.{name}")' for name in CORE)

    added = _delta(body)

    # "projects" and nothing else: the core imports schema/ (zero-dep, and the framework's
    # shared data contract per DESIGN 9's "Shared library" bucket) plus its own modules.
    assert added == ["projects"], (
        f"embodied/ core pulled in non-stdlib packages beyond numpy: {added}. The core must "
        f"load in the Isaac venv, the policy venv and the trainer venv alike; move the "
        f"import into a backend subpackage and reach it through the registry."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_probe_can_actually_fail():
    """Negative control.

    A delta measurement that reported ``[]`` for everything would pass the test above no
    matter what the core imported. ``config/eval.py`` is a sibling module with real
    third-party dependencies (pydantic, yaml), so it must show them.
    """
    added = set(_delta("import cosmos_framework.posttrain.config.eval"))

    assert "pydantic" in added and "yaml" in added, added


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import names in one file. Function-local imports are excluded, which is
    the whole distinction being tested."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    names: set[str] = set()
    for node in tree.body:  # tree.body only -- NOT ast.walk, which would find local imports
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


_ENTRYPOINTS = sorted((_REPO / "cosmos_framework/posttrain/entrypoints").glob("*.py"))


@pytest.mark.L0
@pytest.mark.CPU
def test_no_entrypoint_imports_ray_at_module_scope():
    """An entrypoint's pure half must be importable where there is no Ray.

    Measured, not hypothetical: CI's unit-test image has no `ray`, and `policy_eval.py`
    imported it at module scope. That took out the tests which drive the real eval
    aggregator with the real sink's output -- they failed on the import line, having never
    reached the code they exist to test. Locally they passed, because ray is installed on a
    dev host.

    Ray belongs inside `main` (and `_wait_for_cluster`), which is where a cluster is
    actually needed. Everything else in an entrypoint -- config resolution, plan kwargs, CSV
    building, the summary printers -- is pure, and a pure function that cannot be imported
    without a cluster library is untestable by construction.

    Discovered by globbing rather than listed by name: a list would pass forever the moment
    someone adds `entrypoints/whatever.py`, which is exactly when this should fire.
    """
    assert _ENTRYPOINTS, "no entrypoints found -- the glob is wrong, not the tree"
    offenders = [p.name for p in _ENTRYPOINTS if "ray" in _module_level_imports(p)]
    assert not offenders, (
        f"{offenders} import ray at module scope. Move it into the function that needs a "
        f"cluster; CI has no ray and the entrypoint's pure half must stay importable."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_module_scope_probe_can_actually_fail():
    """Positive control for the walker above.

    `tree.body` rather than `ast.walk` is the load-bearing choice, and a walker that found
    nothing would pass the negative assertion against every file forever. So: it must SEE a
    module-level import that is really there, and must NOT see a function-local one.
    """
    src = textwrap.dedent(
        """
        import os
        from collections import defaultdict

        def main():
            import ray
            return ray, os, defaultdict
        """
    )
    tmp = _REPO / "cosmos_framework/posttrain/entrypoints/__probe_module_scope.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        found = _module_level_imports(tmp)
    finally:
        tmp.unlink()

    assert "os" in found and "collections" in found, "the walker misses module-level imports"
    assert "ray" not in found, "the walker counts function-local imports, so the rule is vacuous"


@pytest.mark.L0
@pytest.mark.CPU
def test_the_plugin_surface_never_reaches_outward():
    """The outward-edge RATCHET.

    Scoped to the declared surface: the core plus ``sims/``, ``sinks/`` and ``sources/``.
    ``policies/`` is excluded because it lives in the framework venv on the other side of the
    seam, so an import outward from there is not an inversion.

    THE EXEMPTION LIST IS EMPTY and should stay that way. It is checked from BOTH ends:
    ``offenders == []`` catches one too few entries, and the staleness assertion catches one
    too many. A stale entry silently re-permits the edge it was granted for long after that
    edge is gone, and nothing about a passing suite would show it.
    """
    surface = _EMBODIED
    inside = {"embodied", "schema"}
    exempt: set[tuple[str, str]] = set()

    paths = sorted(surface.glob("*.py"))
    for sub in ("sims", "sinks", "sources"):
        paths += sorted((surface / sub).rglob("*.py"))

    offenders = []
    used = set()
    for path in paths:
        rel = str(path.relative_to(surface))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            else:
                continue
            for mod in mods:
                if not mod.startswith(f"{_PKG}."):
                    continue
                tail = mod[len(_PKG) + 1 :]
                if tail.split(".")[0] in inside:
                    continue
                if (rel, tail) in exempt:
                    used.add((rel, tail))
                    continue
                offenders.append(f"{rel}:{node.lineno} -> {tail}")

    assert (stale := sorted(exempt - used)) == [], (
        f"exemption {stale} is no longer reached. The inversion it was granted for is gone, so "
        f"the exemption now silently re-permits an edge nobody argued for. Delete it."
    )
    assert offenders == [], (
        f"the plugin surface imports the framework at {offenders}. The dependency runs "
        f"framework -> plugin and never back: a module the sim image must contain cannot "
        f"reach for one it does not. Promote the shared piece down into embodied/, or take "
        f"the framework concept out of the plugin -- do not add an exemption."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_backend_discovery_includes_nested_modules_and_package_initializers(tmp_path: Path) -> None:
    for name in ("sims/flat.py", "sims/nested/backend.py", "sources/package/__init__.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    assert _backend_modules(tmp_path) == ["sims.flat", "sims.nested.backend", "sources.package"]


@pytest.mark.L0
@pytest.mark.CPU
def test_the_backends_import_without_their_engines():
    """The per-module half of the discipline, and the half no other gate reaches.

    The registry defers WHICH file loads; this pins WHEN the engine inside that file loads.
    `the external simulator backend` and `the external native policy adapter` are exempt from the CORE rule precisely
    because they are allowed to touch omnigibson and the i4 stack -- so an author who hoists
    `import omnigibson` to module top passes every other gate here and only finds out in
    production, where `build_sim()` runs inside a Ray actor constructor: the Kit boot moves to
    bind time, it runs before `_boot()`'s `OMNIGIBSON_HEADLESS` setdefault can take effect, and
    the ImportError surfaces as "no sim named 'behavior1k'".

    Importing the MODULE must stay free. Booting the engine is `Simulator._boot()`'s job.

    WALKED over `plugin_surface.SUBPACKAGES`, not written out, for the same reason `CORE` is:
    a listed backend that someone forgets to extend is a backend that never gets checked.
    The walk found `the external simulator metric adapter` immediately, which the hand-written list this
    replaces did not name.
    """
    modules = _backend_modules(_EMBODIED)
    assert modules, "no backend modules found -- the walk is wrong, not the tree"

    #: Modules that CANNOT import free, with the structural reason. Checked from both ends
    #: below: an unused entry is as much a failure as a missing one, because a stale exemption
    #: silently re-permits an edge nobody argued for.
    #:
    #: The rule this gate enforces is about modules the REGISTRY imports. A module reached at
    #: function scope by a backend that has already booted its engine is a different case, and
    #: `BudgetedTaskMetric` is the shape that forces it: it SUBCLASSES
    #: `omnigibson.metrics.TaskMetric`, so the import cannot move into a function without the
    #: base class moving with it. It is imported from `the external simulator backend` at call time, inside
    #: the actor where omnigibson exists, and is registered with nothing.
    #: LIBERO's vector classes likewise inherit the vendor scheduler and worker. They
    #: are imported only by LiberoChunkSession._open_vector(), after simulator startup;
    #: module-level classes also keep the spawned workers' import identity stable.
    engine_bound = {
        "sims.b1k_task_metric": "subclasses omnigibson.metrics.TaskMetric",
        "sims.libero.vector": "subclasses libero.libero.envs.venv.BaseVectorEnv and SubprocEnvWorker",
    }

    free = [m for m in modules if m not in engine_bound]
    stale = sorted(set(engine_bound) - set(modules))
    assert stale == [], (
        f"exemption {stale} names a module that no longer exists. Delete it -- a stale entry "
        f"exempts a path nobody argued for."
    )
    for m, why in engine_bound.items():
        # NOT `_delta` -- that asserts the probe exits 0, and an engine-bound module is exactly
        # one whose bare import does not. Run it and require the failure.
        proc = subprocess.run(
            [sys.executable, "-c", f'__import__("cosmos_framework.posttrain.embodied.{m}")'],
            cwd=_REPO,
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0, (
            f"{m} is exempt as {why!r} but now imports free. Drop the exemption and let the walk cover it."
        )

    added = _delta("\n".join(f'__import__("cosmos_framework.posttrain.embodied.{m}")' for m in free))

    assert added == ["projects"], (
        f"a backend module imported its engine at module level: {added}. Move it inside "
        f"Simulator._boot() / the policy's load path -- the module itself must import in "
        f"any venv, or the registry's lazy __import__ buys nothing."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_roles_env_barrel_is_lazy():
    """`roles/env/__init__.py` must not pull the training stack in on import.

    This is the mechanism the whole package layout rests on, and it used to work against
    us: the barrel eagerly re-exported `wfm`, which imports torch at module scope, so
    importing ANY module under `roles.env` dragged torch, dill, pynvml, torchgen and tqdm
    into the process. `embodied/` lives at top level specifically to escape that -- and yet
    `collector.py`, `sim_gateway.py` and `sim_server.py` all live under `roles.env`, with
    `sim_server` being the process the SIMULATOR container starts.

    Measured before the fix: importing `roles.env.base` added ten packages. After: one.
    The re-exports are resolved by a module-level `__getattr__`, so callers are unaffected;
    only the import-time cost moves to first access.

    Asserted as a DELTA over a numpy-only interpreter, like every other gate here, so an
    ambient `sitecustomize` cannot make it fail for an unrelated reason.
    """
    added = _delta(f"import {_PKG}.roles.env.base")
    assert added == ["projects"], (
        f"importing roles.env.base adds {added}. Something in roles/env/__init__.py is "
        f"imported EAGERLY again -- add it to `_EXPORTS` and let `__getattr__` resolve it, "
        f"or the simulator container carries the training stack."
    )


_RAY_BLOCKED_PROBE = textwrap.dedent(
    """
    import importlib, sys


    class _NoRay:
        \"\"\"Refuse `ray` the way an interpreter without it would.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "ray" or name.startswith("ray."):
                raise ImportError("ray is not installed in the simulator venv")
            return None


    sys.meta_path.insert(0, _NoRay())
    importlib.import_module("{mod}")
    print("OK")
    """
)


def _import_without_ray(mod: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _RAY_BLOCKED_PROBE.format(mod=mod)],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_sim_side_roots_import_without_ray():
    """The claim the whole two-venv split rests on, measured.

    The sim venv carries Ray, and a python pinned to the framework image at PATCH level, for
    exactly one reason: under ``sim_transport: direct`` the simulator's raylet joins the GCS,
    and Ray compares python versions patch-for-patch before it will let a node in. Under
    ``gateway`` there is no sim raylet -- the gateway holds the actor and speaks to the sim
    over a UNIX socket -- so the pin has nothing left to buy.

    Deleting it from ``docker/Dockerfile.behavior1k`` is somebody's future commit, and the
    only thing that makes that commit safe to write is this gate. Without it "the sim side
    does not need Ray" is a design intention: true when written, quietly false the first time
    a convenient ``import ray`` lands in a module the sim entrypoint reaches.

    **The claim is Ray, not torch.** torch IS in this closure and belongs there -- omnigibson
    needs it and the image asserts ``SIM_TORCH``. Asserting its absence would fail honestly
    and for the wrong reason.

    Roots come from ``plugin_surface.sim_side_roots()`` rather than a literal here, so the
    declaration and the check are two statements rather than one agreeing with itself.
    """
    roots = plugin_surface.sim_side_roots()
    assert roots, "sim_side_roots() is empty -- the gate would pass vacuously"
    for mod in roots:
        proc = _import_without_ray(mod)
        assert proc.returncode == 0, (
            f"{mod} cannot be imported without ray:\n{proc.stderr}\n"
            f"It is a sim-side root, so it runs in the Isaac venv. Move the import into the "
            f"function that needs a cluster, or take the module off sim_side_roots()."
        )
        # Ray was necessary, not sufficient. A root that imports cleanly but drags the whole
        # training stack still defeats the split: the point of the sim image is that it does
        # not carry the framework. Measured as the same delta the core is held to.
        added = _delta(f"import {mod}")
        assert added == ["projects"], (
            f"{mod} is a sim-side root but importing it adds {added}. The sim container is "
            f"supposed to need `embodied/` + `schema/` and nothing else; check whether a "
            f"package __init__ on the path re-exports something heavy."
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_ray_blocker_can_actually_fail():
    """Positive control for the gate above.

    A meta-path finder is easy to write in a way that never fires -- an earlier version of
    this probe used the 3.12-removed ``find_module``, so it blocked nothing and BOTH trees
    passed. The gate is worth exactly what this control is worth: the blocker must refuse a
    module that really does import ray at module scope.
    """
    probe = _EMBODIED.parent / "roles/env/__probe_ray_import.py"
    probe.write_text("import ray\n", encoding="utf-8")
    try:
        proc = _import_without_ray(f"{_PKG}.roles.env.__probe_ray_import")
    finally:
        probe.unlink()

    assert proc.returncode != 0, "the ray blocker does not block ray -- the gate is vacuous"
    assert "ray is not installed" in proc.stderr, (
        f"the probe failed for some reason other than the blocker:\n{proc.stderr}"
    )


#: Modules these docstrings cite that live on an archive branch rather than in this
#: series. DECLARED, so that a citation of something genuinely misspelled is still caught.
#: Every entry must be cited by something, or it is stale and the gate says so.
_ON_ARCHIVE_BRANCH = {
    "entrypoints/collect.py": "the collection entrypoint; the fleet plane is not in this series",
    "sinks/package.py": "the frame-writing sink; only sinks/null.py is in this series",
    "sinks/evaljsonl.py": "the eval sink, selected by a policy_eval.py line not in this tree",
    "campaign/dirlease.py": "the work-claim plane; ~13k lines of fleet/campaign stay on the archive branch",
    "config/collect.py": "the collection config; the V=1 refusal here mirrors its escape hatch",
}

#: Top-level posttrain subdirectories. A citation is treated as an IN-REPO pointer -- the kind
#: a reviewer would try to follow -- only when it starts with one of these. Everything else is
#: a reference to another codebase (OmniGibson's ``simulator.py``, ``eval_vec.py`` ...) and is
#: none of this gate's business; policing those would make the gate noisy and it would be
#: turned off, which is how a useful gate dies.
_IN_REPO_ROOTS = (
    "embodied/",
    "roles/",
    "control/",
    "entrypoints/",
    "tests/",
    "campaign/",
    "schema/",
    "config/",
    "utils/",
    "workers/",
    "docs/",
    "sinks/",
    "sims/",
    "sources/",
)

_CITATION = re.compile(r"\b((?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py)(?::(\d+))?")


def _cited_paths():
    """Every `foo/bar.py` and `foo/bar.py:123` mentioned in the packages this series owns."""
    pt = _EMBODIED.parent
    #: `embodied/` is entirely new in this series, so all of it is in scope. Under `roles/env/`
    #: and `control/` only the modules this series ADDS are -- policing `the external simulator environment body`
    #: would report main's own citations as this series' defects.
    owned = sorted(_EMBODIED.rglob("*.py"))
    for rel in (
        "roles/env/sim_role.py",
        "roles/env/collector.py",
        "roles/env/sim_gateway.py",
        "roles/env/sim_server.py",
        "control/recipe.py",
    ):
        if (pt / rel).exists():
            owned.append(pt / rel)
    for path in owned:
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for m in _CITATION.finditer(line):
                cited = m.group(1)
                if not cited.startswith(_IN_REPO_ROOTS):
                    continue
                yield path, lineno, cited, m.group(2)


@pytest.mark.L0
@pytest.mark.CPU
def test_no_docstring_cites_a_file_or_line_that_does_not_exist():
    """A pointer a reviewer cannot follow is worse than no pointer.

    These docstrings were written on an archive branch, where `sinks/package.py`, `campaign/dirlease.py`
    and `entrypoints/collect.py` all exist. Carried into a series where they do not, they send
    a reader hunting for a file that was never in the diff.

    Two failures this caught, and they are the reason this is a gate and not a cleanup:

    * `control/recipe.py` claimed ``tests/test_control_recipe_sim.py`` "pins" the `bind_check`
      vacuity. That test does not exist and **no test in the tree touches `sim_manifest`** --
      a claim of coverage with nothing behind it, which is exactly the assumption the
      paragraph existed to prevent.
    * `sinks/null.py` cited ``the external policy-evaluation entrypoint`` for where `evaljsonl` is
      selected. That file has 692 lines here and does not mention `evaljsonl` at all.

    Line numbers are checked too, because a stale one is the same defect wearing a precise
    number: it looks MORE authoritative than a bare filename while pointing at nothing.
    """
    repo_rel = _EMBODIED.parent
    bad_file, bad_line, used = [], [], set()

    for src, lineno, cited, cited_ln in _cited_paths():
        if cited in _ON_ARCHIVE_BRANCH:
            used.add(cited)
            continue
        hits = [c for c in (repo_rel / cited, _REPO / cited) if c.exists()]
        if not hits:
            hits = list(repo_rel.rglob(Path(cited).name))
        rel = src.relative_to(repo_rel)
        if not hits:
            bad_file.append(f"{rel}:{lineno} -> {cited}")
        elif cited_ln and int(cited_ln) > len(hits[0].read_text(errors="ignore").splitlines()):
            n = len(hits[0].read_text(errors="ignore").splitlines())
            bad_line.append(f"{rel}:{lineno} -> {cited}:{cited_ln} (file has {n} lines)")

    assert not bad_file, (
        "docstrings cite files that do not exist:\n  "
        + "\n  ".join(sorted(bad_file))
        + "\nEither fix the path, or add it to _ON_ARCHIVE_BRANCH with the reason it is absent."
    )
    assert not bad_line, (
        "docstrings cite line numbers past end-of-file:\n  "
        + "\n  ".join(sorted(bad_line))
        + "\nA stale line number looks more authoritative than a bare filename."
    )
    # BOTH ENDS: an entry nothing cites is stale, and would silently permit a real typo.
    stale = sorted(set(_ON_ARCHIVE_BRANCH) - used)
    assert not stale, (
        f"_ON_ARCHIVE_BRANCH names {stale}, which nothing cites any more. Drop the entries -- "
        f"a stale allowance re-permits a misspelling of that exact path forever."
    )
