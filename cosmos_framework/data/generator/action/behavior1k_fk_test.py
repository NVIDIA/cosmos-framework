# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Portable real-MuJoCo regression for provisioned URDF mesh directories."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from cosmos_framework.data.generator.action.behavior1k_fk import _compile_r1pro_urdf

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]

_MESH = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""
_URDF = """<robot name="relative_mesh_regression">{extension}
<link name="base"/>
<link name="arm">
<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><mesh filename="{filename}"/></geometry></collision>
</link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="0.2 0.3 0.4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
</joint></robot>"""


@pytest.mark.parametrize("explicit_compiler", (False, True))
def test_relative_meshes_preserve_fk_and_source_bytes(tmp_path: Path, explicit_compiler: bool) -> None:
    folder = tmp_path / "meshes"
    folder.mkdir()
    (folder / "shape.obj").write_text(_MESH)
    filename = "shape.obj" if explicit_compiler else "meshes/shape.obj"
    extension = '<mujoco><compiler meshdir="meshes" strippath="true"/></mujoco>' if explicit_compiler else ""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_URDF.format(filename=filename, extension=extension))
    original = urdf.read_bytes()
    actual = _compile_r1pro_urdf(urdf)
    # Original XML + exact OBJ bytes uses the original compiler defaults.
    expected = mujoco.MjModel.from_xml_path(str(urdf), assets={"shape.obj": _MESH.encode()})
    assert actual.nmesh == expected.nmesh == 1
    for angle in (0.0, 0.25, -0.5):
        a, b = mujoco.MjData(actual), mujoco.MjData(expected)
        a.qpos[:] = b.qpos[:] = angle
        mujoco.mj_forward(actual, a)
        mujoco.mj_forward(expected, b)
        np.testing.assert_allclose(a.xpos, b.xpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.xquat, b.xquat, rtol=0, atol=1e-12)
    assert urdf.read_bytes() == original
