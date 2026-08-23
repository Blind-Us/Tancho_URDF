#!/usr/bin/env python3
"""Validate the generated Tancho v3 asset and its STEP-derived anchors."""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
URDF = ROOT / "Tancho_v3.urdf"
REPORT = ROOT / "mass_properties.json"
STEP = Path("/media/azul/861896C11896B023/Tancho/Tancho_simplified.step")
BASE_TO_CAD_MM = np.array([10.0, -0.55265617, 0.0])

FAILS: list[str] = []


def cad_modules():
    try:
        import FreeCAD  # type: ignore
        import Import  # type: ignore
    except ModuleNotFoundError:
        sys.path.insert(0, "/home/azul/miniconda3/envs/freecad-urdf/lib")
        import FreeCAD  # type: ignore
        import Import  # type: ignore
    return FreeCAD, Import


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def arr(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]], dtype=float,
    )


def step_global_objects():
    FreeCAD, Import = cad_modules()
    doc = FreeCAD.newDocument("validate_tancho_v3")
    Import.open(str(STEP), doc.Name)
    objects = {}
    for obj in doc.Objects:
        if not hasattr(obj, "Shape") or obj.Shape.isNull():
            continue
        shape = obj.Shape.copy()
        shape.Placement = obj.getGlobalPlacement()
        objects[obj.Label] = shape
    return FreeCAD, doc, objects


def main() -> int:
    root = ET.parse(URDF).getroot()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    check("URDF exists and robot name is Tancho_v3", root.get("name") == "Tancho_v3")
    link_names = {link.get("name") for link in root.findall("link")}
    required_links = {"base_link_root", "base_link", "drawer", "pi_case", "thigh_L", "calf_L", "wheel_L", "thigh_R", "calf_R", "wheel_R", "imu_link"}
    check("required links present", required_links <= link_names)
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    required_joints = {"base_link_rotation_joint", "joint_drawer_R", "joint_pi_case_R", "joint_thigh_L", "joint_calf_L", "joint_wheel_L", "joint_thigh_R", "joint_calf_R", "joint_wheel_R", "imu_joint"}
    check("required joints present", required_joints <= set(joints))

    files_ok = True
    origins_ok = True
    for link in root.findall("link"):
        for element in link.findall("visual") + link.findall("collision"):
            origin = element.find("origin")
            origins_ok &= origin is not None and origin.get("xyz") == "0 0 0" and origin.get("rpy") == "0 0 0"
            mesh = element.find("geometry/mesh")
            if mesh is None:
                files_ok = False
                continue
            path = ROOT / mesh.get("filename")
            files_ok &= path.exists() and path.stat().st_size > 1024
            if path.exists():
                loaded = trimesh.load(path, force="mesh", process=False)
                files_ok &= bool(np.isfinite(loaded.vertices).all()) and float(np.max(loaded.extents)) < 1.0
    check("all visual/collision origins are zero", origins_ok)
    check("all meshes exist, are finite, and use metres", files_ok)

    inertia_ok = True
    for name in ("base_link", "drawer", "pi_case", "thigh_L", "calf_L", "wheel_L", "thigh_R", "calf_R", "wheel_R"):
        link = root.find(f"link[@name='{name}']")
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        ie = inertial.find("inertia")
        matrix = np.array([[float(ie.get("ixx")), float(ie.get("ixy")), float(ie.get("ixz"))],
                           [float(ie.get("ixy")), float(ie.get("iyy")), float(ie.get("iyz"))],
                           [float(ie.get("ixz")), float(ie.get("iyz")), float(ie.get("izz"))]])
        inertia_ok &= mass > 0.0 and np.linalg.eigvalsh(matrix).min() > 0.0
    check("all physical link masses positive and inertias positive-definite", inertia_ok)

    expected_masses = {
        "base_link": 0.348,
        "drawer": 0.400,
        "pi_case": 0.146,
        "thigh_L": 0.379,
        "thigh_R": 0.379,
        "calf_L": 0.341,
        "calf_R": 0.341,
        "wheel_L": 0.152,
        "wheel_R": 0.152,
    }
    masses_ok = all(
        np.isclose(float(root.find(f"link[@name='{name}']/inertial/mass").get("value")), mass, atol=1e-9)
        for name, mass in expected_masses.items()
    )
    check("measured assembly masses are written without double-counting", masses_ok)

    expected_meshes = {
        "base_link.stl", "base_link001.stl", "base_link002.stl", "drawer.stl", "pi_case.stl",
        "joint_thigh_L.stl", "thigh_L.stl", "knee_cover_L.stl", "joint_knee_L.stl", "calf_L.stl",
        "joint_wheel_L.stl", "wheel_L.stl", "wheel_L001.stl", "joint_thigh_R.stl", "thigh_R.stl",
        "knee_cover_R.stl", "joint_knee_R.stl", "calf_R.stl", "joint_wheel_R.stl", "wheel_R.stl", "wheel_R001.stl",
    }
    actual_meshes = {path.name for path in (ROOT / "meshes").glob("*.stl")}
    check("STL files use the Tancho v2 semantic naming convention", actual_meshes == expected_meshes)

    def joint_origin(name):
        return arr(joints[name].find("origin").get("xyz"))

    check("hip-to-knee Y separation is 0.15 m", np.isclose(abs(joint_origin("joint_calf_L")[1]), 0.15, atol=1e-5) and np.isclose(abs(joint_origin("joint_calf_R")[1]), 0.15, atol=1e-5))
    check("knee-to-wheel Y separation is 0.10 m", np.isclose(abs(joint_origin("joint_wheel_L")[1]), 0.09999754, atol=2e-5) and np.isclose(abs(joint_origin("joint_wheel_R")[1]), 0.09999754, atol=2e-5))

    FreeCAD, doc, objects = step_global_objects()
    connector_map = {
        "thigh_L": "Simplified Primitive (3)", "calf_L": "Simplified Primitive (4)", "wheel_L": "Simplified Primitive (5)",
        "thigh_R": "Simplified Primitive", "calf_R": "Simplified Primitive (1)", "wheel_R": "Simplified Primitive (2)",
    }
    urdf_anchor = {
        "base_link": BASE_TO_CAD_MM,
        "thigh_L": BASE_TO_CAD_MM + joint_origin("joint_thigh_L") * 1000.0,
        "calf_L": BASE_TO_CAD_MM + joint_origin("joint_thigh_L") * 1000.0 + joint_origin("joint_calf_L") * 1000.0,
        "wheel_L": BASE_TO_CAD_MM + joint_origin("joint_thigh_L") * 1000.0 + joint_origin("joint_calf_L") * 1000.0 + joint_origin("joint_wheel_L") * 1000.0,
        "thigh_R": BASE_TO_CAD_MM + joint_origin("joint_thigh_R") * 1000.0,
        "calf_R": BASE_TO_CAD_MM + joint_origin("joint_thigh_R") * 1000.0 + joint_origin("joint_calf_R") * 1000.0,
        "wheel_R": BASE_TO_CAD_MM + joint_origin("joint_thigh_R") * 1000.0 + joint_origin("joint_calf_R") * 1000.0 + joint_origin("joint_wheel_R") * 1000.0,
    }
    anchors_ok = True
    print("\n  STEP connector center anchors (mm):")
    for link, body in connector_map.items():
        center = np.array([objects[body].CenterOfMass.x, objects[body].CenterOfMass.y, objects[body].CenterOfMass.z])
        error = float(np.linalg.norm(center - urdf_anchor[link]))
        ok = error < 0.02
        anchors_ok &= ok
        print(f"    {link:9s} URDF={np.round(urdf_anchor[link], 5)} STEP={np.round(center, 5)} error={error:.6f} mm {'OK' if ok else 'OFF'}")
    check("URDF joint anchors match STEP connector centers (<0.02 mm)", anchors_ok)

    main_body = objects["Main_body"].BoundBox
    main_body002 = objects["Main_body002"].BoundBox
    box_a = np.array([(main_body.XMin + main_body.XMax) * 0.5, (main_body.YMin + main_body.YMax) * 0.5, (main_body.ZMin + main_body.ZMax) * 0.5])
    box_b = np.array([(main_body002.XMin + main_body002.XMax) * 0.5, (main_body002.YMin + main_body002.YMax) * 0.5, (main_body002.ZMin + main_body002.ZMax) * 0.5])
    imu_cad = (box_a + box_b) * 0.5 + np.array([0.0, -30.0, 0.0])
    imu_joint = joint_origin("imu_joint") * 1000.0
    expected_imu_base = imu_cad - BASE_TO_CAD_MM
    check("IMU is 30 mm below Main_body/Main_body002 bbox midpoint", np.linalg.norm(imu_joint - expected_imu_base) < 0.02, f"CAD={np.round(imu_cad, 5)} mm base={np.round(imu_joint, 5)} mm")
    check("imu_joint is fixed base_link->imu_link", joints["imu_joint"].get("type") == "fixed" and joints["imu_joint"].find("parent").get("link") == "base_link" and joints["imu_joint"].find("child").get("link") == "imu_link")

    FreeCAD.closeDocument(doc.Name)
    print()
    if FAILS:
        print("RESULT: FAIL ->", FAILS)
        return 1
    print("RESULT: ALL TANCHO_V3 ASSET CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
