#!/usr/bin/env python3
"""Convert the Tancho v3 STEP assembly to a local-frame URDF asset."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _load_cad_modules():
    try:
        import FreeCAD  # type: ignore
        import Import  # type: ignore
        return FreeCAD, Import
    except ModuleNotFoundError:
        cad_lib = "/home/azul/miniconda3/envs/freecad-urdf/lib"
        if cad_lib not in sys.path:
            sys.path.insert(0, cad_lib)
        import FreeCAD  # type: ignore
        import Import  # type: ignore
        return FreeCAD, Import


import trimesh


ROOT = Path(__file__).resolve().parent
DEFAULT_STEP = Path("/media/azul/861896C11896B023/Tancho/Tancho_simplified.step")
MESH_DIR = ROOT / "meshes"
URDF_PATH = ROOT / "Tancho_v3.urdf"
REPORT_PATH = ROOT / "mass_properties.json"
README_PATH = ROOT / "README.md"

DENSITY_KG_M3 = 1240.0
LINEAR_DEFLECTION_MM = 0.08
BASE_TO_CAD_MM = np.array([10.0, -0.55265617, 0.0], dtype=float)
ROOT_RPY = np.array([math.pi / 2.0, 0.0, 0.0], dtype=float)
WHEEL_RPY = np.array([math.pi, 0.0, 0.0], dtype=float)

LINK_ORDER = ["base_link", "drawer", "pi_case", "thigh_L", "calf_L", "wheel_L", "thigh_R", "calf_R", "wheel_R"]
LINK_BODIES = {
    "base_link": ["Main_body", "Main_body001", "Main_body002"],
    "drawer": ["Drawer"],
    "pi_case": ["Pi Case"],
    "thigh_L": ["Hip(Mirror)", "Knee cover(Mirror)", "Simplified Primitive (3)"],
    "calf_L": ["Calf(Mirror)", "Simplified Primitive (4)"],
    "wheel_L": ["Wheel(Mirror)", "Wheel(Mirror)001", "Simplified Primitive (5)"],
    "thigh_R": ["Hip", "Knee cover", "Simplified Primitive"],
    "calf_R": ["Calf", "Simplified Primitive (1)"],
    "wheel_R": ["Wheel", "Wheel001", "Simplified Primitive (2)"],
}
MESH_GROUPS = {
    "base_link": [(["Main_body"], "base_link.stl"), (["Main_body001"], "base_link001.stl"), (["Main_body002"], "base_link002.stl")],
    "drawer": [(["Drawer"], "drawer.stl")],
    "pi_case": [(["Pi Case"], "pi_case.stl")],
    "thigh_L": [(["Simplified Primitive (3)"], "joint_thigh_L.stl"), (["Hip(Mirror)"], "thigh_L.stl"), (["Knee cover(Mirror)"], "knee_cover_L.stl")],
    "calf_L": [(["Simplified Primitive (4)"], "joint_knee_L.stl"), (["Calf(Mirror)"], "calf_L.stl")],
    "wheel_L": [(["Simplified Primitive (5)"], "joint_wheel_L.stl"), (["Wheel(Mirror)"], "wheel_L.stl"), (["Wheel(Mirror)001"], "wheel_L001.stl")],
    "thigh_R": [(["Simplified Primitive"], "joint_thigh_R.stl"), (["Hip"], "thigh_R.stl"), (["Knee cover"], "knee_cover_R.stl")],
    "calf_R": [(["Simplified Primitive (1)"], "joint_knee_R.stl"), (["Calf"], "calf_R.stl")],
    "wheel_R": [(["Simplified Primitive (2)"], "joint_wheel_R.stl"), (["Wheel"], "wheel_R.stl"), (["Wheel001"], "wheel_R001.stl")],
}
MASS_GROUPS_KG = {
    "base_link": [(["Main_body", "Main_body001", "Main_body002"], 0.348)],
    "drawer": [(["Drawer"], 0.400)],
    "pi_case": [(["Pi Case"], 0.146)],
    "thigh_L": [(["Hip(Mirror)", "Knee cover(Mirror)"], 0.079), (["Simplified Primitive (3)"], 0.300)],
    "calf_L": [(["Calf(Mirror)"], 0.041), (["Simplified Primitive (4)"], 0.300)],
    "wheel_L": [(["Wheel(Mirror)", "Wheel(Mirror)001"], 0.036), (["Simplified Primitive (5)"], 0.116)],
    "thigh_R": [(["Hip", "Knee cover"], 0.079), (["Simplified Primitive"], 0.300)],
    "calf_R": [(["Calf"], 0.041), (["Simplified Primitive (1)"], 0.300)],
    "wheel_R": [(["Wheel", "Wheel001"], 0.036), (["Simplified Primitive (2)"], 0.116)],
}
ANCHOR_BODIES = {
    "thigh_L": "Simplified Primitive (3)",
    "calf_L": "Simplified Primitive (4)",
    "wheel_L": "Simplified Primitive (5)",
    "thigh_R": "Simplified Primitive",
    "calf_R": "Simplified Primitive (1)",
    "wheel_R": "Simplified Primitive (2)",
}


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def v3(value: np.ndarray) -> str:
    return " ".join(f"{float(x):.9g}" for x in value)


def vec_from_freecad(value) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)


def shape_global(obj):
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


def matrix3(matrix) -> np.ndarray:
    return np.array([[float(getattr(matrix, f"A{i + 1}{j + 1}")) for j in range(3)] for i in range(3)], dtype=float)


def tensor_at_com(shape, rotation: np.ndarray) -> np.ndarray:
    matrix_mm5 = matrix3(shape.MatrixOfInertia)
    return rotation.T @ (matrix_mm5 * DENSITY_KG_M3 * 1.0e-15) @ rotation


def add_parallel_axis(inertia_at_com: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    return inertia_at_com + mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))


def collect_objects(doc):
    objects = {}
    for obj in doc.Objects:
        if not hasattr(obj, "Shape") or obj.Shape.isNull():
            continue
        objects[obj.Label] = shape_global(obj)
    return objects


def mesh_for_shape(shape, rotation: np.ndarray, anchor_global_mm: np.ndarray) -> trimesh.Trimesh:
    verts, faces = shape.tessellate(LINEAR_DEFLECTION_MM)
    vertices_mm = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
    vertices_local_m = ((vertices_mm - anchor_global_mm) @ rotation) * 1.0e-3
    return trimesh.Trimesh(vertices=vertices_local_m, faces=np.asarray(faces, dtype=np.int64), process=False)


def shape_record(shape, rotation: np.ndarray, anchor_global_mm: np.ndarray) -> dict:
    volume_mm3 = float(shape.Volume)
    mass = volume_mm3 * 1.0e-9 * DENSITY_KG_M3
    com_global_mm = vec_from_freecad(shape.CenterOfMass)
    com_local_m = rotation.T @ ((com_global_mm - anchor_global_mm) * 1.0e-3)
    inertia_local = tensor_at_com(shape, rotation)
    return {
        "volume_mm3": volume_mm3,
        "mass_kg": mass,
        "com_global_mm": com_global_mm,
        "com_local_m": com_local_m,
        "inertia_com_kg_m2": inertia_local,
    }


def combine_records(records: list[dict]) -> dict:
    mass = sum(item["mass_kg"] for item in records)
    com = sum((item["mass_kg"] * item["com_local_m"] for item in records), np.zeros(3)) / mass
    inertia_origin = sum(
        (add_parallel_axis(item["inertia_com_kg_m2"], item["mass_kg"], item["com_local_m"]) for item in records),
        np.zeros((3, 3)),
    )
    inertia_com = inertia_origin - add_parallel_axis(np.zeros((3, 3)), mass, com)
    eigvals, eigvecs = np.linalg.eigh((inertia_com + inertia_com.T) * 0.5)
    eigvals = np.maximum(eigvals, max(float(np.max(eigvals)), 1.0e-12) * 1.0e-9)
    inertia_com = (eigvecs * eigvals) @ eigvecs.T
    return {"mass_kg": float(mass), "com_local_m": com, "inertia_com_kg_m2": inertia_com}


def apply_measured_group_masses(records_by_body: dict[str, dict], groups: list[tuple[list[str], float]]) -> None:
    """Scale CAD-derived mass and inertia within each measured assembly group."""
    for bodies, target_mass in groups:
        cad_mass = sum(records_by_body[body]["mass_kg"] for body in bodies)
        scale = target_mass / cad_mass
        for body in bodies:
            records_by_body[body]["mass_kg"] *= scale
            records_by_body[body]["inertia_com_kg_m2"] *= scale


def add_inertial(link, props):
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=v3(props["com_local_m"]), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f'{props["mass_kg"]:.9g}')
    inertia = props["inertia_com_kg_m2"]
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.9g}",
        iyy=f"{inertia[1, 1]:.9g}",
        izz=f"{inertia[2, 2]:.9g}",
        ixy=f"{inertia[0, 1]:.9g}",
        ixz=f"{inertia[0, 2]:.9g}",
        iyz=f"{inertia[1, 2]:.9g}",
    )


def add_geometry(link, mesh_name):
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "mesh", filename=f"meshes/{mesh_name}")


def add_fixed_joint(robot, name, parent, child, xyz, rpy=(0.0, 0.0, 0.0)):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "origin", xyz=v3(np.asarray(xyz, dtype=float)), rpy=v3(np.asarray(rpy, dtype=float)))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)


def add_revolute_joint(robot, name, parent, child, xyz, joint_type="revolute", lower=None, upper=None):
    joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
    ET.SubElement(joint, "origin", xyz=v3(np.asarray(xyz, dtype=float)), rpy="0 0 0" if name.startswith("joint_thigh") or name.startswith("joint_calf") else v3(WHEEL_RPY))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "axis", xyz="0 0 1")
    if joint_type == "continuous":
        ET.SubElement(joint, "limit", effort="0.45", velocity="52.3598776")
    else:
        ET.SubElement(joint, "limit", lower=str(lower), upper=str(upper), effort="12.5", velocity="12.5663706")


def write_readme(report):
    rows = []
    for link in LINK_ORDER:
        item = report["links"][link]
        rows.append(f"| `{link}` | {item['mass_kg']:.6f} | {v3(np.array(item['com_local_m']))} |")
    imu = report["imu"]
    readme = (
        "# Tancho v3 standalone URDF asset\n\n"
        "This directory is generated from `Tancho_simplified.step`. Meshes use the v2 local-link convention: each mesh origin is the owning joint axis, and the v2 base rotation is retained in `base_link_rotation_joint`.\n\n"
        "Measured assembly masses are distributed over their STEP solids before COM and inertia are recomputed. The base body is 348 g, while the restored drawer and its battery are one 400 g link. The 379 g upper assembly is split into a 300 g joint and 79 g link; the 341 g lower assembly into a 300 g joint and 41 g link; and the 152 g wheel assembly into a 116 g hub and 36 g wheel.\n\n"
        "## Geometry and frames\n\n"
        "- hip-to-knee anchor separation: 0.150000 m\n"
        "- knee-to-wheel anchor separation: 0.09999754 m (the CAD wheel anchors are 99.99754 mm apart)\n"
        "- IMU: midpoint of the Main_body and Main_body002 bounding-box centers, translated CAD Y -30 mm\n"
        f"- IMU CAD position (mm): `{v3(np.array(imu['cad_mm']))}`\n"
        f"- IMU base_link position (m): `{v3(np.array(imu['base_link_m']))}`\n\n"
        "## Recomputed link properties\n\n"
        "| Link | Mass (kg) | COM in local link frame (m) |\n|---|---:|---|\n"
        + "\n".join(rows)
        + "\n\n## Files\n\n"
        "- `Tancho_v3.urdf`: URDF with `imu_link` fixed to `base_link`\n"
        "- `meshes/*.stl`: metre-scale local-frame visual/collision meshes using the Tancho v2 semantic filenames\n"
        "- `mass_properties.json`: source-object and combined mass properties\n"
        "- `convert_tancho_v3.py`: repeatable conversion\n"
        "- `validate_tancho_v3.py`: structural, geometry, anchor, IMU, and inertia checks\n\n"
        "Run the converter with the FreeCAD environment Python, then run the validator with the same interpreter.\n"
    )
    README_PATH.write_text(readme, encoding="utf-8")


def convert(step_path: Path) -> dict:
    FreeCAD, Import = _load_cad_modules()
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for mesh in MESH_DIR.glob("*.stl"):
        mesh.unlink()

    doc = FreeCAD.newDocument("tancho_v3")
    Import.open(str(step_path), doc.Name)
    doc.recompute()
    objects = collect_objects(doc)
    missing = sorted({name for names in LINK_BODIES.values() for name in names} - set(objects))
    if missing:
        raise RuntimeError(f"STEP objects missing: {missing}")

    anchor_global = {"base_link": BASE_TO_CAD_MM.copy(), "drawer": np.array([0.0, 0.0, 0.0]), "pi_case": np.array([0.0, 0.0, 0.0])}
    for link, body in ANCHOR_BODIES.items():
        anchor_global[link] = vec_from_freecad(objects[body].CenterOfMass)
    link_rotation = {link: np.eye(3) for link in LINK_ORDER}
    link_rotation["wheel_L"] = rpy_to_matrix(WHEEL_RPY)
    link_rotation["wheel_R"] = rpy_to_matrix(WHEEL_RPY)
    bbox_centers = {}
    for body in ("Main_body", "Main_body002"):
        box = objects[body].BoundBox
        bbox_centers[body] = np.array(
            [(box.XMin + box.XMax) * 0.5, (box.YMin + box.YMax) * 0.5, (box.ZMin + box.ZMax) * 0.5],
            dtype=float,
        )

    combined = {}
    source_report = {}
    for link in LINK_ORDER:
        anchor = anchor_global[link]
        rotation = link_rotation[link]
        records_by_body = {}
        source_report[link] = {}
        for body in LINK_BODIES[link]:
            shape = objects[body]
            record = shape_record(shape, rotation, anchor)
            records_by_body[body] = record
        apply_measured_group_masses(records_by_body, MASS_GROUPS_KG[link])
        records = []
        for body in LINK_BODIES[link]:
            record = records_by_body[body]
            records.append(record)
            source_report[link][body] = {
                "volume_mm3": record["volume_mm3"],
                "mass_kg": record["mass_kg"],
                "com_global_mm": record["com_global_mm"].tolist(),
                "com_local_m": record["com_local_m"].tolist(),
            }
        mesh_names = []
        for group, filename in MESH_GROUPS[link]:
            meshes = [mesh_for_shape(objects[body], rotation, anchor) for body in group]
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            mesh.export(MESH_DIR / filename)
            mesh_names.append(filename)
        combined[link] = combine_records(records)
        combined[link]["anchor_global_mm"] = anchor.tolist()
        combined[link]["mesh_files"] = mesh_names

    imu_cad = (bbox_centers["Main_body"] + bbox_centers["Main_body002"]) * 0.5 + np.array([0.0, -30.0, 0.0])
    imu_base = (imu_cad - BASE_TO_CAD_MM) * 1.0e-3

    robot = ET.Element("robot", name="Tancho_v3")
    ET.SubElement(robot, "link", name="base_link_root")
    for link_name in LINK_ORDER:
        link = ET.SubElement(robot, "link", name=link_name)
        add_inertial(link, combined[link_name])
        for mesh_name in combined[link_name]["mesh_files"]:
            add_geometry(link, mesh_name)
    ET.SubElement(robot, "link", name="imu_link")

    add_fixed_joint(robot, "base_link_rotation_joint", "base_link_root", "base_link", [0, 0, 0], ROOT_RPY)
    add_fixed_joint(robot, "joint_drawer_R", "base_link", "drawer", [-0.01, 0.00055265617, 0])
    add_fixed_joint(robot, "joint_pi_case_R", "base_link", "pi_case", [-0.01, 0.00055265617, 0])
    add_revolute_joint(robot, "joint_thigh_L", "base_link", "thigh_L", (anchor_global["thigh_L"] - BASE_TO_CAD_MM) * 1.0e-3, lower=-1.57, upper=1.57)
    add_revolute_joint(robot, "joint_calf_L", "thigh_L", "calf_L", (anchor_global["calf_L"] - anchor_global["thigh_L"]) * 1.0e-3, lower=0, upper=3)
    add_revolute_joint(robot, "joint_wheel_L", "calf_L", "wheel_L", (anchor_global["wheel_L"] - anchor_global["calf_L"]) * 1.0e-3, joint_type="continuous")
    add_revolute_joint(robot, "joint_thigh_R", "base_link", "thigh_R", (anchor_global["thigh_R"] - BASE_TO_CAD_MM) * 1.0e-3, lower=-1.57, upper=1.57)
    add_revolute_joint(robot, "joint_calf_R", "thigh_R", "calf_R", (anchor_global["calf_R"] - anchor_global["thigh_R"]) * 1.0e-3, lower=0, upper=3)
    add_revolute_joint(robot, "joint_wheel_R", "calf_R", "wheel_R", (anchor_global["wheel_R"] - anchor_global["calf_R"]) * 1.0e-3, joint_type="continuous")
    add_fixed_joint(robot, "imu_joint", "base_link", "imu_link", imu_base)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)

    report = {
        "robot": "Tancho_v3",
        "step": str(step_path),
        "density_kg_m3": DENSITY_KG_M3,
        "mass_source": "measured assembly masses distributed over STEP geometry",
        "base_to_cad_mm": BASE_TO_CAD_MM.tolist(),
        "links": {
            link: {
                "mass_kg": combined[link]["mass_kg"],
                "com_local_m": combined[link]["com_local_m"].tolist(),
                "inertia_com_kg_m2": combined[link]["inertia_com_kg_m2"].tolist(),
                "anchor_global_mm": combined[link]["anchor_global_mm"],
                "mesh_files": combined[link]["mesh_files"],
                "source_objects": source_report[link],
            }
            for link in LINK_ORDER
        },
        "anchors_global_mm": {link: anchor_global[link].tolist() for link in anchor_global},
        "imu": {
            "bbox_centers_cad_mm": {key: value.tolist() for key, value in bbox_centers.items()},
            "cad_mm": imu_cad.tolist(),
            "base_link_m": imu_base.tolist(),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(report)
    FreeCAD.closeDocument(doc.Name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=Path, default=DEFAULT_STEP)
    args = parser.parse_args()
    report = convert(args.step)
    print(f"WROTE {URDF_PATH}")
    print(f"WROTE {len(list(MESH_DIR.glob('*.stl')))} meshes to {MESH_DIR}")
    for link in LINK_ORDER:
        item = report["links"][link]
        print(f"MASS {link:10s} {item['mass_kg']:.9f} kg COM(m) {v3(np.array(item['com_local_m']))}")
    print(f"IMU CAD(mm) {v3(np.array(report['imu']['cad_mm']))} BASE(m) {v3(np.array(report['imu']['base_link_m']))}")


if __name__ == "__main__":
    main()
