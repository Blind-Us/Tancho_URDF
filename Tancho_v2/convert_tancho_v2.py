#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tancho v2 STEP -> Isaac Lab URDF pipeline  (LOCAL FRAMING edition)

Convention
----------
* Joint skeleton (names, hierarchy, types, origins, axes, limits) is cloned
  verbatim from the validated Tencho_v1 URDF — including the 0.15 m spacing.
* Every exported STL is re-anchored to its *own link frame*:
      V_local = R_chain^T · ( V_global - Joint_Position_global )
  where Joint_Position_global is the link's joint-axis location expressed in
  the STEP global frame, and R_chain is the link-frame rotation implied by the
  v1 joint chain (identity everywhere except the wheel links, whose v1 joint
  carries rpy=(pi,0,0)).
* Consequently each STL's (0,0,0) is exactly the joint rotation axis, the URDF
  <visual>/<collision> origins are all zero, and no centroid alignment is used
  anywhere (removes tessellation-density bias entirely).

Frame relationship (measured, see report):
      STEP global frame == v1 base_link frame + (10, -0.5527, 0) mm
  i.e. b = -v1_base_visual_origin. Verified: v1 chain position of joint_calf_L
  (19.2386, -149.4473, -93) mm + b = (29.2386, -150.0, -93) mm, which matches
  the measured v2 'joint_knee_L' housing cylinder center (29.2, -150, -93).
"""
import os, sys, math
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/azul/miniconda3/envs/freecad-urdf/lib')
import FreeCAD, Part, Import
import numpy as np
import trimesh

STEP_PATH = "/media/azul/861896C11896B023/Tancho/Tancho_simplified (Copy).step"
V1_URDF   = "/media/azul/861896C11896B023/Tancho/Tencho_v1/Tencho_v1.urdf"
OUT_DIR   = "/media/azul/861896C11896B023/Tancho/Tancho_v2"
OUT_MESH  = os.path.join(OUT_DIR, "meshes")
OUT_URDF  = os.path.join(OUT_DIR, "Tancho_v2.urdf")

LINEAR_DEFLECTION = 0.02      # mm chord error  (high precision)
DENSITY = 1240.0              # kg/m^3 (PLA)

# ---------------------------------------------------------------- v1 parse --
tree = ET.parse(V1_URDF)
root = tree.getroot()

v1_joints = []          # file order guarantees parents before children
for j in root.findall('joint'):
    o = j.find('origin')
    v1_joints.append(dict(
        name=j.get('name'), type=j.get('type'),
        parent=j.find('parent').get('link'), child=j.find('child').get('link'),
        xyz=np.array([float(v) for v in o.get('xyz').split()]),
        rpy=np.array([float(v) for v in o.get('rpy').split()]),
        axis=(j.find('axis').get('xyz') if j.find('axis') is not None else None),
        limit=(dict(j.find('limit').attrib) if j.find('limit') is not None else None),
        elem=j))

v1_link_visuals = {}
for link in root.findall('link'):
    lname = link.get('name')
    o = link.find('visual/origin')
    v1_link_visuals[lname] = np.array([float(v) for v in o.get('xyz').split()])

def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,    cp*sr,           cp*cr]])

# ------------------------------------------------- v1 joint chain (base) --
# Forward kinematics at zero joint position, in the v1 base frame.
R_chain = {'base_link': np.eye(3)}
t_chain = {'base_link': np.zeros(3)}
for j in v1_joints:
    Rp, tp = R_chain[j['parent']], t_chain[j['parent']]
    R_chain[j['child']] = Rp @ rpy_to_R(j['rpy'])
    t_chain[j['child']] = tp + Rp @ j['xyz']

# STEP-global == v1 base frame + b   (b = -v1 base_link visual origin)
b = -v1_link_visuals['base_link']

# v2 body label -> URDF link
BODY2LINK = {
    'base_link': 'base_link', 'base_link001': 'base_link', 'base_link002': 'base_link',
    'drawer': 'drawer', 'pi_case': 'pi_case',
    'joint_thigh_R': 'thigh_R', 'joint_knee_R': 'calf_R', 'joint_wheel_R': 'wheel_R',
    'joint_thigh_L': 'thigh_L', 'joint_knee_L': 'calf_L', 'joint_wheel_L': 'wheel_L',
    'thigh_R': 'thigh_R', 'knee_cover_R': 'thigh_R', 'calf_R': 'calf_R',
    'wheel_R': 'wheel_R', 'wheel_R001': 'wheel_R',
    'thigh_L': 'thigh_L', 'knee_cover_L': 'thigh_L', 'calf_L': 'calf_L',
    'wheel_L': 'wheel_L', 'wheel_L001': 'wheel_L',
}
LINK_ORDER = ['base_link', 'pi_case', 'drawer',
              'thigh_L', 'calf_L', 'wheel_L',
              'thigh_R', 'calf_R', 'wheel_R']
# body used to measure joint-axis concentricity per revolute link
AXIS_BODY = {'thigh_L': 'joint_thigh_L', 'calf_L': 'joint_knee_L', 'wheel_L': 'joint_wheel_L',
             'thigh_R': 'joint_thigh_R', 'calf_R': 'joint_knee_R', 'wheel_R': 'joint_wheel_R'}

def make_PD(I, floor_scale=1e-3):
    w, V = np.linalg.eigh(I)
    w = np.maximum(w, max(w.max(), 1e-12) * floor_scale)
    return (V * w) @ V.T

def main():
    os.makedirs(OUT_MESH, exist_ok=True)

    print("=" * 74)
    print("[1/4] Loading STEP with FreeCAD ...")
    doc = FreeCAD.newDocument("tancho")
    Import.open(STEP_PATH, doc.Name)

    bodies = {}
    for obj in doc.Objects:
        if obj.TypeId != 'Part::Feature':
            continue
        if obj.Label not in BODY2LINK:
            continue
        shape = obj.Shape.copy()
        shape.Placement = obj.getGlobalPlacement()
        bodies[obj.Label] = shape
    print(f"      found {len(bodies)} v2 bodies")

    print(f"[2/4] High-precision tessellation (LinearDeflection={LINEAR_DEFLECTION} mm) ...")
    v2_mesh_mm = {}
    for label, shape in bodies.items():
        verts, faces = shape.tessellate(LINEAR_DEFLECTION)
        v2_mesh_mm[label] = trimesh.Trimesh(
            vertices=np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64),
            faces=np.array(faces, dtype=np.int64), process=False)
        print(f"      {label:18s} verts={len(verts):7d} faces={len(faces):7d}")

    print("[3/4] Local framing: V_local = R_chain^T (V_global - Joint_Position) ...")
    out_link_meshes = {l: [] for l in LINK_ORDER}
    link_massprops, local_meshes = {}, {}
    for link in LINK_ORDER:
        R, t = R_chain[link], t_chain[link]
        J = t + b                       # joint-axis position in STEP global frame (m)
        labels = [l for l in v2_mesh_mm if BODY2LINK[l] == link]
        locals_ = []
        for label in labels:
            mm = v2_mesh_mm[label].copy()
            mm.vertices = (mm.vertices * 0.001 - J) @ R     # -> link frame, meters
            mm.export(os.path.join(OUT_MESH, label + '.stl'))
            out_link_meshes[link].append(label + '.stl')
            locals_.append(mm)
        union_m = trimesh.util.concatenate(locals_)
        local_meshes[link] = union_m
        if not union_m.is_watertight:
            trimesh.repair.fill_holes(union_m)
        volume = abs(union_m.volume)
        if not np.isfinite(volume) or volume <= 0:
            volume = union_m.convex_hull.volume
        mass = volume * DENSITY
        com = union_m.center_mass
        I_com = union_m.moment_inertia * DENSITY
        I_origin = make_PD(I_com + mass * ((com @ com) * np.eye(3) - np.outer(com, com)))
        link_massprops[link] = dict(mass=mass, com=com, I=I_origin)
        print(f"      {link:10s} J_global={np.round(J, 5)}  mass={mass:7.4f} kg")

    print("[4/4] Writing URDF (visual/collision origins = 0) ...")
    robot = ET.Element('robot', name='Tancho_v2')
    for lname in LINK_ORDER:
        le = ET.SubElement(robot, 'link', name=lname)
        mp = link_massprops[lname]
        iner = ET.SubElement(le, 'inertial')
        ET.SubElement(iner, 'origin', xyz=' '.join(f'{v:.8g}' for v in mp['com']), rpy='0 0 0')
        ET.SubElement(iner, 'mass', value=f"{mp['mass']:.8g}")
        I = mp['I']
        ET.SubElement(iner, 'inertia',
                      ixx=f'{I[0,0]:.8g}', iyy=f'{I[1,1]:.8g}', izz=f'{I[2,2]:.8g}',
                      ixy=f'{I[0,1]:.8g}', ixz=f'{I[0,2]:.8g}', iyz=f'{I[1,2]:.8g}')
        for fname in out_link_meshes[lname]:
            for tag in ('visual', 'collision'):
                el = ET.SubElement(le, tag)
                ET.SubElement(el, 'origin', xyz='0 0 0', rpy='0 0 0')
                ET.SubElement(el, 'geometry').append(ET.Element('mesh', filename=f'meshes/{fname}'))
    for j in v1_joints:
        robot.append(j['elem'])
    ET.indent(ET.ElementTree(robot), space='  ')
    ET.ElementTree(robot).write(OUT_URDF, encoding='utf-8', xml_declaration=True)
    print(f"      wrote {OUT_URDF}")

    # ------------------------------------------------ alignment report ----
    print("=" * 74)
    print("ALIGNMENT & CONCENTRICITY REVIEW")
    print("-" * 74)
    print("Joint positions in STEP-global frame vs measured v2 housing centers (mm):")
    print(f"  {'link':9s} {'J_global (from v1 chain + b)':>34s} {'housing center':>24s}  radial err")
    for link, bodyname in AXIS_BODY.items():
        Jmm = (t_chain[link] + b) * 1000.0
        m = v2_mesh_mm[bodyname]
        bb = m.bounds
        # housing cylinder is coaxial with the joint axis (Z) -> measure (X,Y) center
        center = np.array([(bb[0][0]+bb[1][0])/2, (bb[0][1]+bb[1][1])/2, Jmm[2]])
        err = np.hypot(center[0]-Jmm[0], center[1]-Jmm[1])
        print(f"  {link:9s} {str(np.round(Jmm,3)):>34s} {str(np.round(center,3)):>24s}  {err:6.3f} mm")

    print("-" * 74)
    print("Sanity check — each link's exported STL contains its own joint axis (0,0,0):")
    for link in ['thigh_L', 'calf_L', 'wheel_L', 'thigh_R', 'calf_R', 'wheel_R']:
        m = local_meshes[link]
        # closest surface vertex to the joint axis (origin) — pure numpy, no rtree
        d = np.linalg.norm(m.vertices, axis=1).min()
        print(f"  {link:9s}  nearest surface vertex to axis = {d*1000:7.3f} mm")

    print("-" * 74)
    print("Joint-to-joint distances (v1 skeleton, should match CAD):")
    pairs = [('base_link', 'thigh_L'), ('base_link', 'thigh_R'),
             ('thigh_L', 'calf_L'), ('thigh_R', 'calf_R'),
             ('calf_L', 'wheel_L'), ('calf_R', 'wheel_R')]
    for p, c in pairs:
        d = np.linalg.norm(t_chain[c] - t_chain[p]) * 1000.0
        print(f"  {p:9s} -> {c:9s} : {d:8.3f} mm")

    print("-" * 74)
    print("Per-link exported STL bounds in LOCAL link frame (m); origin = joint axis:")
    for link in LINK_ORDER:
        bb = local_meshes[link].bounds
        print(f"  {link:10s} min={np.round(bb[0],4)}  max={np.round(bb[1],4)}")
    print("=" * 74)

if __name__ == '__main__':
    main()
