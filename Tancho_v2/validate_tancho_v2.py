#!/usr/bin/env python3
"""Validate Tancho_v2 URDF (LOCAL FRAMING edition) against v1 skeleton.

Ad-hoc verification — checks:
  1. robot name, link/joint counts
  2. joint skeleton identical to v1 (origins incl. 0.15 m spacing, axes, limits)
  3. all visual/collision origins are zero (local framing contract)
  4. joint-axis concentricity: each revolute link's housing cylinder is
     centered on (0,0) in its local XY plane
  5. child joint positions land inside the parent link's geometry
  6. inertia tensors positive-definite, mass > 0
"""
import os, sys
import xml.etree.ElementTree as ET
import numpy as np, trimesh

V2 = '/media/azul/861896C11896B023/Tancho/Tancho_v2'
V1 = '/media/azul/861896C11896B023/Tancho/Tencho_v1/Tencho_v1.urdf'

fails = []
def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))
    if not ok: fails.append(name)

r2 = ET.parse(os.path.join(V2, 'Tancho_v2.urdf')).getroot()
r1 = ET.parse(V1).getroot()
check('robot name == Tancho_v2', r2.get('name') == 'Tancho_v2')
check('9 links', len(r2.findall('link')) == 9)
check('8 joints', len(r2.findall('joint')) == 8)

def skel(r):
    out = {}
    for j in r.findall('joint'):
        o = j.find('origin')
        out[j.get('name')] = (j.get('type'), j.find('parent').get('link'),
            j.find('child').get('link'), o.get('xyz'), o.get('rpy'),
            j.find('axis').get('xyz') if j.find('axis') is not None else None,
            j.find('limit').attrib if j.find('limit') is not None else None)
    return out
check('joint skeleton identical to v1', skel(r1) == skel(r2))

zero, exist, meters = True, True, True
for l in r2.findall('link'):
    for tag in ('visual', 'collision'):
        for el in l.findall(tag):
            o = el.find('origin')
            if o.get('xyz') != '0 0 0' or o.get('rpy') != '0 0 0': zero = False
            p = os.path.join(V2, el.find('geometry/mesh').get('filename'))
            if not (os.path.exists(p) and os.path.getsize(p) > 1000): exist = False
            elif max(trimesh.load(p, force='mesh', process=False).extents) > 1.0: meters = False
check('all visual/collision origins are zero', zero)
check('all mesh files present & in meters', exist and meters)

print('\n  concentricity — housing cylinder XY center in local frame (mm):')
conc = True
for link, housing in [('thigh_L','joint_thigh_L'), ('calf_L','joint_knee_L'), ('wheel_L','joint_wheel_L'),
                      ('thigh_R','joint_thigh_R'), ('calf_R','joint_knee_R'), ('wheel_R','joint_wheel_R')]:
    m = trimesh.load(os.path.join(V2, 'meshes', housing + '.stl'), force='mesh', process=False)
    bb = m.bounds
    cx, cy = (bb[0,0]+bb[1,0])/2*1000, (bb[0,1]+bb[1,1])/2*1000
    ok = abs(cx) < 1.0 and abs(cy) < 1.0
    conc &= ok
    print(f'    {link:9s} ({housing:15s}) center = ({cx:7.4f}, {cy:7.4f}) mm  {"OK" if ok else "OFF-AXIS"}')
check('all joint housings concentric with joint axis (<1 mm)', conc)

print('\n  child joint position inside parent link geometry (mm):')
reach = True
child_in_parent = {'thigh_L': ('calf_L', (0.0, -0.15, -0.016)),
                   'calf_L': ('wheel_L', (0.0, -0.14999754, -0.037004615)),
                   'thigh_R': ('calf_R', (0.0, -0.15, 0.016)),
                   'calf_R': ('wheel_R', (0.0, -0.14999754, 0.060004615))}
for parent, (child, p) in child_in_parent.items():
    l = r2.find(f"link[@name='{parent}']")
    meshes = [os.path.join(V2, v.find('geometry/mesh').get('filename')) for v in l.findall('visual')]
    m = trimesh.util.concatenate([trimesh.load(x, force='mesh', process=False) for x in meshes])
    d = np.linalg.norm(m.vertices - np.array(p), axis=1).min() * 1000.0
    ok = d < 45.0   # child joint axis must lie within the parent's housing
    reach &= ok
    print(f'    {parent:9s} -> {child:9s} nearest surface vertex = {d:6.3f} mm  {"OK" if ok else "UNREACHABLE"}')
check('child joint positions inside parent geometry (<45 mm)', reach)

pd = True
for l in r2.findall('link'):
    i = l.find('inertial/inertia')
    I = np.array([[float(i.get(a)) for a in ('ixx','ixy','ixz')],
                  [float(i.get('ixy')), float(i.get('iyy')), float(i.get('iyz'))],
                  [float(i.get('ixz')), float(i.get('iyz')), float(i.get('izz'))]])
    if np.linalg.eigvalsh(I).min() <= 0 or float(l.find('inertial/mass').get('value')) <= 0: pd = False
check('all inertia positive-definite, mass>0', pd)

print()
if fails:
    print('RESULT: FAIL ->', fails); sys.exit(1)
print('RESULT: ALL CHECKS PASSED (ad-hoc verification, not a test suite)')
