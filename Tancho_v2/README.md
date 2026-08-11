# Tancho_v2 — Isaac Lab URDF Package

Converted from `Tancho_simplified (Copy).step` (Fusion 360 export) with the
joint skeleton inherited 1:1 from the validated `Tencho_v1` URDF.

## Contents

```
Tancho_v2/
├── Tancho_v2.urdf          # robot name="Tancho_v2", 9 links / 8 joints
├── meshes/                 # 21 high-precision STL files (meters)
│   ├── base_link.stl  base_link001.stl  base_link002.stl
│   ├── drawer.stl  pi_case.stl
│   ├── thigh_L.stl  knee_cover_L.stl  joint_thigh_L.stl
│   ├── calf_L.stl   joint_knee_L.stl
│   ├── wheel_L.stl  wheel_L001.stl  joint_wheel_L.stl
│   └── (mirrored _R set)
├── convert_tancho_v2.py    # conversion pipeline (FreeCAD + trimesh)
└── validate_tancho_v2.py   # skeleton + world-geometry + PD-inertia check
```

## Key conventions

| Item | Value |
|---|---|
| Robot name | `Tancho_v2` |
| Mesh units | **meters** (STEP mm × 0.001, applied to vertices) |
| Mesh frame | **local link frame** — each STL's origin `(0,0,0)` is exactly that link's joint rotation axis. `V_local = R_chain^T (V_global − Joint_Position)`. No visual-origin compensation anywhere. |
| URDF `<visual>/<collision>` origins | all `0 0 0` (mesh already in link frame) |
| Tessellation | `Shape.tessellate(LinearDeflection=0.02 mm)` |
| Density (uniform) | 1240 kg/m³ (PLA) |
| Inertia | computed from v2 mesh at link origin, eigenvalue-clamped positive-definite |
| Joints | cloned verbatim from v1 (names, types, origins — incl. 0.15 m spacing — axes, limits) |

## Loading in Isaac Lab

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
import isaaclab.sim as sim_utils

TANCHO_V2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/media/azul/861896C11896B023/Tancho/Tancho_v2/Tancho_v2.urdf",
        fix_base=False,
        merge_fixed_joints=False,   # keep drawer/pi_case as separate bodies
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),       # base height above ground
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=["joint_thigh_.*", "joint_calf_.*"],
            stiffness=20.0, damping=1.0, effort_limit=10.0, velocity_limit=5.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            stiffness=0.0, damping=0.5, effort_limit=10.0, velocity_limit=5.0,
        ),
    },
)
```

Notes for Isaac Lab:
* Meshes are in **meters** and already in each link's local frame, so
  `UrdfFileCfg` needs **no** extra `scale` and there are no visual-origin
  offsets to fight.
* Joint limits/efforts come straight from v1 (`±1.57 rad` thigh/calf,
  `±3.14 rad` wheels, `effort=10`, `velocity=5`).
* Collision meshes == visual meshes (watertight, high-resolution). If PhysX
  complains about contact density, replace `collision` with a convex
  decomposition (`coacd`) per link.

## Re-running / re-validation

```bash
# rebuild everything from the STEP file
~/miniconda3/envs/freecad-urdf/bin/python convert_tancho_v2.py

# check skeleton vs v1 + world bboxes + positive-definite inertia
python3 validate_tancho_v2.py
```
