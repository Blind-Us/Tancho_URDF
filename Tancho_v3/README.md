# Tancho v3 standalone URDF asset

This directory is generated from `Tancho_simplified.step`. Meshes use the v2 local-link convention: each mesh origin is the owning joint axis, and the v2 base rotation is retained in `base_link_rotation_joint`.

Measured assembly masses are distributed over their STEP solids before COM and inertia are recomputed. The base body is 348 g, while the restored drawer and its battery are one 400 g link. The 379 g upper assembly is split into a 300 g joint and 79 g link; the 341 g lower assembly into a 300 g joint and 41 g link; and the 152 g wheel assembly into a 116 g hub and 36 g wheel.

## Geometry and frames

- hip-to-knee anchor separation: 0.150000 m
- knee-to-wheel anchor separation: 0.09999754 m (the CAD wheel anchors are 99.99754 mm apart)
- IMU: midpoint of the Main_body and Main_body002 bounding-box centers, translated CAD Y -30 mm
- IMU CAD position (mm): `1.64258001 -29.9864503 -1.60456261e-05`
- IMU base_link position (m): `-0.00835741999 -0.0294337942 -1.60456261e-08`

## Recomputed link properties

| Link | Mass (kg) | COM in local link frame (m) |
|---|---:|---|
| `base_link` | 0.348000 | 0.0011187875 0.00127367658 -1.70173615e-08 |
| `drawer` | 0.400000 | -0.0211872329 -0.00792307398 1.62167735e-08 |
| `pi_case` | 0.146000 | 0.0233176478 0.0504234963 2.16819542e-07 |
| `thigh_L` | 0.379000 | -0.000408445945 -0.0229151318 -0.00280925304 |
| `calf_L` | 0.341000 | 0.000188809828 -0.00516026443 -0.00324688923 |
| `wheel_L` | 0.152000 | 4.74095232e-11 5.83255949e-07 0.00192447727 |
| `thigh_R` | 0.379000 | -0.00040844648 -0.0229151208 0.00280925412 |
| `calf_R` | 0.341000 | 0.000188809828 -0.00516026443 0.00324688923 |
| `wheel_R` | 0.152000 | 2.1907452e-11 5.83325029e-07 -0.00192447731 |

## Files

- `Tancho_v3.urdf`: URDF with `imu_link` fixed to `base_link`
- `meshes/*.stl`: metre-scale local-frame visual/collision meshes using the Tancho v2 semantic filenames
- `mass_properties.json`: source-object and combined mass properties
- `convert_tancho_v3.py`: repeatable conversion
- `validate_tancho_v3.py`: structural, geometry, anchor, IMU, and inertia checks

Run the converter with the FreeCAD environment Python, then run the validator with the same interpreter.
