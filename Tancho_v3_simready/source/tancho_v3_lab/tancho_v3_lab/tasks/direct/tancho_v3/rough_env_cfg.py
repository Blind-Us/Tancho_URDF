import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .flat_env_cfg import FlatRewardsCfg, TanchoV3FlatEnvCfg, TanchoV3FlatSceneCfg, make_flat_terrain


def make_rough_terrain() -> TerrainImporterCfg:
    generator = TerrainGeneratorCfg(seed=0, size=(6.0, 6.0), border_width=10.0, num_rows=8, num_cols=12, horizontal_scale=0.05, vertical_scale=0.0025, slope_threshold=0.75, use_cache=False, curriculum=True, difficulty_range=(0.0, 1.0), sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.5, noise_range=(0.002, 0.025), noise_step=0.0025, border_width=0.25),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.3, slope_range=(0.0, 0.18), platform_width=2.0, border_width=0.25),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.2, grid_width=0.26, grid_height_range=(0.005, 0.035), platform_width=2.0),
    })
    flat = make_flat_terrain()
    return TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator", terrain_generator=generator, max_init_terrain_level=1, collision_group=-1, physics_material=flat.physics_material, debug_vis=False)


@configclass
class RoughRewardsCfg(FlatRewardsCfg):
    pass


@configclass
class TanchoV3RoughSceneCfg(TanchoV3FlatSceneCfg):
    terrain = make_rough_terrain()


@configclass
class TanchoV3RoughEnvCfg(TanchoV3FlatEnvCfg):
    scene: TanchoV3RoughSceneCfg = TanchoV3RoughSceneCfg(num_envs=4096, env_spacing=3.0)
    rewards: RoughRewardsCfg = RoughRewardsCfg()
