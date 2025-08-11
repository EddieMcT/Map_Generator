from pathlib import Path
import unittest
import tempfile

from map_generator.globals import REPO_ROOT
from map_generator.parameters import load_params, WorldParams, ImagingParams, to_yaml


class TestRootParams(unittest.TestCase):

    def test_round_trip_load_then_save_then_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            params = load_params(REPO_ROOT / "params/default", output_folder=tmpdir_path)
            timestamped_folder = params.root_params.timestamped_output_folder

            params.root_params.save()

            tmp_world = timestamped_folder / "world_params.yaml"
            self.assertTrue(tmp_world.exists(), "world_params.yaml was not written in temp folder")
            tmp_imaging = timestamped_folder / "imaging_params.yaml"
            self.assertTrue(tmp_imaging.exists(), "imaging_params.yaml was not written in temp folder")

            params_reloaded = load_params(timestamped_folder, output_folder=tmpdir_path)

            self.assertEqual(
                params.root_params, params_reloaded.root_params,
                "first load and reloaded root params differ"
            )

    def test_world_params_round_trip_save_then_load_yaml(self):
        world = WorldParams(
            world_size=10,
            mountain_heights=0.3,
            river_scale=100.0,
            centroids=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
            heights_tectonic_plates=[-1.0, 0.35, 0.1],
            slopes_x=[0.0, 0.12, -0.05],
            slopes_y=[0.0, -0.2, 0.3],
            river_density=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            yaml_path = tmpdir_path / 'world_params.yaml'

            to_yaml(world, yaml_path)
            world2 = WorldParams.from_yaml(yaml_path)

            self.assertTrue(yaml_path.exists(), "YAML file was not written in temp folder")
            self.assertEqual(world, world2, "Mismatch after YAML roundtrip")

    def test_imaging_params_round_trip_save_then_load_yaml(self):
        imaging = ImagingParams(
            offset_x=1.0,
            offset_y=-2.0,
            zoom=2.0,
            resolution=16,
            tiling=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            yaml_path = tmpdir_path / 'imaging_params.yaml'

            to_yaml(imaging, yaml_path)
            imaging2 = ImagingParams.from_yaml(yaml_path)

            self.assertTrue(yaml_path.exists(), "YAML file was not written in temp folder")
            self.assertEqual(imaging, imaging2, "Mismatch after YAML roundtrip")


if __name__ == '__main__':
    unittest.main()
