import ast
from pathlib import Path
import unittest
import tempfile

from map_generator.parameters import load_parameters, save_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_file_values(path: Path):
    """
    Parse the right-hand side of each "key: value" line into Python types for
    robust, formatting-agnostic comparisons.

    Why this complexity:
    - The repository’s parameter files are human-written and may vary in benign
      formatting (e.g., 0.4 vs. 0.40, 300 vs. 300.0, spacing within lists).
    - A strict string comparison would fail on these harmless differences.
    - By parsing RHS values (numbers, lists), we compare the actual data rather
      than its textual representation, keeping the test stable while still
      detecting real semantic differences.

    If parsing fails for any line, we fall back to comparing the original RHS
    string, ensuring we don’t hide unexpected content changes.
    """
    parsed = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.strip() == "":
            continue
        key, rhs = line.split(':', 1)
        rhs = rhs.strip()
        # Try to parse RHS into Python types when possible
        try:
            val = ast.literal_eval(rhs)
        except (ValueError, SyntaxError):
            val = rhs
        parsed.append((key.strip(), val))
    return parsed


class TestParameters(unittest.TestCase):

    def test_round_trip_load_save_matches_originals(self):
        orig_world = REPO_ROOT / 'world_parameters.txt'
        orig_imaging = REPO_ROOT / 'imaging_parameters.txt'
        orig_sampling = REPO_ROOT / 'sampling_coordinates.csv'
        params = load_parameters(
            world_params_path=str(orig_world),
            imaging_params_path=str(orig_imaging),
            sampling_coords_path=str(orig_sampling),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tmp_world = tmpdir_path / 'world_parameters.txt'
            tmp_imaging = tmpdir_path / 'imaging_parameters.txt'
            tmp_sampling = tmpdir_path / 'sampling_coordinates.csv'

            save_parameters(
                params,
                world_params_path=str(tmp_world),
                imaging_params_path=str(tmp_imaging),
                sampling_coords_path=str(tmp_sampling),
            )

            self.assertTrue(tmp_world.exists(), "world_parameters was not written in temp folder")
            self.assertTrue(tmp_imaging.exists(), "imaging_parameters was not written in temp folder")

            orig_world_vals = _parse_file_values(orig_world)
            tmp_world_vals = _parse_file_values(tmp_world)
            self.assertEqual(
                orig_world_vals, tmp_world_vals,
                "world_parameters.txt values differ between original and round-trip saved file"
            )

            orig_imaging_vals = _parse_file_values(orig_imaging)
            tmp_imaging_vals = _parse_file_values(tmp_imaging)
            self.assertEqual(
                orig_imaging_vals, tmp_imaging_vals,
                "imaging_parameters.txt values differ between original and round-trip saved file"
            )


if __name__ == '__main__':
    unittest.main()
