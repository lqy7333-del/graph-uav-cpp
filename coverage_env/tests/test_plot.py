"""Headless smoke test for the field plot."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from coverage_env.env.field import Field
from coverage_env.geometry.discretization import discretize_field
from coverage_env.visualization.plot import plot_field, plot_grid


class FieldPlotTests(unittest.TestCase):
    """Check that the requested geometry visualization can be rendered."""

    def test_plot_saves_image(self):
        field = Field(
            1,
            [(31.0, 121.0), (31.0, 121.001), (31.001, 121.001), (31.001, 121.0)],
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "field.png"
            figure, _ = plot_field(field, show=False, save_path=str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)

    def test_grid_plot_saves_image(self):
        field = Field(
            2,
            [(31.0, 121.0), (31.0, 121.001), (31.001, 121.001), (31.001, 121.0)],
        )
        discretize_field(field, resolution=5.0)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "field_grid.png"
            figure, _ = plot_grid(field, show=False, save_path=str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
