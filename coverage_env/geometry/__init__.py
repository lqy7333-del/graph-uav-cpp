"""Coordinate, polygon, and discretization utilities."""

from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import GridPoint, discretize_field

__all__ = ["CoordinateTransformer", "GridPoint", "discretize_field"]
