import math
import pytest

from coverage_env.planner.framework_contracts import (
    RegionTask, normalize_visit_weights, validate_partition,
)


def test_partition_uses_field_scoped_ids():
    validate_partition(
        [RegionTask('a', 'f1', ('p',)), RegionTask('b', 'f2', ('p',))],
        {'f1': ['p'], 'f2': ['p']},
    )


@pytest.mark.parametrize('tasks', [
    [],
    [RegionTask('a', 'f', ('p',))],
    [RegionTask('a', 'f', ('p', 'q')), RegionTask('b', 'f', ('p',))],
    [RegionTask('a', 'f', ('p',)), RegionTask('a', 'f', ('q',))],
    [RegionTask('a', 'other', ('p', 'q'))],
    [RegionTask('a', 'f', ('p', 'unknown'))],
])
def test_invalid_partitions_rejected(tasks):
    with pytest.raises(ValueError):
        validate_partition(tasks, {'f': ['p', 'q']})


@pytest.mark.parametrize('points', [(), ('p', 'p'), ['p'], ('',)])
def test_region_rejects_empty_duplicate_or_mutable_ids(points):
    with pytest.raises(ValueError):
        RegionTask('a', 'f', points)


def test_invalid_reference_grid_rejected():
    for reference in ({}, {'f': []}, {'f': ['p', 'p']}):
        with pytest.raises(ValueError):
            validate_partition([], reference)


def test_weights_are_scale_invariant_and_handle_large_finite_inputs():
    assert normalize_visit_weights([1, 3]) == (0.25, 0.75)
    assert normalize_visit_weights([1e308, 1e308]) == (0.5, 0.5)
    assert sum(normalize_visit_weights([1, 2, 7])) == pytest.approx(1)


@pytest.mark.parametrize('weights', [[], [0], [-1, 2], [math.nan], [math.inf], [1e-320, 1e308]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        normalize_visit_weights(weights)
