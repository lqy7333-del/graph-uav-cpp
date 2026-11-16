"""M2 payload endpoint checks: 120 cells, actual capacities, no M1 maps."""
from pathlib import Path
import json
import pytest
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/m2_payload/results')
pytestmark=pytest.mark.skipif(not BASE.exists(),reason='M2 results not present')

def _rows():
    return [json.load(open(p))['summary'] for p in BASE.glob('instance_*.json')]

def test_m2_all_cells_capacity_and_completion():
    rows=_rows(); assert len(rows)==120
    assert all(r['completed'] and r['failure'] is None for r in rows)
    assert len({r['seed'] for r in rows})==20
    assert all(r['capacity'] in (25.0,100.0) for r in rows)
    assert all(r['seed'] not in (1970000,1970001,1970002) for r in rows)
    for seed in {r['seed'] for r in rows}:
        for cap in (25.0,100.0):
            cells=[r for r in rows if r['seed']==seed and r['capacity']==cap]
            assert len(cells)==3
    assert all(r['search_seconds']<=1810 for r in rows)

def test_m2_saved_result_matches_summary():
    d=json.load(open(BASE/'instance_1980000_c25_cc.json')); s=d['summary']; r=d['result']
    assert s['capacity']==25.0 and r['best']['completed']
    assert r['best']['summary']['total_time']==pytest.approx(s['cost'])
    assert s['snapshot_sha256']
