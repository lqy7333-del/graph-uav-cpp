"""M1 payload capacity checks: applied capacity, completeness, replay metadata."""
from pathlib import Path
import json
import pytest
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/m1_payload/results')
pytestmark=pytest.mark.skipif(not BASE.exists(),reason='M1 results not present')

def test_all_capacity_cells_present_and_complete():
    rows=[]
    for p in sorted(BASE.glob('instance_*.json')):
        d=json.load(open(p)); s=d['summary']; rows.append(s)
    assert len(rows)==27
    assert all(r['completed'] and r['failure'] is None for r in rows)
    for seed in (1970000,1970001,1970002):
        for factor,capacity in ((0.5,25.0),(1.0,50.0),(2.0,100.0)):
            cells=[r for r in rows if r['seed']==seed and r['payload_factor']==factor]
            assert len(cells)==3
            assert all(abs(r['capacity']-capacity)<1e-9 for r in cells)
    assert all(r['search_seconds']<=1810 for r in rows)

def test_capacity_run_result_is_present_and_replayable_metadata():
    d=json.load(open(BASE/'instance_1970000_p0.5_cc.json')); s=d['summary']; r=d['result']
    assert s['capacity']==25.0
    assert r is not None and r['best'] and r['best']['completed']
    assert r['best']['summary']['total_time']==pytest.approx(s['cost'])
    assert s['snapshot_sha256']
