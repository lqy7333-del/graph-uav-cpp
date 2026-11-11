from dataclasses import asdict
from hashlib import sha256
import json
import pickle
from unittest.mock import patch
import pytest

from coverage_env.tests.test_framework_execution import make_env
from coverage_env.tests.test_divide_then_plan import config
from coverage_env.planner.coverage_plan import fingerprints
from coverage_env.rl.evaluation.framework_comparison import load_panel, run_panel, paired_summary, prepare_manifest


def panel(tmp_path):
    env = make_env(); raw = pickle.dumps(env)
    (tmp_path/'instance.pkl').write_bytes(raw)
    manifest = dict(schema_version=1, physical_config_fingerprint=fingerprints(env, env.fields[0])[1],
        instances=[dict(id='one', fields=1, snapshot='instance.pkl', sha256=sha256(raw).hexdigest())])
    path = tmp_path/'panel.json'; path.write_text(json.dumps(manifest))
    return path


def test_shared_panel_queue_and_resume_verification(tmp_path):
    source = panel(tmp_path); output = tmp_path/'run'
    rows = run_panel(source, output, config())
    assert len(rows) == 3 and all(r['completed'] for r in rows)
    # One-instance smoke must not unlock formal experiments.
    assert not json.loads((output/'acceptance_receipt.json').read_text())['passed']
    with patch('coverage_env.baseline.nested_ga.NestedGA.run', side_effect=AssertionError('no rerun')):
        assert run_panel(source, output, config(), resume=True) == rows
    saved = output/'0000_plan_then_divide.json'
    data = json.loads(saved.read_text()); data['result']['best']['cost'] += 10
    saved.write_text(json.dumps(data))
    with pytest.raises(ValueError): run_panel(source, output, config(), resume=True)


def test_manifest_hash_and_formal_gate(tmp_path):
    source = panel(tmp_path)
    with pytest.raises(ValueError, match='receipt'):
        run_panel(source, tmp_path/'formal', config(), stage='formal')
    (tmp_path/'instance.pkl').write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash'): load_panel(source)


def test_prepare_refuses_duplicate_geometry_and_overwrite(tmp_path):
    source = panel(tmp_path)
    raw = tmp_path/'instance.pkl'
    prepare_manifest([raw], tmp_path/'prepared.json')
    with pytest.raises(FileExistsError): prepare_manifest([raw], tmp_path/'prepared.json')
    with pytest.raises(ValueError): prepare_manifest([raw, raw], tmp_path/'duplicates.json')
    assert not (tmp_path/'duplicates.json').exists()


def test_resume_budget_change_is_rejected(tmp_path):
    source = panel(tmp_path); out = tmp_path/'run'; out.mkdir()
    (out/'run_contract.json').write_text('{}')
    with pytest.raises(ValueError, match='contract mismatch'):
        run_panel(source, out, config(), resume=True)


def test_wrong_acceptance_receipt_cannot_start_formal(tmp_path):
    source = panel(tmp_path)
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(passed=True, domain_verified=True, code_digest='wrong',
                                       physical_config_fingerprint='wrong')))
    with pytest.raises(ValueError, match='compatible'):
        run_panel(source, tmp_path/'formal', config(), stage='formal', receipt=receipt)
    assert not (tmp_path/'formal').exists()


def test_failure_not_removed_from_statistics():
    rows = [dict(id=i, fields=10, method=m, completed=not(i==0 and m=='plan_then_divide'),
                 total_time=100. if m=='cc' else 110., search_seconds=1.)
            for i in range(3) for m in ('cc', 'divide_then_plan', 'plan_then_divide')]
    report = paired_summary(rows)['10']
    assert report['methods']['plan_then_divide']['mean_total_time'] is None
    assert report['comparisons']['plan_then_divide minus cc']['n_pairs'] == 2
    assert 'auxiliary' in report['comparisons']['plan_then_divide minus cc']['scope']
