import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.final_evaluation import EvaluationRefused, MODEL_SET, run_final_evaluation


FEATURES = ['age', 'balance']
Y = pd.Series([0, 0, 1, 1, 0, 1], index=[101, 102, 103, 104, 105, 106])
X = pd.DataFrame({'age': [22, 30, 42, 55, 36, 61],
                  'balance': [-20, 100, 300, 1000, 20, 500]}, index=Y.index)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding='utf-8')


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    (experiment / 'run_protocol.json').write_text(
        json.dumps({'test_evaluated': False}), encoding='utf-8')
    for name in MODEL_SET:
        model_dir = experiment / name
        model_dir.mkdir()
        pipeline = LogisticRegression().fit(X, Y)
        joblib.dump(pipeline, model_dir / 'pipeline.joblib')
        _write_json(model_dir / 'model_metadata.json', {
            'model': name, 'refit_partition': 'train', 'test_evaluated': False,
            'classes': [0, 1],
        })
    _write_json(experiment / 'completion.json', {
        'status': 'complete', 'test_evaluated': False, 'models': MODEL_SET,
    })
    protocol = {
        'frozen': True,
        'owner': 'D',
        'frozen_at': '2026-10-18T10:00:00+08:00',
        'decision_rule': 'Thresholds fixed from validation before test evaluation.',
        'feature_policy': 'pre_campaign_history_flag_v1',
        'models': MODEL_SET,
        'thresholds': {name: 0.5 for name in MODEL_SET},
        'pipeline_sha256': {
            name: _digest(experiment / name / 'pipeline.joblib') for name in MODEL_SET
        },
    }
    protocol_path = tmp_path / 'protocol.json'
    _write_json(protocol_path, protocol)
    return experiment, protocol_path, protocol


def _run(experiment, protocol_path, partition_calls, **kwargs):
    def fake_loader(split, *, allow_test=False):
        partition_calls.append((split, allow_test))
        return X.copy(), Y.copy()

    return run_final_evaluation(
        unlock_test=True, protocol_path=protocol_path, experiment_path=experiment,
        load_partition_fn=fake_loader, features=FEATURES, **kwargs)


def test_default_lock_refuses_before_loading_test(tmp_path):
    experiment, protocol_path, _ = _fixture(tmp_path)
    calls = []
    with pytest.raises(EvaluationRefused, match='explicit --unlock-test'):
        run_final_evaluation(unlock_test=False, protocol_path=protocol_path,
                             experiment_path=experiment,
                             load_partition_fn=lambda *args, **kwargs: calls.append(True),
                             features=FEATURES)
    assert calls == []


@pytest.mark.parametrize('mutate, message', [
    (lambda p: p.update(frozen=False), 'frozen to true'),
    (lambda p: p['thresholds'].__setitem__('mlp', float('nan')), 'finite number'),
    (lambda p: p['thresholds'].__setitem__('mlp', 1.01), 'finite number'),
    (lambda p: p['pipeline_sha256'].__setitem__('mlp', '0' * 64), 'SHA256 mismatch'),
])
def test_invalid_protocol_refuses_before_loading_test(tmp_path, mutate, message):
    experiment, protocol_path, protocol = _fixture(tmp_path)
    mutate(protocol)
    _write_json(protocol_path, protocol)
    calls = []
    with pytest.raises(EvaluationRefused, match=message):
        _run(experiment, protocol_path, calls)
    assert calls == []
    assert not (experiment / 'final_test').exists()


def test_completion_marker_cannot_hide_missing_model_artifact(tmp_path):
    experiment, protocol_path, _ = _fixture(tmp_path)
    (experiment / 'random_forest' / 'model_metadata.json').unlink()
    calls = []
    with pytest.raises(EvaluationRefused, match='incomplete'):
        _run(experiment, protocol_path, calls)
    assert calls == []


def test_success_writes_metrics_predictions_and_receipt_from_synthetic_data(tmp_path):
    experiment, protocol_path, _ = _fixture(tmp_path)
    calls = []
    final_dir = _run(experiment, protocol_path, calls)

    assert calls == [('test', True)]
    assert set(path.name for path in final_dir.iterdir()) == {
        'frozen_protocol.json', 'test_metrics.csv', 'test_predictions.csv',
        'evaluation_receipt.json',
    }
    metrics = pd.read_csv(final_dir / 'test_metrics.csv')
    predictions = pd.read_csv(final_dir / 'test_predictions.csv')
    receipt = json.loads((final_dir / 'evaluation_receipt.json').read_text(encoding='utf-8'))
    assert metrics['model'].tolist() == MODEL_SET
    assert predictions['row_id'].tolist() == Y.index.tolist()
    assert predictions['y_true'].tolist() == Y.tolist()
    assert receipt['status'] == 'complete'
    assert receipt['n'] == len(Y)
    assert receipt['thresholds']['dummy'] == 0.5


def test_existing_final_output_refuses_to_overwrite(tmp_path):
    experiment, protocol_path, _ = _fixture(tmp_path)
    final_dir = experiment / 'final_test'
    final_dir.mkdir()
    calls = []
    with pytest.raises(EvaluationRefused, match='refusing to overwrite'):
        _run(experiment, protocol_path, calls)
    assert calls == []
