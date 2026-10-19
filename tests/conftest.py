"""Synthetic approval evidence; never reads the real dataset or trained models."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.final_evaluation import MODEL_SET


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_approval():
    def prepare(root, experiment, protocol):
        inputs = ['data/raw/bank-full.csv', 'data/splits/split_indices.json',
                  'data/splits/split_manifest.csv']
        for relative in inputs:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'SYNTHETIC identity fixture: {relative}\n'.encode())
        config = {
            'schema_version': 1, 'partition': 'val', 'positive_class': 1,
            'label_mapping': {'no': 0, 'yes': 1},
            'classification_rule': 'p_yes >= threshold',
            'threshold_objective': 'maximum unweighted validation F1',
            'dummy_threshold': 0.5, 'primary_ranking_metric': 'average_precision',
            'test_evaluated': False, 'retraining_allowed': False,
        }
        _write(root / 'config/member_d_analysis.json', config)
        completion = json.loads((experiment / 'completion.json').read_bytes())
        completion['cv_recommended_family'] = 'random_forest'
        _write(experiment / 'completion.json', completion)
        protected = {relative: _digest(root / relative) for relative in inputs}
        for path in experiment.rglob('*'):
            if path.is_file():
                protected['results/experiment/' + path.relative_to(experiment).as_posix()] = _digest(path)
        selected_path = root / 'results/member_d/evaluation/thresholds.json'
        _write(selected_path, protocol['thresholds'])
        receipt = {
            'status': 'complete', 'stage': 'evaluation', 'partition': 'val',
            'positive_class': 1, 'thresholds': deepcopy(protocol['thresholds']),
            'analysis_config_sha256': _digest(root / 'config/member_d_analysis.json'),
            'protected_files_before': protected,
            'protected_files_unchanged': True, 'in_memory_models_unchanged': True,
            'fit_methods_blocked': True, 'test_rows_parsed': False,
            'test_predictions_or_metrics_computed': False,
            'outputs_sha256': {'thresholds.json': _digest(selected_path)},
        }
        receipt_path = root / 'results/member_d/evaluation/receipt.json'
        _write(receipt_path, receipt)
        protocol.update(
            approval_status='approved', primary_metric='average_precision',
            report_metrics=['n', 'positive_rate', 'threshold', 'average_precision',
                            'roc_auc', 'accuracy', 'balanced_accuracy', 'precision',
                            'recall', 'f1', 'brier_score', 'log_loss', 'tn', 'fp', 'fn', 'tp'],
            cv_recommended_family='random_forest',
            validation_analysis_config_sha256=receipt['analysis_config_sha256'],
            validation_receipt_sha256=_digest(receipt_path),
            input_sha256={key: protected[key] for key in inputs},
        )
        proposal = deepcopy(protocol)
        proposal.update(frozen=False, owner='', frozen_at='',
                        approval_status='pending_explicit_user_approval')
        _write(root / 'config/final_protocol.proposed.json', proposal)
        return protocol
    return prepare


@pytest.fixture
def synthetic_final_case(tmp_path, synthetic_approval):
    root = tmp_path / 'synthetic_project'
    experiment = root / 'results/experiment'
    hashes = {}
    for name in MODEL_SET:
        folder = experiment / name
        folder.mkdir(parents=True)
        (folder / 'pipeline.joblib').write_bytes(f'SYNTHETIC unfitted bytes: {name}'.encode())
        hashes[name] = _digest(folder / 'pipeline.joblib')
        _write(folder / 'model_metadata.json', {
            'model': name, 'refit_partition': 'train', 'test_evaluated': False, 'classes': [0, 1],
        })
    _write(experiment / 'completion.json', {
        'status': 'complete', 'test_evaluated': False, 'models': MODEL_SET,
    })
    _write(experiment / 'run_protocol.json', {'test_evaluated': False})
    protocol = {
        'frozen': True, 'owner': 'synthetic test fixture only',
        'frozen_at': '2026-10-19T00:00:00+00:00',
        'decision_rule': 'Synthetic validation F1 rule; inclusive threshold.',
        'feature_policy': 'pre_campaign_history_flag_v1', 'models': list(MODEL_SET),
        'thresholds': dict(zip(MODEL_SET, [.5, .17, .26, .61, .19])),
        'pipeline_sha256': hashes,
    }
    synthetic_approval(root, experiment, protocol)
    protocol_path = root / 'synthetic_approved_protocol.json'
    _write(protocol_path, protocol)
    return {'root': root, 'experiment': experiment,
            'protocol_path': protocol_path, 'protocol': protocol}
