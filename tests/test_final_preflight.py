"""Fail-closed preflight checks use synthetic bytes and inference-only models."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.final_evaluation import (
    ANALYSIS_CONFIG, INPUT_PATHS, MODEL_SET, PROPOSED_PROTOCOL,
    VALIDATION_RECEIPT, EvaluationRefused, run_final_evaluation,
)


THRESHOLD_OUTPUT = 'results/member_d/evaluation/thresholds.json'


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_both_anchors(case, field, value):
    """Exercise internal consistency after updating synthetic outer identities."""
    for path in [case['protocol_path'], case['root'] / PROPOSED_PROTOCOL]:
        document = _read(path)
        document[field] = value
        _write(path, document)


def _rewrite_receipt(case, mutate):
    path = case['root'] / VALIDATION_RECEIPT
    receipt = _read(path)
    mutate(receipt)
    _write(path, receipt)
    _set_both_anchors(case, 'validation_receipt_sha256', _digest(path))


def _refuses_before_loading(case, match=None):
    partition_calls, model_calls = [], []

    def partition_loader(*args, **kwargs):
        partition_calls.append((args, kwargs))
        raise AssertionError('Preflight must refuse before partition loading')

    def model_loader(path):
        model_calls.append(path)
        raise AssertionError('Preflight must refuse before model loading')

    with pytest.raises(EvaluationRefused, match=match):
        run_final_evaluation(
            unlock_test=True, project_root=case['root'],
            experiment_path=case['experiment'], protocol_path=case['protocol_path'],
            load_partition_fn=partition_loader, features=['age'], model_loader=model_loader)
    assert partition_calls == []
    assert model_calls == []
    assert not (case['experiment'] / 'final_test').exists()


@pytest.mark.parametrize('relative', INPUT_PATHS)
def test_each_frozen_input_byte_mismatch_refuses(synthetic_final_case, relative):
    path = synthetic_final_case['root'] / relative
    path.write_bytes(path.read_bytes() + b'\nchanged synthetic identity\n')
    _refuses_before_loading(synthetic_final_case, 'SHA256 mismatch')


@pytest.mark.parametrize('name', MODEL_SET)
def test_each_frozen_pipeline_byte_mismatch_refuses(synthetic_final_case, name):
    path = synthetic_final_case['experiment'] / name / 'pipeline.joblib'
    path.write_bytes(path.read_bytes() + b'changed synthetic pipeline')
    _refuses_before_loading(synthetic_final_case, 'Pipeline SHA256 mismatch')


@pytest.mark.parametrize('relative', [
    PROPOSED_PROTOCOL, ANALYSIS_CONFIG, VALIDATION_RECEIPT, THRESHOLD_OUTPUT,
])
@pytest.mark.parametrize('failure', ['missing', 'malformed'])
def test_required_evidence_missing_or_malformed_refuses(
        synthetic_final_case, relative, failure):
    path = synthetic_final_case['root'] / relative
    if failure == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'{invalid synthetic JSON')
    _refuses_before_loading(synthetic_final_case, 'Cannot read valid JSON')


@pytest.mark.parametrize('status', [None, False, 'pending_explicit_user_approval', 'APPROVED'])
def test_explicit_approval_status_is_required(synthetic_final_case, status):
    path = synthetic_final_case['protocol_path']
    protocol = _read(path)
    if status is None:
        protocol.pop('approval_status')
    else:
        protocol['approval_status'] = status
    _write(path, protocol)
    _refuses_before_loading(synthetic_final_case, 'approval_status')


def test_ambiguous_duplicate_approval_key_refuses(synthetic_final_case):
    path = synthetic_final_case['protocol_path']
    document = path.read_text(encoding='utf-8')
    document = document.replace('"approval_status": "approved"',
                                '"approval_status": "pending", "approval_status": "approved"')
    path.write_text(document, encoding='utf-8')
    _refuses_before_loading(synthetic_final_case, 'Duplicate JSON key')


def test_invalid_unicode_evidence_refuses(synthetic_final_case):
    (synthetic_final_case['root'] / VALIDATION_RECEIPT).write_bytes(b'\xff\xfe\xff')
    _refuses_before_loading(synthetic_final_case, 'Cannot read valid JSON')


@pytest.mark.parametrize('field,value', [
    ('decision_rule', 'A different threshold rule'),
    ('primary_metric', 'roc_auc'),
    ('report_metrics', ['average_precision', 'f1']),
    ('cv_recommended_family', 'logistic_regression'),
    ('validation_analysis_config_sha256', '0' * 64),
    ('validation_receipt_sha256', '0' * 64),
])
def test_final_protocol_semantics_must_equal_reviewed_proposal(
        synthetic_final_case, field, value):
    path = synthetic_final_case['protocol_path']
    protocol = _read(path)
    protocol[field] = value
    _write(path, protocol)
    _refuses_before_loading(synthetic_final_case, 'differs from reviewed proposal')


def test_final_threshold_cannot_drift_from_reviewed_proposal(synthetic_final_case):
    path = synthetic_final_case['protocol_path']
    protocol = _read(path)
    protocol['thresholds']['mlp'] = 0.75
    _write(path, protocol)
    _refuses_before_loading(synthetic_final_case, 'differs from reviewed proposal: thresholds')


def test_reviewed_proposal_must_contain_required_experiment_fields(synthetic_final_case):
    path = synthetic_final_case['root'] / PROPOSED_PROTOCOL
    proposal = _read(path)
    proposal.pop('thresholds')
    _write(path, proposal)
    _refuses_before_loading(synthetic_final_case, 'differs from reviewed proposal: thresholds')


@pytest.mark.parametrize('change', ['missing', 'unexpected'])
def test_input_identity_map_requires_exact_fixed_paths(synthetic_final_case, change):
    inputs = _read(synthetic_final_case['protocol_path'])['input_sha256']
    if change == 'missing':
        inputs.pop(INPUT_PATHS[0])
    else:
        inputs['data/unapproved.csv'] = '0' * 64
    _set_both_anchors(synthetic_final_case, 'input_sha256', inputs)
    _refuses_before_loading(synthetic_final_case, 'exactly the three frozen inputs')


@pytest.mark.parametrize('relative', [ANALYSIS_CONFIG, VALIDATION_RECEIPT])
def test_valid_json_with_unapproved_byte_identity_refuses(synthetic_final_case, relative):
    path = synthetic_final_case['root'] / relative
    path.write_bytes(path.read_bytes() + b'\n')
    _refuses_before_loading(synthetic_final_case, 'SHA256 mismatch')


@pytest.mark.parametrize('field,value', [
    ('partition', 'test'),
    ('test_rows_parsed', True),
    ('test_predictions_or_metrics_computed', True),
    ('fit_methods_blocked', False),
    ('analysis_config_sha256', '0' * 64),
])
def test_authenticated_receipt_must_remain_semantically_consistent(
        synthetic_final_case, field, value):
    _rewrite_receipt(synthetic_final_case, lambda receipt: receipt.__setitem__(field, value))
    _refuses_before_loading(synthetic_final_case)


@pytest.mark.parametrize('field,value', [
    ('protected_files_before', None), ('outputs_sha256', []), ('thresholds', None),
])
def test_authenticated_receipt_must_contain_required_identity_objects(
        synthetic_final_case, field, value):
    _rewrite_receipt(synthetic_final_case, lambda receipt: receipt.__setitem__(field, value))
    _refuses_before_loading(synthetic_final_case)


@pytest.mark.parametrize('protected_path', [
    INPUT_PATHS[0], 'results/experiment/mlp/pipeline.joblib',
])
def test_receipt_must_bind_approved_input_and_model_hashes(
        synthetic_final_case, protected_path):
    _rewrite_receipt(synthetic_final_case, lambda receipt:
                     receipt['protected_files_before'].__setitem__(protected_path, '0' * 64))
    _refuses_before_loading(synthetic_final_case, 'SHA256 mismatch')


def test_receipt_thresholds_cannot_disagree_with_approved_protocol(synthetic_final_case):
    _rewrite_receipt(synthetic_final_case, lambda receipt:
                     receipt['thresholds'].__setitem__('mlp', 0.75))
    _refuses_before_loading(synthetic_final_case, 'thresholds disagree with validation receipt')


@pytest.mark.parametrize('relative', [
    'completion.json', 'run_protocol.json', 'mlp/model_metadata.json',
])
def test_experiment_metadata_bytes_must_match_validation_receipt(
        synthetic_final_case, relative):
    path = synthetic_final_case['experiment'] / relative
    path.write_bytes(path.read_bytes() + b'\n')
    _refuses_before_loading(synthetic_final_case, 'SHA256 mismatch')


@pytest.mark.parametrize('update_receipt', [False, True])
def test_selected_threshold_output_requires_matching_identity_and_values(
        synthetic_final_case, update_receipt):
    path = synthetic_final_case['root'] / THRESHOLD_OUTPUT
    selected = _read(path)
    selected['mlp'] = 0.75
    _write(path, selected)
    if update_receipt:
        _rewrite_receipt(synthetic_final_case, lambda receipt:
                         receipt['outputs_sha256'].__setitem__('thresholds.json', _digest(path)))
        message = 'thresholds disagree with the validation threshold output'
    else:
        message = 'SHA256 mismatch for validation thresholds.json'
    _refuses_before_loading(synthetic_final_case, message)


@pytest.mark.parametrize('field,value', [('classification_rule', 'p_yes > threshold'),
                                         ('retraining_allowed', True)])
def test_authenticated_analysis_configuration_must_match_evaluation_semantics(
        synthetic_final_case, field, value):
    path = synthetic_final_case['root'] / ANALYSIS_CONFIG
    configuration = _read(path)
    configuration[field] = value
    _write(path, configuration)
    digest = _digest(path)
    _set_both_anchors(synthetic_final_case, 'validation_analysis_config_sha256', digest)
    _rewrite_receipt(synthetic_final_case, lambda receipt:
                     receipt.__setitem__('analysis_config_sha256', digest))
    _refuses_before_loading(synthetic_final_case)


def test_success_preserves_verified_protocol_snapshot_thresholds_and_one_shot(synthetic_final_case):
    case = synthetic_final_case
    verified_bytes = case['protocol_path'].read_bytes()
    expected_thresholds = json.loads(verified_bytes)['thresholds']
    X = pd.DataFrame({'age': [20, 30, 40, 50]}, index=[101, 102, 103, 104])
    y = pd.Series([0, 1, 0, 1], index=X.index)
    p_yes = np.array([0.05, 0.25, 0.65, 0.95])
    partition_calls, model_calls = [], []

    class InferenceOnly:
        classes_ = np.array([0, 1])

        def fit(self, *args, **kwargs):
            pytest.fail('Final evaluation must never fit a model')

        def predict_proba(self, frame):
            assert frame.equals(X)
            return np.column_stack([1 - p_yes, p_yes])

    def partition_loader(name, *, allow_test=False):
        partition_calls.append((name, allow_test))
        # A caller changing the file after preflight must not change the evidence
        # saved for the operation that was already validated.
        mutated = _read(case['protocol_path'])
        mutated['thresholds']['mlp'] = 0.99
        _write(case['protocol_path'], mutated)
        return X.copy(), y.copy()

    def model_loader(path):
        model_calls.append(path.parent.name)
        return InferenceOnly()

    arguments = dict(
        unlock_test=True, project_root=case['root'], experiment_path=case['experiment'],
        protocol_path=case['protocol_path'], load_partition_fn=partition_loader,
        features=['age'], model_loader=model_loader)
    final_dir = run_final_evaluation(**arguments)
    assert partition_calls == [('test', True)]
    assert model_calls == MODEL_SET
    assert (final_dir / 'frozen_protocol.json').read_bytes() == verified_bytes
    predictions = pd.read_csv(final_dir / 'test_predictions.csv')
    receipt = _read(final_dir / 'evaluation_receipt.json')
    metrics = pd.read_csv(final_dir / 'test_metrics.csv', float_precision='round_trip')
    assert receipt['thresholds'] == expected_thresholds
    assert receipt['protocol_sha256'] == hashlib.sha256(verified_bytes).hexdigest()
    assert receipt['preflight']['status'] == 'passed'
    for name in MODEL_SET:
        np.testing.assert_array_equal(predictions[f'{name}_prediction'],
                                      (p_yes >= expected_thresholds[name]).astype(int))
        assert metrics.set_index('model').loc[name, 'threshold'] == expected_thresholds[name]

    # Restore the synthetic approved source solely to reach the one-shot gate.
    case['protocol_path'].write_bytes(verified_bytes)
    prior_receipt = (final_dir / 'evaluation_receipt.json').read_bytes()
    with pytest.raises(EvaluationRefused, match='refusing to overwrite'):
        run_final_evaluation(**arguments)
    assert partition_calls == [('test', True)]
    assert model_calls == MODEL_SET
    assert (final_dir / 'evaluation_receipt.json').read_bytes() == prior_receipt
