"""One-shot evaluation of frozen development pipelines on the test partition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.metrics import binary_metrics


# This final gate intentionally knows only the frozen family names, so importing it
# does not require the development-only preprocessing implementation.
MODEL_SET = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'mlp']
FEATURE_POLICY = 'pre_campaign_history_flag_v1'
SHA256_RE = re.compile(r'^[0-9a-fA-F]{64}$')
INPUT_PATHS = (
    'data/raw/bank-full.csv',
    'data/splits/split_indices.json',
    'data/splits/split_manifest.csv',
)
ANALYSIS_CONFIG = 'config/member_d_analysis.json'
VALIDATION_RECEIPT = 'results/member_d/evaluation/receipt.json'
PROPOSED_PROTOCOL = 'config/final_protocol.proposed.json'
REPORT_METRICS = [
    'n', 'positive_rate', 'threshold', 'average_precision', 'roc_auc', 'accuracy',
    'balanced_accuracy', 'precision', 'recall', 'f1', 'brier_score', 'log_loss',
    'tn', 'fp', 'fn', 'tp',
]
EXPERIMENT_FIELDS = (
    'decision_rule', 'feature_policy', 'models', 'thresholds', 'pipeline_sha256',
    'primary_metric', 'report_metrics', 'cv_recommended_family',
    'validation_analysis_config_sha256', 'validation_receipt_sha256', 'input_sha256',
)


class EvaluationRefused(ValueError):
    """Raised when a final-test precondition is not satisfied."""


def _read_json_snapshot(path: Path) -> tuple[dict, bytes]:
    """Parse exactly the bytes whose identity will be checked or saved."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationRefused(f'Duplicate JSON key {key!r} in {path}')
            result[key] = value
        return result

    try:
        content = path.read_bytes()
        value = json.loads(content, object_pairs_hook=unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationRefused(f'Cannot read valid JSON from {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise EvaluationRefused(f'Expected a JSON object in {path}')
    return value, content


def _read_json(path: Path) -> dict:
    return _read_json_snapshot(path)[0]


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_protocol(protocol: dict) -> dict[str, float]:
    if protocol.get('frozen') is not True:
        raise EvaluationRefused('Protocol must set frozen to true')
    if protocol.get('approval_status') != 'approved':
        raise EvaluationRefused('Protocol approval_status must explicitly equal "approved"')
    if not _nonblank(protocol.get('owner')):
        raise EvaluationRefused('Protocol owner must be nonblank')
    frozen_at = protocol.get('frozen_at')
    if not _nonblank(frozen_at):
        raise EvaluationRefused('Protocol frozen_at must be a nonblank ISO8601 timestamp')
    try:
        parsed_at = datetime.fromisoformat(frozen_at.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise EvaluationRefused('Protocol frozen_at must be a valid ISO8601 timestamp') from exc
    if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise EvaluationRefused('Protocol frozen_at must include a timezone')
    if not _nonblank(protocol.get('decision_rule')):
        raise EvaluationRefused('Protocol decision_rule must be nonblank')
    if protocol.get('feature_policy') != FEATURE_POLICY:
        raise EvaluationRefused(f'Protocol feature_policy must equal {FEATURE_POLICY!r}')
    if protocol.get('models') != MODEL_SET:
        raise EvaluationRefused(f'Protocol models must be exactly {MODEL_SET!r} in this order')

    thresholds = protocol.get('thresholds')
    if not isinstance(thresholds, dict) or set(thresholds) != set(MODEL_SET):
        raise EvaluationRefused(f'Protocol thresholds must contain exactly {MODEL_SET!r}')
    normalized_thresholds = {}
    for name in MODEL_SET:
        value = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationRefused(f'Threshold for {name} must be a finite number in [0, 1]')
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise EvaluationRefused(f'Threshold for {name} must be a finite number in [0, 1]')
        normalized_thresholds[name] = float(value)
    if normalized_thresholds['dummy'] != 0.5:
        raise EvaluationRefused('Dummy threshold is fixed at 0.5')

    hashes = protocol.get('pipeline_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(MODEL_SET):
        raise EvaluationRefused(f'Protocol pipeline_sha256 must contain exactly {MODEL_SET!r}')
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise EvaluationRefused(f'Protocol SHA256 for {name} must be 64 hexadecimal characters')
    return normalized_thresholds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise EvaluationRefused(f'Cannot hash required identity {path}: {exc}') from exc
    return digest.hexdigest()


def _require_digest(actual: str, expected, label: str) -> None:
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise EvaluationRefused(f'Missing or invalid SHA256 for {label}')
    if actual.lower() != expected.lower():
        raise EvaluationRefused(f'SHA256 mismatch for {label}')


def _validate_approved_identities(root: Path, experiment: Path, protocol: dict) -> dict:
    """Authenticate the validation evidence without parsing any raw data rows.

    The reviewed proposal is the local approval baseline. Freezing may change only
    authorization fields and notes; experiment identities/settings stay identical.
    Approval is a recorded human decision, not a cryptographic authentication system.
    """
    proposal, proposal_bytes = _read_json_snapshot(root / PROPOSED_PROTOCOL)
    allowed_fields = set(EXPERIMENT_FIELDS) | {
        'frozen', 'owner', 'frozen_at', 'approval_status', 'notes',
    }
    if set(protocol) - allowed_fields:
        raise EvaluationRefused('Protocol contains unsupported fields')
    for field in EXPERIMENT_FIELDS:
        if field not in proposal or field not in protocol or protocol[field] != proposal[field]:
            raise EvaluationRefused(f'Protocol differs from reviewed proposal: {field}')

    inputs = protocol['input_sha256']
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_PATHS):
        raise EvaluationRefused('Protocol input_sha256 must contain exactly the three frozen inputs')
    for relative in INPUT_PATHS:
        # Opaque byte hashing only: no CSV parsing, labels or partition selection.
        _require_digest(_sha256(root / relative), inputs[relative], relative)

    config, config_bytes = _read_json_snapshot(root / ANALYSIS_CONFIG)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    _require_digest(config_digest, protocol['validation_analysis_config_sha256'], ANALYSIS_CONFIG)
    required_config = {
        'schema_version': 1, 'partition': 'val', 'positive_class': 1,
        'label_mapping': {'no': 0, 'yes': 1},
        'classification_rule': 'p_yes >= threshold',
        'threshold_objective': 'maximum unweighted validation F1',
        'dummy_threshold': 0.5, 'primary_ranking_metric': 'average_precision',
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise EvaluationRefused(f'Validation analysis configuration is inconsistent: {key}')
    if config.get('test_evaluated') is not False or config.get('retraining_allowed') is not False:
        raise EvaluationRefused('Validation configuration must prohibit test evaluation and retraining')
    if protocol['primary_metric'] != config['primary_ranking_metric']:
        raise EvaluationRefused('Protocol primary metric disagrees with validation configuration')
    if protocol['report_metrics'] != REPORT_METRICS:
        raise EvaluationRefused('Protocol report_metrics must match the complete final metric schema')

    receipt, receipt_bytes = _read_json_snapshot(root / VALIDATION_RECEIPT)
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    _require_digest(receipt_digest, protocol['validation_receipt_sha256'], VALIDATION_RECEIPT)
    for key, expected in {'status': 'complete', 'stage': 'evaluation', 'partition': 'val',
                          'positive_class': 1}.items():
        if receipt.get(key) != expected:
            raise EvaluationRefused(f'Validation evaluation receipt is inconsistent: {key}')
    for key, expected in {
        'protected_files_unchanged': True, 'in_memory_models_unchanged': True,
        'fit_methods_blocked': True, 'test_rows_parsed': False,
        'test_predictions_or_metrics_computed': False,
    }.items():
        if receipt.get(key) is not expected:
            raise EvaluationRefused(f'Validation evaluation receipt must certify {key}={expected}')
    _require_digest(config_digest, receipt.get('analysis_config_sha256'), 'receipt analysis configuration')
    if receipt.get('thresholds') != protocol['thresholds']:
        raise EvaluationRefused('Protocol thresholds disagree with validation receipt')

    protected = receipt.get('protected_files_before')
    if not isinstance(protected, dict):
        raise EvaluationRefused('Validation receipt must contain protected input/model identities')
    for relative in INPUT_PATHS:
        _require_digest(inputs[relative], protected.get(relative), f'receipt {relative}')
    # Bind metadata as well as model bytes to the actual validation experiment.
    experiment_files = ['completion.json', 'run_protocol.json']
    for name in MODEL_SET:
        key = f'results/experiment/{name}/pipeline.joblib'
        _require_digest(protocol['pipeline_sha256'][name], protected.get(key), f'receipt {key}')
        experiment_files.append(f'{name}/model_metadata.json')
    for relative in experiment_files:
        key = f'results/experiment/{relative}'
        _require_digest(_sha256(experiment / relative), protected.get(key), key)
    completion = _read_json(experiment / 'completion.json')
    if protocol['cv_recommended_family'] not in MODEL_SET[1:] or (
            completion.get('cv_recommended_family') != protocol['cv_recommended_family']):
        raise EvaluationRefused('Protocol CV family disagrees with the validation experiment')

    selected, selected_bytes = _read_json_snapshot(
        root / 'results/member_d/evaluation/thresholds.json')
    outputs = receipt.get('outputs_sha256')
    if not isinstance(outputs, dict):
        raise EvaluationRefused('Validation receipt must identify its threshold output')
    _require_digest(hashlib.sha256(selected_bytes).hexdigest(), outputs.get('thresholds.json'),
                    'validation thresholds.json')
    if selected != protocol['thresholds']:
        raise EvaluationRefused('Protocol thresholds disagree with the validation threshold output')
    return {
        'status': 'passed', 'approval_status': protocol['approval_status'],
        'proposal_sha256': hashlib.sha256(proposal_bytes).hexdigest(),
        'input_sha256': dict(inputs), 'pipeline_sha256': dict(protocol['pipeline_sha256']),
        'validation_analysis_config_sha256': config_digest,
        'validation_receipt_sha256': receipt_digest,
        'validation_thresholds_sha256': hashlib.sha256(selected_bytes).hexdigest(),
    }


def _validate_experiment(experiment: Path, protocol: dict) -> None:
    completion = _read_json(experiment / 'completion.json')
    if completion.get('status') != 'complete':
        raise EvaluationRefused('Experiment completion.json must have status="complete"')
    if completion.get('test_evaluated') is not False:
        raise EvaluationRefused('Experiment completion.json must confirm test_evaluated=false')
    if completion.get('models') != MODEL_SET:
        raise EvaluationRefused(f'Experiment completion.json models must be exactly {MODEL_SET!r}')

    run_protocol = _read_json(experiment / 'run_protocol.json')
    if run_protocol.get('test_evaluated') is not False:
        raise EvaluationRefused('Experiment run_protocol.json must confirm test_evaluated=false')

    for name in MODEL_SET:
        model_dir = experiment / name
        pipeline = model_dir / 'pipeline.joblib'
        metadata_path = model_dir / 'model_metadata.json'
        if not pipeline.is_file() or not metadata_path.is_file():
            raise EvaluationRefused(f'Experiment is incomplete: missing pipeline or metadata for {name}')
        metadata = _read_json(metadata_path)
        if metadata.get('model') != name:
            raise EvaluationRefused(f'Metadata model identity does not match directory {name}')
        if metadata.get('refit_partition') != 'train' or metadata.get('test_evaluated') is not False:
            raise EvaluationRefused(f'Metadata for {name} does not certify a train-only, unevaluated pipeline')
        if metadata.get('classes') != [0, 1]:
            raise EvaluationRefused(f'Metadata for {name} must declare binary classes [0, 1]')
        actual_digest = _sha256(pipeline)
        if actual_digest.lower() != protocol['pipeline_sha256'][name].lower():
            raise EvaluationRefused(f'Pipeline SHA256 mismatch for {name}')


def run_final_evaluation(*, unlock_test: bool, protocol_path, experiment_path=None,
                         load_partition_fn=None, features=None, model_loader=None,
                         project_root=None):
    """Validate frozen artifacts, then evaluate each saved pipeline exactly once.

    Optional loader/features/model_loader/project_root arguments support synthetic tests;
    production defaults are imported lazily so preflight never touches the test data.
    """
    if unlock_test is not True:
        raise EvaluationRefused('Test evaluation requires explicit --unlock-test authorization')
    if protocol_path is None:
        raise EvaluationRefused('A frozen protocol path is required')
    protocol_path = Path(protocol_path)
    protocol, protocol_bytes = _read_json_snapshot(protocol_path)
    thresholds = _validate_protocol(protocol)

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    if experiment_path is None:
        experiment_path = root / 'results/experiment'
    experiment = Path(experiment_path)
    if not experiment.is_dir():
        raise EvaluationRefused(f'Experiment directory does not exist: {experiment}')
    _validate_experiment(experiment, protocol)
    preflight = _validate_approved_identities(root, experiment, protocol)

    final_dir = experiment / 'final_test'
    if final_dir.exists():
        raise EvaluationRefused(f'Final evaluation output already exists; refusing to overwrite: {final_dir}')

    # Create the fixed destination only after every preflight gate. If inference or
    # writing fails, leave it in place so a later invocation cannot silently replace it.
    final_dir.mkdir()

    if load_partition_fn is None or features is None:
        from src.data import FEATURES, load_partition
        if load_partition_fn is None:
            def load_partition_fn(name, *, allow_test=False):
                return load_partition(name, allow_test=allow_test, root=root)
        if features is None:
            features = FEATURES
    if model_loader is None:
        model_loader = joblib.load

    X_test, y_test = load_partition_fn('test', allow_test=True)
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError('Test features must be a pandas DataFrame')
    if list(X_test.columns) != list(features):
        raise ValueError('Test feature columns do not match the frozen FEATURES order')
    if not X_test.index.is_unique:
        raise ValueError('Test row_id values must be unique')
    if not isinstance(y_test, pd.Series) or not X_test.index.equals(y_test.index):
        raise ValueError('Test labels must be a Series aligned to feature row_id values')
    y = y_test.to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Test labels must contain both binary classes 0 and 1')

    metric_rows = []
    predictions = pd.DataFrame({'row_id': X_test.index.to_numpy(), 'y_true': y})
    for name in MODEL_SET:
        pipeline = model_loader(experiment / name / 'pipeline.joblib')
        if not hasattr(pipeline, 'predict_proba') or list(getattr(pipeline, 'classes_', [])) != [0, 1]:
            raise ValueError(f'Saved pipeline for {name} must expose predict_proba and classes [0, 1]')
        with threadpool_limits(limits=1):
            probability = np.asarray(pipeline.predict_proba(X_test), dtype=float)
        if probability.shape != (len(X_test), 2):
            raise ValueError(f'Saved pipeline for {name} returned an unexpected probability shape')
        p_yes = probability[:, 1]
        threshold = thresholds[name]
        metrics = binary_metrics(y, p_yes, threshold)
        metric_rows.append({'model': name, **metrics})
        predictions[f'{name}_p_yes'] = p_yes
        predictions[f'{name}_prediction'] = (p_yes >= threshold).astype(int)

    # Save the approved bytes we actually validated, not a potentially edited file.
    (final_dir / 'frozen_protocol.json').write_bytes(protocol_bytes)
    pd.DataFrame(metric_rows).to_csv(final_dir / 'test_metrics.csv', index=False)
    predictions.to_csv(final_dir / 'test_predictions.csv', index=False)
    receipt = {
        'status': 'complete',
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'n': len(X_test),
        'row_id_source': 'X_test.index',
        'feature_policy': FEATURE_POLICY,
        'features': list(features),
        'protocol_sha256': _sha256(final_dir / 'frozen_protocol.json'),
        'pipeline_sha256': dict(protocol['pipeline_sha256']),
        'thresholds': thresholds,
        'preflight': preflight,
        'outputs': ['frozen_protocol.json', 'test_metrics.csv', 'test_predictions.csv'],
    }
    (final_dir / 'evaluation_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return final_dir
