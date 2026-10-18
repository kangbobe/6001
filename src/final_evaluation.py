"""One-shot evaluation of frozen development pipelines on the test partition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

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


class EvaluationRefused(ValueError):
    """Raised when a final-test precondition is not satisfied."""


def _read_json(path: Path) -> dict:
    try:
        with path.open('r', encoding='utf-8') as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationRefused(f'Cannot read valid JSON from {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise EvaluationRefused(f'Expected a JSON object in {path}')
    return value


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_protocol(protocol: dict) -> dict[str, float]:
    if protocol.get('frozen') is not True:
        raise EvaluationRefused('Protocol must set frozen to true')
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
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
                         load_partition_fn=None, features=None, model_loader=None):
    """Validate frozen artifacts, then evaluate each saved pipeline exactly once.

    Optional loader/features/model_loader arguments exist for isolated synthetic tests;
    production defaults are imported lazily so preflight never touches the test data.
    """
    if unlock_test is not True:
        raise EvaluationRefused('Test evaluation requires explicit --unlock-test authorization')
    if protocol_path is None:
        raise EvaluationRefused('A frozen protocol path is required')
    protocol_path = Path(protocol_path)
    protocol = _read_json(protocol_path)
    thresholds = _validate_protocol(protocol)

    if experiment_path is None:
        from src.data import ROOT
        experiment_path = ROOT / 'results/experiment'
    experiment = Path(experiment_path)
    if not experiment.is_dir():
        raise EvaluationRefused(f'Experiment directory does not exist: {experiment}')
    _validate_experiment(experiment, protocol)

    final_dir = experiment / 'final_test'
    if final_dir.exists():
        raise EvaluationRefused(f'Final evaluation output already exists; refusing to overwrite: {final_dir}')

    # Create the fixed destination only after every preflight gate. If inference or
    # writing fails, leave it in place so a later invocation cannot silently replace it.
    final_dir.mkdir()

    if load_partition_fn is None or features is None:
        from src.data import FEATURES, load_partition
        if load_partition_fn is None:
            load_partition_fn = load_partition
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

    shutil.copyfile(protocol_path, final_dir / 'frozen_protocol.json')
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
        'outputs': ['frozen_protocol.json', 'test_metrics.csv', 'test_predictions.csv'],
    }
    (final_dir / 'evaluation_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return final_dir
