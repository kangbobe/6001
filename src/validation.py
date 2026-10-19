"""Member D validation-only integration. Existing A/B/C artifacts stay read-only."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits

from src.data import ROOT, FEATURES, load_split_indices
from src.final_evaluation import MODEL_SET, _validate_experiment
from src.metrics import binary_metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_input_identities(root=ROOT):
    """Opaque file hashing, not test-row parsing or target analysis."""
    root = Path(root)
    manifest = read_json(root / 'results/experiment/post_training_review/manifest.json')
    for record in manifest['inputs']:
        path = root / record['path']
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Frozen input identity mismatch: {record["path"]}')
    return {r['path']: r['sha256'] for r in manifest['inputs']}


def load_validation(root=ROOT):
    """Materialize only validation records, in frozen order.

    The locked legacy CSV has one record per physical line. All file bytes are
    scanned for identity; non-validation lines are never decoded or CSV-parsed.
    This is stricter than the existing A loader, which parses the whole table.
    """
    root = Path(root)
    verify_input_identities(root)
    ids = load_split_indices(root)['val']
    wanted = set(ids.tolist())
    selected = {}
    with (root / 'data/raw/bank-full.csv').open('rb') as stream:
        header = next(stream)
        for row_id, line in enumerate(stream):
            if row_id in wanted:
                selected[row_id] = line
    if set(selected) != wanted:
        raise ValueError('Missing validation records')
    content = header + b''.join(selected[int(i)] for i in ids)
    # No full-table dataframe is created, even temporarily.
    frame = pd.read_csv(io.BytesIO(content), sep=';', keep_default_na=False)
    lock = read_json(root / 'config/dataset_lock.json')
    if list(frame.columns) != lock['columns'] or len(frame) != len(ids):
        raise ValueError('Validation schema mismatch')
    if frame.isna().any().any() or not frame.y.isin(['no', 'yes']).all():
        raise ValueError('Invalid validation values or labels')
    frame.index = pd.Index(ids, name='row_id')
    return frame[FEATURES].copy(), frame.y.map({'no': 0, 'yes': 1}).astype('int64')


def validate_predictions(frame, name, ids, y=None):
    columns = {'row_id', 'model', 'split', 'y_true', 'p_yes', 'prediction_at_0_5'}
    if set(frame.columns) != columns or len(frame.columns) != len(columns):
        raise ValueError('Unexpected prediction schema')
    if not frame.row_id.is_unique or not np.array_equal(frame.row_id.to_numpy(), ids):
        raise ValueError('Prediction row_id alignment mismatch')
    if not frame.model.eq(name).all() or not frame['split'].eq('val').all():
        raise ValueError('Prediction model or split mismatch')
    labels = frame.y_true.to_numpy()
    p = frame.p_yes.to_numpy(dtype=float)
    binary_metrics(labels, p)
    if isinstance(y, pd.Series) and not np.array_equal(y.index.to_numpy(), ids):
        raise ValueError('Validation label row_id alignment mismatch')
    if y is not None and not np.array_equal(labels, np.asarray(y)):
        raise ValueError('Prediction labels do not match validation labels')
    if not np.array_equal(frame.prediction_at_0_5.to_numpy(), (p >= .5).astype(int)):
        raise ValueError('Stored 0.5 predictions mismatch')
    return p


def threshold_sweep(y, p):
    """All realizable score-cut decision sets; exact score ties stay together."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    binary_metrics(y, p)  # shared strict validation, not a separate metric definition
    unique = np.unique(p)
    middle = unique[:-1] + (unique[1:] - unique[:-1]) / 2
    # Adjacent floating values have no representable interior point.
    middle = np.where(middle <= unique[:-1], unique[1:], middle)
    thresholds = np.unique(np.r_[0., middle, 1.])
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order].astype(np.int64)
    prefix = np.r_[0, np.cumsum(sorted_y)]
    cut = np.searchsorted(sorted_p, thresholds, side='left')
    positive_n = int(y.sum())
    fn = prefix[cut]
    tp = positive_n - fn
    tn = cut - fn
    fp = len(y) - cut - tp
    selected = tp + fp
    precision = np.divide(tp, selected, out=np.zeros(len(tp), dtype=float), where=selected > 0)
    return pd.DataFrame({'threshold': thresholds, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                         'precision': precision, 'recall': tp / positive_n,
                         'f1': 2 * tp / (positive_n + selected), 'selected_n': selected})


def select_threshold(y, p):
    sweep = threshold_sweep(y, p)
    # Integer cross-multiplication avoids treating merely close F1 values as ties.
    best_num, best_den, best = -1, 1, None
    for row in sweep.itertuples():
        numerator = 2 * int(row.tp)
        denominator = 2 * int(row.tp) + int(row.fp) + int(row.fn)
        if numerator * best_den >= best_num * denominator:
            best_num, best_den, best = numerator, denominator, float(row.threshold)
    return best


@contextmanager
def forbid_fitting(models):
    """Fail loudly if any estimator's fit/partial_fit/fit_transform is called."""
    def refused(*args, **kwargs):
        raise RuntimeError('Member D validation must never fit or retrain estimators')
    estimator_types = set()
    for model in models.values():
        estimator_types.add(type(model))
        estimator_types.update(type(v) for v in model.get_params(deep=True).values()
                               if isinstance(v, BaseEstimator))
    with ExitStack() as stack:
        for cls in sorted(estimator_types, key=lambda c: (c.__module__, c.__name__)):
            for method in ['fit', 'partial_fit', 'fit_transform']:
                if hasattr(cls, method):
                    stack.enter_context(patch.object(cls, method, refused))
        yield


def protected_identities(root):
    root = Path(root)
    paths = [p for folder in ['data', 'results/experiment'] for p in (root / folder).rglob('*') if p.is_file()]
    original = read_json(root / 'results/experiment/post_training_review/manifest.json')
    paths += [root / record['path'] for record in original.get('source_files', [])]
    paths += [root / p for p in ['config/dataset_lock.json', 'config/feature_policy.csv',
                                'config/final_protocol.draft.json', 'requirements.lock.txt']]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def load_handoff(root=ROOT):
    root = Path(root)
    experiment = root / 'results/experiment'
    # The later posthoc receipt identifies the delivered source. The earlier
    # post_training_review snapshot predates the future-training provenance fix.
    posthoc = read_json(experiment / 'posthoc_verification.json')
    for base, records in [(root, posthoc['project_files']), (experiment, posthoc['experiment_files'])]:
        for record in records:
            path = base / record['path']
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError(f'Delivered handoff identity mismatch: {record["path"]}')
    draft = read_json(root / 'config/final_protocol.draft.json')
    _validate_experiment(experiment, draft)
    verify_input_identities(root)
    X, y = load_validation(root)
    models, predictions, checks = {}, {}, []
    summary = pd.read_csv(experiment / 'development_summary.csv').set_index('model')
    for name in MODEL_SET:
        folder = experiment / name
        saved = pd.read_csv(folder / 'validation_predictions.csv', float_precision='round_trip')
        p = validate_predictions(saved, name, X.index.to_numpy(), y)
        meta = read_json(folder / 'model_metadata.json')
        if meta['input_features'] != FEATURES or meta['fit_n'] != 27126:
            raise ValueError(f'Unexpected training metadata for {name}')
        candidates = pd.read_csv(folder / 'cv_candidates.csv')
        selected = candidates.sort_values(['mean_average_precision', 'config_id'], ascending=[False, True]).iloc[0]
        if int(selected.config_id) != meta['config_id']:
            raise ValueError(f'CV selection mismatch for {name}')
        model = joblib.load(folder / 'pipeline.joblib')
        if list(model.classes_) != [0, 1]:
            raise ValueError(f'Wrong positive class for {name}')
        models[name], predictions[name] = model, p
        with forbid_fitting({name: model}), threadpool_limits(limits=1):
            proba = np.asarray(model.predict_proba(X), dtype=float)
        if (proba.shape != (len(X), 2) or not np.isfinite(proba).all()
                or ((proba < 0) | (proba > 1)).any()):
            raise ValueError(f'Invalid pipeline probability shape or values for {name}')
        np.testing.assert_allclose(proba.sum(axis=1), 1., atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(p, proba[:, 1], atol=1e-12, rtol=1e-12)
        for key, value in binary_metrics(y, p).items():
            np.testing.assert_allclose(value, meta['validation_metrics_at_0_5'][key], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(value, summary.loc[name, key], atol=1e-12, rtol=1e-12)
        checks.append({'model': name, 'n': len(X), 'selected_config': int(selected.config_id),
                       'max_abs_score_difference': float(np.abs(p - proba[:, 1]).max()),
                       'metrics_match_metadata_and_summary': True})
    return X, y, predictions, models, checks


def run_validation(*, stage, output, root=ROOT, protocol_output=None):
    """Run one non-overwriting stage; no test loader or final-evaluation call."""
    if stage not in {'evaluation', 'errors', 'importance'}:
        raise ValueError('Unknown validation stage')
    root, output = Path(root).resolve(), Path(output).resolve()
    stage_dir = output / stage
    protocol_output = (Path(protocol_output) if protocol_output else root / 'config/final_protocol.proposed.json').resolve()
    for destination in [stage_dir, protocol_output]:
        for frozen in [root / 'data', root / 'results/experiment']:
            if destination == frozen or frozen in destination.parents:
                raise ValueError('Output destination must be outside frozen data and experiment directories')
    if stage_dir.exists():
        raise FileExistsError(f'Preserve previous stage output: {stage_dir}')
    if stage == 'evaluation' and protocol_output.exists():
        raise FileExistsError(f'Preserve previous proposed protocol: {protocol_output}')
    config = read_json(root / 'config/member_d_analysis.json')
    if config['partition'] != 'val' or config['test_evaluated'] is not False or config['retraining_allowed'] is not False:
        raise ValueError('Analysis configuration must be validation-only without retraining')
    if config['permutation'] != {'scoring': 'average_precision', 'n_repeats': 10, 'random_state': 42,
                                  'n_jobs': 1, 'max_samples': 1.0, 'std_ddof': 0}:
        raise ValueError('Unexpected permutation protocol; do not silently change prespecified settings')
    before = protected_identities(root)
    X, y, predictions, models, checks = load_handoff(root)
    object_before = {name: joblib.hash(model) for name, model in models.items()}
    if stage == 'evaluation':
        thresholds = {name: .5 if name == 'dummy' else select_threshold(y, p) for name, p in predictions.items()}
    else:
        prior = read_json(output / 'evaluation/receipt.json')
        if prior['protected_files_before'] != before or prior['analysis_config_sha256'] != sha256(root / 'config/member_d_analysis.json'):
            raise ValueError('Inputs/configuration changed since threshold selection')
        thresholds = prior['thresholds']
    # Verify that serialized pipeline scores and stored score precision produce
    # exactly the same selected decisions, not merely approximately equal scores.
    with forbid_fitting(models), threadpool_limits(limits=1):
        for name, model in models.items():
            fresh = model.predict_proba(X)[:, 1]
            np.testing.assert_array_equal(fresh >= thresholds[name], predictions[name] >= thresholds[name])
        stage_dir.mkdir(parents=True, exist_ok=False)
        from src.validation_analysis import (write_curves_and_confusions, write_error_analysis,
                                              write_permutation_importance)
        if stage == 'evaluation':
            metric_rows, sweeps = [], []
            for name, p in predictions.items():
                for rule, threshold in [('default_0_5', .5), ('selected', thresholds[name])]:
                    metric_rows.append({'partition': 'val', 'model': name, 'rule': rule,
                                        **binary_metrics(y, p, threshold)})
                sweeps.append(threshold_sweep(y, p).assign(model=name, partition='val'))
            pd.DataFrame(metric_rows).to_csv(stage_dir / 'metrics.csv', index=False)
            pd.concat(sweeps, ignore_index=True).to_csv(stage_dir / 'threshold_sweep.csv', index=False)
            write_json(stage_dir / 'thresholds.json', thresholds)
            write_curves_and_confusions(predictions, y.to_numpy(), thresholds, stage_dir)
        elif stage == 'errors':
            write_error_analysis(X, y, predictions, thresholds, stage_dir)
        else:
            write_permutation_importance(models, X, y, stage_dir, n_repeats=10, random_state=42)
    after = protected_identities(root)
    if before != after:
        raise RuntimeError('Protected A/B/C files changed during validation')
    if object_before != {name: joblib.hash(model) for name, model in models.items()}:
        raise RuntimeError('Fitted pipeline state changed during validation')
    receipt = {
        'status': 'complete', 'stage': stage, 'partition': 'val', 'n': len(y),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'positive_class': 1,
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib','matplotlib','pytest','threadpoolctl']},
        'thresholds': thresholds, 'validation_checks': checks,
        'analysis_config_sha256': sha256(root / 'config/member_d_analysis.json'),
        'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in [root / 'src/validation.py', root / 'src/validation_analysis.py', root / 'scripts/evaluate_validation.py']},
        'protected_files_before': before, 'protected_files_unchanged': before == after,
        'in_memory_models_unchanged': True, 'fit_methods_blocked': True,
        'test_rows_parsed': False, 'test_predictions_or_metrics_computed': False,
        'data_access': 'Raw bytes hashed/scanned; only frozen validation lines decoded and parsed. Split membership metadata checked.',
        'outputs_sha256': {p.relative_to(stage_dir).as_posix(): sha256(p) for p in sorted(stage_dir.rglob('*')) if p.is_file()},
    }
    write_json(stage_dir / 'receipt.json', receipt)
    if stage == 'evaluation':
        proposed = read_json(root / 'config/final_protocol.draft.json')
        proposed.update(frozen=False, owner='', frozen_at='', thresholds=thresholds,
            approval_status='pending_explicit_user_approval',
            decision_rule='Unweighted validation F1 maximum over 0, adjacent distinct score midpoints, and 1; exact rational F1 ties choose highest candidate; predict yes iff p_yes >= threshold; Dummy fixed at 0.5.',
            primary_metric='average_precision',
            report_metrics=list(binary_metrics(y, predictions['dummy'])),
            cv_recommended_family=read_json(root / 'results/experiment/completion.json')['cv_recommended_family'],
            validation_analysis_config_sha256=sha256(root / 'config/member_d_analysis.json'),
            validation_receipt_sha256=sha256(stage_dir / 'receipt.json'),
            input_sha256=verify_input_identities(root),
            notes='Approval-pending proposal, not an authorized frozen protocol. No test evaluation; no retraining. Keep all five models and full-precision thresholds. F1 is an academic convention, not a profit optimum; threshold-selected validation performance is optimistic. After explicit approval record actual owner/time and freeze a separate final_protocol.json. Verify recorded input/config/source identities before the one-shot final entry point; existing final gate enforces model hashes but not these additional hashes.')
        protocol_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(protocol_output, proposed)
    return stage_dir
