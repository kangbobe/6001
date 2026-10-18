"""Explicit fold-local fitting and complete development experiment records."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src.data import ROOT, FEATURES, load_partition
from src.metrics import binary_metrics
from src.models import MODEL_NAMES, build_model, search_configs


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit_candidate(name, config, X, y):
    """Only this fitting subset determines balanced weights and preprocessing."""
    kwargs = {}
    if config['weight'] == 'balanced':
        kwargs['model__sample_weight'] = compute_sample_weight('balanced', y)
    attempts = []
    for cap in ([300, 600] if name == 'mlp' else [None]):
        pipe = build_model(name, config, iteration_cap=cap)
        start = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipe.fit(X, y, **kwargs)
        estimator = pipe.named_steps['model']
        n_iter = getattr(estimator, 'n_iter_', None)
        n_iter = int(np.max(n_iter)) if n_iter is not None else None
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        attempts.append({
            'max_iter': cap if cap is not None else getattr(estimator, 'max_iter', None),
            'n_iter': n_iter, 'fit_seconds': time.perf_counter() - start,
            'convergence_warning': convergence_warning,
            'warnings': [f'{w.category.__name__}: {w.message}' for w in caught],
            'loss_curve': [float(x) for x in getattr(estimator, 'loss_curve_', [])],
        })
        # Apply the predeclared retry equally to every MLP fit, not based on its score.
        if not (name == 'mlp' and cap == 300 and n_iter >= cap and convergence_warning):
            break
    return pipe, attempts


def score_fold(name, config, config_id, fold_id, fit_ids, score_ids, X, y):
    fitted, attempts = fit_candidate(name, config, X.iloc[fit_ids], y.iloc[fit_ids])
    with threadpool_limits(limits=1):
        probability = fitted.predict_proba(X.iloc[score_ids])[:, 1]
        train_probability = fitted.predict_proba(X.iloc[fit_ids])[:, 1]
    result = {
        'model': name, 'config_id': config_id, 'fold_id': fold_id,
        'params': json.dumps(config, sort_keys=True), 'fit_n': len(fit_ids),
        'fit_seconds': sum(a['fit_seconds'] for a in attempts),
        'attempt_count': len(attempts), 'n_iter': attempts[-1]['n_iter'],
        'final_convergence_warning': attempts[-1]['convergence_warning'],
        'warnings': json.dumps([a['warnings'] for a in attempts]),
        'train_ap': float(average_precision_score(y.iloc[fit_ids], train_probability)),
        **binary_metrics(y.iloc[score_ids], probability),
    }
    return result, {'config_id': config_id, 'fold_id': fold_id, 'attempts': attempts}


def train_family(name, X, y, X_val, y_val, folds, output, jobs=2):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=False)
    configs = search_configs(name)
    save_json(directory / 'search_configs.json', configs)
    rows, histories, candidates = [], [], []
    for config_id, config in enumerate(configs):
        results = joblib.Parallel(n_jobs=jobs)(joblib.delayed(score_fold)(
            name, config, config_id, fold_id, fit, score, X, y)
            for fold_id, (fit, score) in enumerate(folds))
        config_rows = [item[0] for item in results]
        rows.extend(config_rows)
        histories.extend(item[1] for item in results)
        candidate = {'model': name, 'config_id': config_id, 'params': json.dumps(config, sort_keys=True)}
        for metric in ['average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy',
                       'balanced_accuracy', 'train_ap', 'fit_seconds']:
            values = [r[metric] for r in config_rows]
            candidate[f'mean_{metric}'] = float(np.mean(values))
            candidate[f'std_{metric}'] = float(np.std(values, ddof=1))
        candidate['unconverged_folds'] = sum(r['final_convergence_warning'] for r in config_rows)
        candidates.append(candidate)
        pd.DataFrame(rows).to_csv(directory / 'cv_folds.csv', index=False)
        pd.DataFrame(candidates).to_csv(directory / 'cv_candidates.csv', index=False)
        save_json(directory / 'fit_histories.json', histories)
        print(f'{name} config {config_id+1}/{len(configs)} CV AP={candidate["mean_average_precision"]:.5f}', flush=True)
    selected = sorted(candidates, key=lambda c: (-c['mean_average_precision'], c['config_id']))[0]
    config = configs[selected['config_id']]
    fitted, attempts = fit_candidate(name, config, X, y)
    with threadpool_limits(limits=1):
        probability = fitted.predict_proba(X_val)[:, 1]
    assert list(fitted.classes_) == [0, 1]
    joblib.dump(fitted, directory / 'pipeline.joblib', compress=3)
    # Serialization is a real handoff requirement: reload all validation probabilities.
    restored = joblib.load(directory / 'pipeline.joblib')
    with threadpool_limits(limits=1):
        restored_probability = restored.predict_proba(X_val)[:, 1]
    np.testing.assert_allclose(probability, restored_probability, rtol=1e-12, atol=1e-12)
    pd.DataFrame({'row_id': X_val.index, 'model': name, 'split': 'val',
                  'y_true': y_val.to_numpy(), 'p_yes': probability,
                  'prediction_at_0_5': (probability >= .5).astype(int)}).to_csv(
                      directory / 'validation_predictions.csv', index=False)
    features = fitted.named_steps['preprocess'].get_feature_names_out().tolist()
    save_json(directory / 'feature_names.json', features)
    metadata = {
        'model': name, 'config_id': selected['config_id'], 'selected_params': config,
        'selection_rule': 'highest mean training CV AP; exact ties lowest config_id',
        'cv_average_precision': selected['mean_average_precision'],
        'cv_ap_std': selected['std_average_precision'],
        'refit_partition': 'train', 'fit_n': len(X), 'input_features': FEATURES,
        'transformed_feature_count': len(features), 'classes': [0, 1],
        'refit_attempts': attempts, 'serialization_check_max_abs_error': float(np.max(np.abs(probability-restored_probability))),
        'validation_metrics_at_0_5': binary_metrics(y_val, probability),
        'test_evaluated': False,
    }
    save_json(directory / 'model_metadata.json', metadata)
    print(f'{name} complete; selected config {selected["config_id"]}, validation AP={metadata["validation_metrics_at_0_5"]["average_precision"]:.5f}', flush=True)
    return metadata


def run_experiment(output=ROOT / 'results/experiment', jobs=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    X, y = load_partition('train')
    X_val, y_val = load_partition('val')
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
    fold_assignment = np.full(len(X), -1)
    for k, (_, held) in enumerate(folds):
        fold_assignment[held] = k
    pd.DataFrame({'row_id': X.index, 'cv_holdout_fold': fold_assignment}).to_csv(output / 'cv_manifest.csv', index=False)
    protocol = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib','matplotlib']},
        'random_state': 42, 'cv': 'StratifiedKFold(n_splits=3, shuffle=True, random_state=42)',
        'primary_metric': 'average_precision', 'development_threshold': .5,
        'weight_rule': 'compute_sample_weight balanced using only each fit fold; evaluation unweighted',
        'mlp_retry_rule': '300->600 from scratch for every max_iter convergence warning; same seed',
        'pdays_rule': 'has_previous_contact only; exact elapsed days excluded',
        'search_spaces': {name: search_configs(name) for name in MODEL_NAMES},
        'train_n': len(X), 'validation_n': len(X_val), 'test_evaluated': False,
        'interpretation': 'Retrospective pre-campaign screening simulation; record-level random-split generalization',
    }
    save_json(output / 'run_protocol.json', protocol)
    summaries = []
    for name in ['dummy'] + MODEL_NAMES:
        metadata = train_family(name, X, y, X_val, y_val, folds, output, jobs)
        summaries.append({'model': name, 'selected_config': metadata['config_id'],
                          'cv_ap_mean': metadata['cv_average_precision'], 'cv_ap_std': metadata['cv_ap_std'],
                          'refit_n_iter': metadata['refit_attempts'][-1]['n_iter'],
                          'refit_convergence_warning': metadata['refit_attempts'][-1]['convergence_warning'],
                          **metadata['validation_metrics_at_0_5']})
        pd.DataFrame(summaries).to_csv(output / 'development_summary.csv', index=False)
    recommendation = sorted([s for s in summaries if s['model'] != 'dummy'], key=lambda s: -s['cv_ap_mean'])[0]['model']
    save_json(output / 'completion.json', {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter()-started,
        'cv_recommended_family': recommendation, 'test_evaluated': False,
        'models': ['dummy'] + MODEL_NAMES, 'status': 'complete',
    })
    return pd.DataFrame(summaries)
