#!/usr/bin/env python3
"""Independently verify saved final scores without source/project metrics or inference.

Only reads frozen protocol, evaluation receipt, saved test metric/prediction CSVs,
validation metric CSV, and split-indices JSON. Uses CSV float round-trip decoding.
AP is summed at tied-score group boundaries; ROC AUC uses the pairwise rank
probability with half credit for ties. No sklearn metrics or model/data loaders.
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
FINAL = REPO / 'results/experiment/final_test'
OUT = Path(__file__).with_suffix('.json')


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def independent_metrics(y, p, threshold):
    n = len(y)
    positive = int(y.sum())
    negative = n - positive
    predicted = p >= threshold
    tp = int(np.count_nonzero(predicted & (y == 1)))
    fp = int(np.count_nonzero(predicted & (y == 0)))
    tn = int(np.count_nonzero(~predicted & (y == 0)))
    fn = int(np.count_nonzero(~predicted & (y == 1)))
    # Each unique score is one tied group. Ascending group counts suffice for
    # exact pairwise AUC and, in reverse order, noninterpolated average precision.
    scores, groups = np.unique(p, return_inverse=True)
    group_positive = np.bincount(groups, weights=y).astype(np.int64)
    group_total = np.bincount(groups)
    group_negative = group_total - group_positive
    below_negative = np.cumsum(group_negative) - group_negative
    auc = float(np.sum(group_positive * (below_negative + 0.5 * group_negative)) / (positive * negative))
    descending_positive = group_positive[::-1]
    descending_total = group_total[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    cumulative_total = np.cumsum(descending_total)
    ap = float(np.sum((descending_positive / positive) * (cumulative_positive / cumulative_total)))
    clipped = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    metrics = {
        'n': n,
        'positive_rate': positive / n,
        'threshold': threshold,
        'average_precision': ap,
        'roc_auc': auc,
        'accuracy': (tp + tn) / n,
        'balanced_accuracy': (tp / positive + tn / negative) / 2,
        'precision': tp / (tp + fp) if (tp + fp) else 0.0,
        'recall': tp / positive,
        'f1': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        'brier_score': float(np.mean((p - y) ** 2)),
        'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }
    return metrics, predicted, len(scores)


def main():
    paths = {
        'protocol': FINAL / 'frozen_protocol.json',
        'receipt': FINAL / 'evaluation_receipt.json',
        'test_metrics': FINAL / 'test_metrics.csv',
        'test_predictions': FINAL / 'test_predictions.csv',
        'validation_metrics': REPO / 'results/member_d/evaluation/metrics.csv',
        'split_indices': REPO / 'data/splits/split_indices.json',
    }
    protocol = read_json(paths['protocol'])
    receipt = read_json(paths['receipt'])
    split = read_json(paths['split_indices'])
    rows = read_csv(paths['test_predictions'])
    reported = {row['model']: row for row in read_csv(paths['test_metrics'])}
    validation = {row['model']: row for row in read_csv(paths['validation_metrics']) if row['rule'] == 'selected'}
    model_names = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'mlp']
    assert protocol['models'] == model_names
    assert list(reported) == model_names
    assert set(validation) == set(model_names)
    rowids = [int(row['row_id']) for row in rows]
    assert rowids == split['test'], 'Test order or sample identities do not match split'
    assert len(set(rowids)) == len(rowids)
    assert not (set(rowids) & set(split['train']))
    assert not (set(rowids) & set(split['val']))
    y = np.asarray([int(row['y_true']) for row in rows], dtype=np.int64)
    assert set(y) == {0, 1}
    assert receipt['status'] == 'complete'
    assert receipt['n'] == len(y)
    assert protocol['frozen'] is True and protocol['approval_status'] == 'approved'
    assert receipt['preflight']['status'] == 'passed'
    assert receipt['preflight']['approval_status'] == 'approved'
    assert receipt['protocol_sha256'] == hashlib.sha256(paths['protocol'].read_bytes()).hexdigest()
    assert receipt['thresholds'] == protocol['thresholds']
    assert receipt['pipeline_sha256'] == protocol['pipeline_sha256']
    assert receipt['preflight']['pipeline_sha256'] == protocol['pipeline_sha256']
    assert receipt['preflight']['input_sha256'] == protocol['input_sha256']
    for field in ('validation_analysis_config_sha256', 'validation_receipt_sha256'):
        assert receipt['preflight'][field] == protocol[field]
    assert hashlib.sha256(paths['split_indices'].read_bytes()).hexdigest() == protocol['input_sha256']['data/splits/split_indices.json']
    result = {
        'status': 'passed',
        'scope': 'Independent arithmetic from saved outputs only; no test reload, inference, training, tuning, or model selection',
        'implementation': 'Python csv/numpy; no sklearn or project metrics imports',
        'metric_tolerance_absolute': 1e-13,
        'n': len(y), 'n_positive': int(y.sum()), 'n_negative': int(len(y) - y.sum()),
        'row_ids_exact_test_split_in_exact_order': True,
        'row_ids_unique': True, 'train_overlap': 0, 'validation_overlap': 0,
        'positive_class': 1,
        'approved_protocol_receipt_consistent': True,
        'inputs_sha256': {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        'models': {},
    }
    for name in model_names:
        p = np.asarray([float(row[name + '_p_yes']) for row in rows], dtype=np.float64)
        saved_class = np.asarray([int(row[name + '_prediction']) for row in rows], dtype=np.int64)
        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
        assert set(saved_class).issubset({0, 1})
        threshold = protocol['thresholds'][name]
        assert threshold == float(validation[name]['threshold']) == float(reported[name]['threshold'])
        metric, predicted, unique_count = independent_metrics(y, p, threshold)
        assert np.array_equal(predicted, saved_class)
        errors = {field: abs(value - float(reported[name][field])) for field, value in metric.items()}
        assert all(error <= 1e-13 for error in errors.values()), (name, errors)
        deltas = {field: value - float(validation[name][field]) for field, value in metric.items()}
        result['models'][name] = {
            'recomputed_metrics': metric,
            'max_absolute_difference_from_saved_metrics': max(errors.values()),
            'absolute_differences_by_metric': errors,
            'threshold_unchanged_from_selected_validation': True,
            'stored_predictions_equal_p_ge_threshold': True,
            'exact_threshold_equalities': int(np.count_nonzero(p == threshold)),
            'unique_probability_values': unique_count,
            'delta_test_minus_selected_validation': deltas,
        }
    # Tiny analytical self-check verifies tied-score AP/AUC and the inclusive
    # threshold even if real saved probabilities do not hit a selected boundary.
    synthetic, synthetic_pred, _ = independent_metrics(np.array([0, 1, 1, 0]), np.array([.2, .2, .8, .8]), .2)
    assert np.array_equal(synthetic_pred, np.ones(4, dtype=bool))
    assert synthetic['average_precision'] == .5 and synthetic['roc_auc'] == .5
    result['synthetic_tie_and_inclusive_boundary_selfcheck'] = 'passed'
    result['interpretation'] = [
        'Preselected random forest remains the report model; no test-driven reselection is performed.',
        'All four fitted families have modestly lower test AP, ROC AUC, and selected-threshold F1 than their validation estimates. Validation threshold optimization creates optimism and sampling variation can contribute; no significance conclusion is made.',
        'Random forest test AP and selected-threshold F1 are descriptively highest among the five fixed models, but this observation is not a new selection step.',
        'Random forest Brier score and log loss exceed the constant Dummy baseline, so good ranking does not imply reliable probabilities. This does not justify post-test calibration or threshold adjustment.',
        'The test prevalence is 11.6997%, so the Dummy accuracy of 88.3003% illustrates why accuracy alone is misleading.',
    ]
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'], 'n': result['n'], 'n_positive': result['n_positive'],
        'max_absolute_metric_difference': max(v['max_absolute_difference_from_saved_metrics'] for v in result['models'].values()),
        'output': str(OUT),
        'model_deltas': {name: {field: item['delta_test_minus_selected_validation'][field] for field in ['average_precision', 'roc_auc', 'f1']} for name, item in result['models'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
