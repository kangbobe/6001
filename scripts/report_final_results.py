#!/usr/bin/env python3
"""Derive figures and comparisons from the saved one-shot final evaluation.

This program reads saved prediction/metric files only. It never imports a project
model, loads a fitted pipeline, accesses raw data, or searches for a threshold.
Repeated reporting cannot change the frozen predictions or evaluation receipt.
A fresh output directory is required on every invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

MODELS = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'mlp']
LABELS = {
    'dummy': 'Dummy', 'logistic_regression': 'Logistic regression',
    'decision_tree': 'Decision tree', 'random_forest': 'Random forest', 'mlp': 'MLP',
}
COLORS = dict(zip(MODELS, ['#777777', '#0072B2', '#E69F00', '#009E73', '#CC79A7']))
METRICS = [
    'n', 'positive_rate', 'threshold', 'average_precision', 'roc_auc', 'accuracy',
    'balanced_accuracy', 'precision', 'recall', 'f1', 'brier_score', 'log_loss',
    'tn', 'fp', 'fn', 'tp',
]
INTEGER_METRICS = {'n', 'tn', 'fp', 'fn', 'tp'}
SOURCES = {
    'protocol': 'results/experiment/final_test/frozen_protocol.json',
    'metrics': 'results/experiment/final_test/test_metrics.csv',
    'predictions': 'results/experiment/final_test/test_predictions.csv',
    'receipt': 'results/experiment/final_test/evaluation_receipt.json',
    'validation_metrics': 'results/member_d/evaluation/metrics.csv',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key!r}')
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=unique_keys)
    require(isinstance(value, dict), f'Expected JSON object: {path}')
    return value


def independent_metrics(y, p, threshold):
    """Independent sklearn calculation, without importing project metric code."""
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        'n': len(y), 'positive_rate': float(y.mean()), 'threshold': threshold,
        'average_precision': average_precision_score(y, p),
        'roc_auc': roc_auc_score(y, p), 'accuracy': accuracy_score(y, prediction),
        'balanced_accuracy': balanced_accuracy_score(y, prediction),
        'precision': precision_score(y, prediction, zero_division=0),
        'recall': recall_score(y, prediction, zero_division=0),
        'f1': f1_score(y, prediction, zero_division=0),
        'brier_score': brier_score_loss(y, p), 'log_loss': log_loss(y, p, labels=[0, 1]),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def validate_sources(root):
    paths = {key: root / relative for key, relative in SOURCES.items()}
    identities = {SOURCES[key]: sha256(path) for key, path in paths.items()}
    protocol = read_json(paths['protocol'])
    receipt = read_json(paths['receipt'])
    require(protocol.get('frozen') is True, 'Saved protocol is not frozen')
    require(protocol.get('approval_status') == 'approved', 'Saved protocol is not approved')
    require(protocol.get('models') == MODELS, 'Unexpected frozen model set/order')
    require(receipt.get('status') == 'complete', 'One-shot evaluation is incomplete')
    require(receipt.get('protocol_sha256') == identities[SOURCES['protocol']],
            'Saved protocol hash disagrees with evaluation receipt')
    require(receipt.get('thresholds') == protocol.get('thresholds'),
            'Receipt thresholds disagree with frozen protocol')
    require(receipt.get('pipeline_sha256') == protocol.get('pipeline_sha256'),
            'Receipt model identities disagree with frozen protocol')
    require(receipt.get('preflight', {}).get('status') == 'passed',
            'Evaluation receipt does not certify successful preflight')
    require(receipt.get('preflight', {}).get('approval_status') == 'approved',
            'Evaluation receipt does not certify explicit approval')
    thresholds = protocol.get('thresholds')
    require(isinstance(thresholds, dict) and set(thresholds) == set(MODELS),
            'Unexpected threshold model set')
    for name, threshold in thresholds.items():
        require(not isinstance(threshold, bool) and isinstance(threshold, (float, int))
                and np.isfinite(threshold) and 0 <= threshold <= 1,
                f'Invalid frozen threshold: {name}')
    require(thresholds['dummy'] == 0.5, 'Dummy threshold is not 0.5')

    # Round-trip conversion preserves threshold boundary decisions from saved doubles.
    test = pd.read_csv(paths['metrics'], float_precision='round_trip')
    predictions = pd.read_csv(paths['predictions'], float_precision='round_trip')
    validation_all = pd.read_csv(paths['validation_metrics'], float_precision='round_trip')
    require(list(test.columns) == ['model', *METRICS], 'Unexpected final metric schema')
    require(test['model'].tolist() == MODELS, 'Final metrics must contain all five models once')
    expected_columns = ['row_id', 'y_true']
    for model in MODELS:
        expected_columns.extend([f'{model}_p_yes', f'{model}_prediction'])
    require(list(predictions.columns) == expected_columns, 'Unexpected prediction schema')
    require(not predictions.isna().any().any(), 'Missing values in saved predictions')
    require(predictions['row_id'].is_unique, 'Duplicate prediction row IDs')
    require(np.isfinite(predictions['row_id'].to_numpy(dtype=float)).all(), 'Invalid row IDs')
    require(np.equal(predictions['row_id'], np.floor(predictions['row_id'])).all(),
            'Prediction row IDs must be integers')
    require(set(predictions['y_true'].unique()) == {0, 1},
            'Saved labels must contain both binary classes; yes is 1')
    y = predictions['y_true'].to_numpy(dtype=int)
    require(receipt.get('n') == len(y), 'Receipt sample count disagrees with predictions')
    require({'partition', 'model', 'rule', *METRICS} <= set(validation_all.columns),
            'Validation metric schema is incomplete')
    validation = validation_all.loc[
        (validation_all['partition'] == 'val') & (validation_all['rule'] == 'selected')
    ].copy()
    require(validation['model'].tolist() == MODELS,
            'Validation selected rows must contain all five models once in order')
    test = test.set_index('model')
    validation = validation.set_index('model')
    checks = []
    for model in MODELS:
        p = predictions[f'{model}_p_yes'].to_numpy(dtype=float)
        threshold = float(thresholds[model])
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                f'Invalid saved probabilities for {model}')
        require(np.array_equal(predictions[f'{model}_prediction'].to_numpy(),
                               (p >= threshold).astype(int)),
                f'Saved decisions disagree with frozen threshold for {model}')
        require(float(validation.loc[model, 'threshold']) == threshold,
                f'Validation and frozen thresholds differ for {model}')
        measured = independent_metrics(y, p, threshold)
        maximum_delta = 0.0
        for metric, actual in measured.items():
            saved = float(test.loc[model, metric])
            require(np.isfinite(saved), f'Non-finite final metric {model}/{metric}')
            matches = (saved == actual if metric in INTEGER_METRICS | {'threshold'}
                       else np.isclose(saved, actual, rtol=1e-12, atol=1e-12))
            require(matches, f'Final metric mismatch for {model}/{metric}: {saved} != {actual}')
            maximum_delta = max(maximum_delta, abs(saved - actual))
        checks.append({'model': model, 'n': len(y), 'threshold': threshold,
                       'saved_decisions_match_frozen_threshold': True,
                       'all_16_metrics_independently_reconciled': True,
                       'maximum_absolute_metric_difference': maximum_delta})
    return protocol, receipt, test, validation, predictions, identities, checks


def save_figure(fig, destination):
    fig.savefig(destination.with_suffix('.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(destination.with_suffix('.svg'), bbox_inches='tight', facecolor='white',
                metadata={'Date': None})
    plt.close(fig)


def write_tables_and_curves(protocol, test, validation, predictions, output):
    rows, confusion_rows, roc_rows, pr_rows = [], [], [], []
    y = predictions['y_true'].to_numpy(dtype=int)
    curves = {}
    for model in MODELS:
        row = {'model': model}
        for metric in METRICS:
            val_value = validation.loc[model, metric]
            test_value = test.loc[model, metric]
            row.update({f'validation_{metric}': val_value, f'test_{metric}': test_value,
                        f'delta_{metric}': test_value - val_value})
        rows.append(row)
        confusion_rows.append({'model': model, 'partition': 'test',
                               'threshold': protocol['thresholds'][model],
                               **{metric: int(test.loc[model, metric])
                                  for metric in ['tn', 'fp', 'fn', 'tp']}})
        p = predictions[f'{model}_p_yes'].to_numpy(dtype=float)
        fpr, tpr, roc_threshold = roc_curve(y, p, pos_label=1, drop_intermediate=False)
        precision, recall, pr_threshold = precision_recall_curve(y, p, pos_label=1)
        curves[model] = (fpr, tpr, precision, recall)
        for index, (x, z, threshold) in enumerate(zip(fpr, tpr, roc_threshold)):
            roc_rows.append({'model': model, 'point_index': index,
                             'false_positive_rate': x, 'true_positive_rate': z,
                             'score_threshold': threshold})
        for index, (x, z) in enumerate(zip(recall, precision)):
            pr_rows.append({'model': model, 'point_index': index, 'recall': x,
                            'precision': z,
                            'score_threshold': pr_threshold[index] if index < len(pr_threshold)
                            else None})
    pd.DataFrame(rows).to_csv(output / 'validation_test_comparison.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(output / 'confusion_matrices.csv', index=False)
    pd.DataFrame(roc_rows).to_csv(output / 'roc_points.csv', index=False)
    pd.DataFrame(pr_rows).to_csv(output / 'pr_points.csv', index=False)
    return curves


def write_figures(protocol, test, validation, curves, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'figure.titlesize': 16, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'ca6001-final-report-v1'})
    figures = output / 'figures'
    figures.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    for model in MODELS:
        fpr, tpr, precision, recall = curves[model]
        axes[0].plot(fpr, tpr, color=COLORS[model], lw=1.8,
                     label=f"{LABELS[model]}  AUC={test.loc[model, 'roc_auc']:.3f}")
        axes[1].step(recall, precision, where='post', color=COLORS[model], lw=1.8,
                     label=f"{LABELS[model]}  AP={test.loc[model, 'average_precision']:.3f}")
        tn, fp, fn, tp = (int(test.loc[model, key]) for key in ['tn', 'fp', 'fn', 'tp'])
        axes[0].scatter([fp / (tn + fp)], [tp / (tp + fn)], s=35,
                        facecolors='white', edgecolors=COLORS[model], zorder=5)
        # A zero-prediction classifier has undefined precision; do not plot the
        # project's numeric zero_division=0 convention as an empirical PR point.
        if tp + fp > 0:
            axes[1].scatter([test.loc[model, 'recall']], [test.loc[model, 'precision']], s=35,
                            facecolors='white', edgecolors=COLORS[model], zorder=5)
    axes[0].plot([0, 1], [0, 1], '--', color='#333333', alpha=0.65, lw=1,
                 label='Chance ranking')
    prevalence = float(test.iloc[0]['positive_rate'])
    axes[1].axhline(prevalence, ls='--', color='#333333', alpha=0.65, lw=1,
                   label=f'Test prevalence = {prevalence:.3f}')
    for ax, title, xlabel, ylabel in zip(axes, ['ROC', 'Precision–recall'],
                                        ['False positive rate', 'Recall'],
                                        ['True positive rate', 'Precision']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(-0.015, 1.015), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.16)
        ax.legend(loc='lower right' if title == 'ROC' else 'upper right', fontsize=8)
    fig.suptitle('One-time final test: ranking performance', y=1.01)
    fig.text(0.5, -0.025, 'Open circles: frozen validation-selected thresholds (Dummy: 0.5). No test threshold selection.\n'
             'PR operating point omitted when no positives are predicted. AP is average precision.',
             ha='center', fontsize=9)
    fig.tight_layout()
    save_figure(fig, figures / 'test_roc_pr')

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
    for model, ax in zip(MODELS, axes.flat):
        matrix = np.array([[int(test.loc[model, 'tn']), int(test.loc[model, 'fp'])],
                           [int(test.loc[model, 'fn']), int(test.loc[model, 'tp'])]])
        # Shared normalization makes shading comparable across models.
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=int(test.iloc[0]['n']))
        for i, j in np.ndindex((2, 2)):
            ax.text(j, i, f"{[['TN', 'FP'], ['FN', 'TP']][i][j]}\n{matrix[i, j]:,}",
                    ha='center', va='center', fontsize=13,
                    color='white' if matrix[i, j] > 0.50 * test.iloc[0]['n'] else '#111111')
        ax.set(title=f"{LABELS[model]}  |  threshold={protocol['thresholds'][model]:.6f}",
               xlabel='Predicted class', ylabel='Observed class',
               xticks=[0, 1], yticks=[0, 1], xticklabels=['No (0)', 'Yes (1)'],
               yticklabels=['No (0)', 'Yes (1)'])
        ax.spines[['top', 'right']].set_visible(False)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0.02, 0.8, 'Fixed validation-selected thresholds\n'
                      'Positive class: yes = 1\n'
                      f"Test sample: {int(test.iloc[0]['n']):,}\n"
                      'Counts shown without normalization\n'
                      'Shading uses the same scale in every panel.',
                      transform=axes.flat[-1].transAxes, fontsize=11, va='top', linespacing=1.6)
    fig.suptitle('One-time final test: confusion matrices', y=1.01)
    fig.tight_layout()
    save_figure(fig, figures / 'test_confusion_matrices')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [('average_precision', 'Average precision ↑'), ('roc_auc', 'ROC AUC ↑'),
              ('f1', 'F1 at frozen threshold ↑'), ('precision', 'Precision at frozen threshold ↑'),
              ('recall', 'Recall at frozen threshold ↑'), ('brier_score', 'Brier score ↓')]
    positions = np.arange(len(MODELS))
    short_labels = ['Dummy', 'Logistic\nregression', 'Decision\ntree', 'Random\nforest', 'MLP']
    for ax, (metric, title) in zip(axes.flat, panels):
        val_values = validation.loc[MODELS, metric].to_numpy(dtype=float)
        test_values = test.loc[MODELS, metric].to_numpy(dtype=float)
        for index, model in enumerate(MODELS):
            ax.bar(index - 0.19, val_values[index], width=0.36, color=COLORS[model],
                   alpha=0.38, edgecolor=COLORS[model], linewidth=0.8)
            ax.bar(index + 0.19, test_values[index], width=0.36, color=COLORS[model],
                   alpha=1, edgecolor=COLORS[model], linewidth=0.8)
        ax.set(title=title, xticks=positions, xticklabels=short_labels,
               ylim=(0, min(1.04, 1.18 * max(val_values.max(), test_values.max(), 0.1))))
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.16)
        ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='#555555', alpha=0.38, label='Validation'),
                        Patch(facecolor='#555555', alpha=1, label='Final test')],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle('Validation and final test: unchanged models and thresholds', y=1.02)
    fig.text(0.5, -0.015, 'Validation F1 informed threshold selection and is optimistic. '
             'Differences are descriptive; no uncertainty intervals or significance claims.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, figures / 'validation_test_comparison')


def generate_report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), f'Refusing to overwrite existing report directory: {output}')
    protocol, receipt, test, validation, predictions, identities, checks = validate_sources(root)
    output.mkdir(parents=True, exist_ok=False)
    curves = write_tables_and_curves(protocol, test, validation, predictions, output)
    write_figures(protocol, test, validation, curves, output)
    require(identities == {relative: sha256(root / relative) for relative in identities},
            'A saved input changed during report derivation')
    outputs = {str(path.relative_to(output)): sha256(path)
               for path in sorted(output.rglob('*')) if path.is_file()}
    derivation = {
        'status': 'complete', 'stage': 'post_test_reporting',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'generator_script_sha256': sha256(Path(__file__).resolve()),
        'source_sha256': identities, 'source_files_unchanged': True,
        'outputs_sha256': outputs, 'positive_class': 1,
        'thresholds': protocol['thresholds'],
        'cv_recommended_family_unchanged': protocol['cv_recommended_family'],
        'test_n': len(predictions), 'validation_n': int(validation.iloc[0]['n']),
        'checks': checks, 'models_loaded': False, 'model_inference_run': False,
        'training_run': False, 'raw_dataset_read': False, 'threshold_search_run': False,
        'data_access': 'Only the five saved final-evaluation and validation-metric files listed above.',
        'curve_notes': 'All score cutoffs are descriptive ROC/PR curve coordinates, not proposed '
                       'decisions. ROC starts at an infinite cutoff; the last PR point has no cutoff.',
        'comparison_notes': 'Deltas are final test minus selected validation. Count deltas reflect '
                            'partition sizes as well as rates. Higher is better for AP, ROC AUC, '
                            'precision, recall, F1, accuracy and balanced accuracy; lower is better '
                            'for Brier score and log loss.',
        'limitations': 'Validation F1 was optimized on the same validation sample. Differences are '
                       'descriptive, without confidence intervals or significance tests. Historical '
                       'training-time source provenance remains incomplete; this derivation does '
                       'not reconstruct it. Feature importance is not causality.',
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                     'matplotlib': matplotlib.__version__,
                     'sklearn': __import__('sklearn').__version__},
    }
    (output / 'derivation_receipt.json').write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root containing the saved one-shot final outputs')
    parser.add_argument('--output', type=Path, default=None,
                        help='New output directory (default: ROOT/results/member_d/final_test)')
    args = parser.parse_args()
    output = args.output or args.root / 'results/member_d/final_test'
    print(generate_report(args.root, output))


if __name__ == '__main__':
    main()
