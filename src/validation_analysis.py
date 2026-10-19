"""Validation-only descriptive diagnostics for frozen fitted pipelines.

The caller supplies already aligned validation data; this module has no data loader,
model fitting, threshold selection, or final-test entry point. Error-group bins are
fixed below rather than estimated from observed outcomes. Permutation standard
deviations describe shuffle variability for one fitted model, not confidence intervals.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits

from src.data import FEATURES
from src.metrics import binary_metrics
from src.preprocessing import CATEGORICAL


MODEL_LABELS = {
    "dummy": "Dummy", "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree", "random_forest": "Random forest", "mlp": "MLP",
}
SMALL_DENOMINATOR = 30
GROUP_BINS = {
    "age": ([-np.inf, 30, 40, 50, 60, np.inf], ["<30", "30–39", "40–49", "50–59", ">=60"]),
    "balance": ([-np.inf, 0, 1, 1000, 5000, np.inf], ["<0", "0", "1–999", "1000–4999", ">=5000"]),
    "previous": ([-np.inf, 1, 2, 5, np.inf], ["0", "1", "2–4", ">=5"]),
}


def _label(name):
    return MODEL_LABELS.get(name, name)


def _output_directory(output):
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    return output


def _validate_predictions(predictions, y, thresholds):
    if not predictions or set(predictions) != set(thresholds):
        raise ValueError("Predictions and thresholds must contain the same non-empty model set")
    # The shared metric implementation enforces binary classes, shape, finite
    # scores, score range, and threshold range, using yes=1 and >= consistently.
    checked = {}
    for name, values in predictions.items():
        binary_metrics(y, values, thresholds[name])
        checked[name] = np.asarray(values, dtype=float)
    return np.asarray(y), checked


def _validate_xy(X, y):
    if not isinstance(X, pd.DataFrame) or list(X.columns) != FEATURES:
        raise ValueError("Expected exactly the frozen 11 source features in order")
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("X and y must have identical unique row_id indices")
    if X.isna().any().any():
        raise ValueError("Unexpected missing feature values; literal unknown is retained")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes must be present")


def _save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_curves_and_confusions(predictions, y, thresholds, output):
    """Write score-based curves and confusion matrices at 0.5/selected thresholds.

    PR's terminal point has no associated threshold, represented by a blank CSV
    value; ROC's initial all-negative point has threshold infinity. AP is computed
    with sklearn average_precision_score, never trapezoidal PR integration.
    """
    y, predictions = _validate_predictions(predictions, y, thresholds)
    output = _output_directory(output)
    roc_rows, pr_rows, confusion_rows = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    for name, scores in predictions.items():
        fpr, tpr, roc_thresholds = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
        precision, recall, pr_thresholds = precision_recall_curve(y, scores, pos_label=1)
        baseline = binary_metrics(y, scores, 0.5)
        selected = binary_metrics(y, scores, thresholds[name])
        roc_rows.extend({"model": name, "point": i, "fpr": f, "tpr": t, "threshold": threshold}
                        for i, (f, t, threshold) in enumerate(zip(fpr, tpr, roc_thresholds)))
        pr_rows.extend({"model": name, "point": i, "precision": p, "recall": r,
                        "threshold": pr_thresholds[i] if i < len(pr_thresholds) else np.nan,
                        "terminal_point": i == len(pr_thresholds)}
                       for i, (p, r) in enumerate(zip(precision, recall)))
        for rule, metrics in (("baseline_0_5", baseline), ("selected", selected)):
            confusion_rows.append({"model": name, "rule": rule, **metrics})
        line, = axes[0].plot(fpr, tpr, label=f"{_label(name)} (AUC {selected['roc_auc']:.3f})")
        color = line.get_color()
        axes[0].scatter(selected["fp"] / (selected["fp"] + selected["tn"]),
                        selected["recall"], color=color, s=28, zorder=3)
        axes[1].step(recall, precision, where="post", color=color,
                     label=f"{_label(name)} (AP {selected['average_precision']:.3f})")
        # Dummy has no positive predictions at 0.5; precision is undefined for
        # that decision and its conventional metric value of zero is not plotted.
        if selected["tp"] + selected["fp"]:
            axes[1].scatter(selected["recall"], selected["precision"], color=color, s=28, zorder=3)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", linewidth=1, label="Chance ranking")
    axes[1].axhline(y.mean(), linestyle=":", color="0.4", linewidth=1,
                   label=f"Positive prevalence {y.mean():.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="Validation ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Validation Precision–Recall")
    for ax in axes:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Fixed fitted models; circles mark selected validation thresholds", fontsize=12)
    _save_figure(fig, output / "figures/roc_pr.png")
    roc_table, pr_table, confusion_table = map(pd.DataFrame, (roc_rows, pr_rows, confusion_rows))
    roc_table.to_csv(output / "roc_points.csv", index=False)
    pr_table.to_csv(output / "pr_points.csv", index=False)
    confusion_table.to_csv(output / "confusion_matrices.csv", index=False)

    fig, axes = plt.subplots(len(predictions), 2, figsize=(8, 2.55 * len(predictions)), squeeze=False,
                             layout="constrained")
    vmax = int(max((y == 0).sum(), (y == 1).sum()))
    for row, (name, _) in enumerate(predictions.items()):
        for col, rule in enumerate(("baseline_0_5", "selected")):
            metrics = confusion_table.loc[(confusion_table.model == name) & (confusion_table.rule == rule)].iloc[0]
            counts = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]], dtype=int)
            ax = axes[row, col]
            ax.imshow(counts, cmap="Blues", vmin=0, vmax=vmax)
            for (i, j), count in np.ndenumerate(counts):
                ax.text(j, i, f"{count:,}", ha="center", va="center",
                        color="white" if count > vmax / 2 else "black", fontsize=11)
            ax.set(xticks=[0, 1], xticklabels=["No (0)", "Yes (1)"], yticks=[0, 1],
                   yticklabels=["No (0)", "Yes (1)"], xlabel="Predicted label", ylabel="Actual label",
                   title=f"{_label(name)} | {'0.5 baseline' if col == 0 else 'selected'}\nt = {metrics.threshold:.6g}")
    fig.suptitle("Validation confusion counts; the selected rule was optimized on these rows", fontsize=11)
    _save_figure(fig, output / "figures/confusion_matrices.png")
    return {"roc_points": roc_table, "pr_points": pr_table, "confusion_matrices": confusion_table}


def _groups(X):
    groups = {}
    for feature, (bins, labels) in GROUP_BINS.items():
        groups[feature] = pd.cut(X[feature], bins=bins, labels=labels, right=False)
    groups["pdays"] = pd.Series(np.where(X.pdays.ge(0), "Prior contact", "No prior contact"), index=X.index)
    for feature in CATEGORICAL:
        groups[feature] = X[feature].astype(str)
    return groups


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def write_error_analysis(X, y, predictions, thresholds, output):
    """Write exploratory validation error groups, pairwise comparisons, and row errors.

    Each grouping partitions the validation sample. Rates carry their applicable
    denominators and flags when those denominators are below 30. Age/group
    differences are descriptive associations, not causal or fairness findings.
    """
    _validate_xy(X, y)
    y_array, predictions = _validate_predictions(predictions, y, thresholds)
    output = _output_directory(output)
    group_rows, row_tables, decisions = [], [], {}
    groups = _groups(X)
    for name, scores in predictions.items():
        predicted = scores >= thresholds[name]
        decisions[name] = predicted
        outcomes = np.select([(y_array == 0) & ~predicted, (y_array == 0) & predicted,
                              (y_array == 1) & ~predicted], ["TN", "FP", "FN"], default="TP")
        row_tables.append(pd.DataFrame({"model": name, "split": "val", "row_id": X.index,
                                        "y_true": y_array, "p_yes": scores,
                                        "threshold": thresholds[name], "prediction": predicted.astype(int),
                                        "outcome": outcomes}))
        for feature, values in groups.items():
            levels = list(values.cat.categories) if isinstance(values.dtype, pd.CategoricalDtype) else sorted(values.unique())
            for group in levels:
                mask = np.asarray(values == group)
                n = int(mask.sum())
                if n == 0:
                    continue
                counts = {kind.lower(): int(np.sum(outcomes[mask] == kind)) for kind in ("TN", "FP", "FN", "TP")}
                tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
                positives, negatives, selected = tp + fn, tn + fp, tp + fp
                group_rows.append({"model": name, "split": "val", "feature": feature, "group": str(group),
                    "threshold": thresholds[name], "n": n, "positives": positives, "negatives": negatives,
                    "predicted_positive": selected, **counts,
                    "positive_rate": positives / n, "selection_rate": selected / n,
                    "precision": _divide(tp, selected), "recall": _divide(tp, positives),
                    "false_negative_rate": _divide(fn, positives), "false_positive_rate": _divide(fp, negatives),
                    "f1": _divide(2 * tp, 2 * tp + fp + fn), "error_rate": (fp + fn) / n,
                    "small_n": n < SMALL_DENOMINATOR,
                    "small_positive_denominator": positives < SMALL_DENOMINATOR,
                    "small_negative_denominator": negatives < SMALL_DENOMINATOR,
                    "small_precision_denominator": selected < SMALL_DENOMINATOR})
    disagreements = []
    for a, b in combinations(predictions, 2):
        pa, pb = decisions[a], decisions[b]
        a_fp, b_fp = pa & (y_array == 0), pb & (y_array == 0)
        a_fn, b_fn = ~pa & (y_array == 1), ~pb & (y_array == 1)
        disagreements.append({"model_a": a, "model_b": b, "n": len(y_array),
            "threshold_a": thresholds[a], "threshold_b": thresholds[b],
            "disagreement_n": int((pa != pb).sum()), "disagreement_rate": float((pa != pb).mean()),
            "a_correct_b_wrong": int(((pa == y_array) & (pb != y_array)).sum()),
            "b_correct_a_wrong": int(((pb == y_array) & (pa != y_array)).sum()),
            "both_wrong": int(((pa != y_array) & (pb != y_array)).sum()),
            "both_false_positive": int((a_fp & b_fp).sum()),
            "a_only_false_positive": int((a_fp & ~b_fp).sum()),
            "b_only_false_positive": int((b_fp & ~a_fp).sum()),
            "both_false_negative": int((a_fn & b_fn).sum()),
            "a_only_false_negative": int((a_fn & ~b_fn).sum()),
            "b_only_false_negative": int((b_fn & ~a_fn).sum())})
    group_table = pd.DataFrame(group_rows)
    row_table = pd.concat(row_tables, ignore_index=True)
    disagreement_table = pd.DataFrame(disagreements)
    group_table.to_csv(output / "error_groups.csv", index=False)
    row_table.to_csv(output / "row_errors.csv", index=False)
    disagreement_table.to_csv(output / "model_disagreements.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for i, feature in enumerate(("age", "poutcome")):
        for j, (metric, title) in enumerate((("false_negative_rate", "False negative rate: FN / positives"),
                                            ("false_positive_rate", "False positive rate: FP / negatives"))):
            ax = axes[i, j]
            first = group_table[(group_table.feature == feature) & (group_table.model == next(iter(predictions)))]
            levels = list(first.group)
            x = np.arange(len(levels))
            for name in predictions:
                subset = group_table[(group_table.feature == feature) & (group_table.model == name)].set_index("group").loc[levels]
                ax.plot(x, subset[metric], marker="o", markersize=4, label=_label(name))
            ticklabels = [f"{row.group}\n+{row.positives}/−{row.negatives}" for row in first.itertuples()]
            ax.set(xticks=x, xticklabels=ticklabels, ylim=(-0.03, 1.03), ylabel=title,
                   title=f"{feature} | labels include positive/negative support")
            ax.grid(alpha=0.18)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Exploratory validation errors at selected thresholds\nAll feature groups and small-denominator flags (<30) are in error_groups.csv", fontsize=12)
    _save_figure(fig, output / "figures/error_groups.png")
    return {"error_groups": group_table, "model_disagreements": disagreement_table, "row_errors": row_table}


def write_permutation_importance(models, X, y, output, n_repeats=10, random_state=42):
    """Permute raw source columns through existing pipelines; never fit an estimator.

    Importance = original validation AP minus AP after a marginal column shuffle.
    The pdays pipeline transformation retains only prior-contact status. Correlated
    features may substitute for each other; shuffling can create implausible
    combinations. Neither importance nor its sign establishes causality.
    """
    _validate_xy(X, y)
    if not models:
        raise ValueError("At least one fitted model is required")
    if type(n_repeats) is not int or n_repeats < 2:
        raise ValueError("At least two permutation repetitions are required")
    if type(random_state) is not int:
        raise ValueError("An explicit integer random seed is required")
    # Reject any class-order issue before writing output or evaluating a model.
    for model in models.values():
        if not np.array_equal(getattr(model, "classes_", None), [0, 1]):
            raise ValueError("Fitted model classes_ must equal [0, 1]; yes is class 1")
    output = _output_directory(output)
    repeat_rows, summary_rows = [], []
    for name, model in models.items():
        with threadpool_limits(limits=1):
            scores = np.asarray(model.predict_proba(X))
            if scores.shape != (len(y), 2):
                raise ValueError("Expected predict_proba shape (n_validation, 2)")
            binary_metrics(y, scores[:, 1])
            baseline_ap = float(average_precision_score(y, scores[:, 1]))
            importance = permutation_importance(model, X, y, scoring="average_precision",
                                                n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        for i, feature in enumerate(X.columns):
            values = importance.importances[i]
            if not np.isfinite(values).all():
                raise ValueError("Non-finite permutation importance")
            for repeat, value in enumerate(values, start=1):
                repeat_rows.append({"model": name, "split": "val", "feature": feature, "repeat": repeat,
                    "baseline_ap": baseline_ap, "permuted_ap": baseline_ap - float(value),
                    "ap_decrease": float(value), "random_state": random_state})
            summary_rows.append({"model": name, "split": "val", "feature": feature,
                "interpretation": "prior-contact flag only" if feature == "pdays" else "raw source field",
                "baseline_ap": baseline_ap, "importance_mean": float(values.mean()),
                "importance_std": float(values.std(ddof=0)), "std_ddof": 0,
                "n_repeats": n_repeats, "random_state": random_state,
                "uncertainty": "permutation variability, not a confidence interval"})
    repeat_table, summary_table = pd.DataFrame(repeat_rows), pd.DataFrame(summary_rows)
    repeat_table.to_csv(output / "permutation_repeats.csv", index=False)
    summary_table.to_csv(output / "permutation_summary.csv", index=False)
    ncols = min(len(models), 3)
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), squeeze=False, layout="constrained")
    for ax, name in zip(axes.flat, models):
        subset = summary_table[summary_table.model == name].sort_values("importance_mean", ascending=True)
        positions = np.arange(len(subset))
        labels = ["pdays (contact flag)" if f == "pdays" else f for f in subset.feature]
        ax.barh(positions, subset.importance_mean, xerr=subset.importance_std,
                color="#3d789e", ecolor="#263846", capsize=3)
        ax.set(yticks=positions, yticklabels=labels, xlabel="Decrease in validation Average Precision",
               title=f"{_label(name)} | baseline AP {subset.baseline_ap.iloc[0]:.3f}")
        ax.axvline(0, color="0.3", linewidth=0.8)
        ax.grid(axis="x", alpha=0.18)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle(f"Validation permutation importance | {n_repeats} shuffles, seed {random_state}\nBars: mean AP decrease; error bars: shuffle SD (ddof=0), not confidence intervals", fontsize=12)
    _save_figure(fig, output / "figures/permutation_importance.png")
    return {"permutation_repeats": repeat_table, "permutation_summary": summary_table}
