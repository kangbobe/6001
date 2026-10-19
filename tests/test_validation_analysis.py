"""Synthetic, hand-checkable diagnostics; no real partition is loaded or model fitted."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data import FEATURES
from src.validation_analysis import (
    write_curves_and_confusions, write_error_analysis, write_permutation_importance,
)


def _validation_frame():
    X = pd.DataFrame({
        "age": [29, 30, 39, 40, 59, 60],
        "job": ["unknown", "admin.", "admin.", "admin.", "retired", "retired"],
        "marital": ["single", "married", "single", "married", "married", "married"],
        "education": ["secondary"] * 6, "default": ["no"] * 6,
        "balance": [-1, 0, 1, 999, 1000, 5000], "housing": ["no"] * 6, "loan": ["no"] * 6,
        "pdays": [-1, 0, -1, 1, 5, -1], "previous": [0, 1, 2, 4, 5, 6],
        "poutcome": ["unknown", "failure", "unknown", "success", "failure", "unknown"],
    }, index=pd.Index([101, 103, 106, 108, 109, 112], name="row_id"))
    return X.loc[:, FEATURES]


class AlreadyFittedRule(ClassifierMixin, BaseEstimator):
    """Deterministic classifier whose only predictive feature is age."""
    def __init__(self, constant=False):
        self.constant = constant
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        raise AssertionError("Analysis must never fit or retrain")

    def predict_proba(self, X):
        p = np.full(len(X), .25) if self.constant else np.where(X.age >= 40, .9, .1)
        return np.column_stack([1 - p, p])


def test_curves_and_confusions_use_scores_and_inclusive_threshold(tmp_path):
    tables = write_curves_and_confusions(
        {"model": np.array([.1, .8, .4, .9]), "dummy": np.full(4, .5)},
        np.array([0, 0, 1, 1]), {"model": .4, "dummy": .5}, tmp_path)
    confusion = tables["confusion_matrices"].set_index(["model", "rule"])
    baseline = confusion.loc[("model", "baseline_0_5")]
    selected = confusion.loc[("model", "selected")]
    assert tuple(baseline[["tn", "fp", "fn", "tp"]]) == (1, 1, 1, 1)
    assert tuple(selected[["tn", "fp", "fn", "tp"]]) == (1, 1, 0, 2)
    assert selected["f1"] == pytest.approx(.8)
    assert baseline.average_precision == selected.average_precision == pytest.approx(5 / 6)
    assert baseline.roc_auc == selected.roc_auc == .75
    assert confusion.loc[("dummy", "selected"), "tp"] == 2  # p == threshold is positive.
    for _, points in tables["pr_points"].groupby("model", sort=False):
        assert points.iloc[-1].terminal_point
        assert np.isnan(points.iloc[-1].threshold)
        assert (points.iloc[-1].precision, points.iloc[-1].recall) == (1, 0)
        assert points.terminal_point.sum() == 1
    for _, points in tables["roc_points"].groupby("model", sort=False):
        assert np.isinf(points.iloc[0].threshold)
        assert (points.iloc[0].fpr, points.iloc[0].tpr) == (0, 0)
        assert (points.iloc[-1].fpr, points.iloc[-1].tpr) == (1, 1)
    for file in ("roc_points.csv", "pr_points.csv", "confusion_matrices.csv"):
        assert len(pd.read_csv(tmp_path / file)) > 0
    for file in ("roc_pr.png", "confusion_matrices.png"):
        assert (tmp_path / "figures" / file).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("predictions,y,thresholds", [
    ({"a": [.1, .5]}, [0, 1], {"b": .5}),
    ({"a": [.1, np.nan]}, [0, 1], {"a": .5}),
    ({"a": [.1]}, [0, 1], {"a": .5}),
    ({"a": [.1, .2]}, [0, 0], {"a": .5}),
    ({"a": [.1, .2]}, [0, 1], {"a": 1.1}),
])
def test_invalid_curve_inputs_fail_before_output(tmp_path, predictions, y, thresholds):
    output = tmp_path / "not_created"
    with pytest.raises(ValueError):
        write_curves_and_confusions(predictions, y, thresholds, output)
    assert not output.exists()


def test_error_counts_bins_denominators_and_pairwise_overlap(tmp_path):
    X = _validation_frame()
    y = pd.Series([0, 0, 1, 1, 0, 1], index=X.index)
    scores = {"model": np.array([.2, .5, .5, .4, .8, .9]), "dummy": np.full(6, .1)}
    tables = write_error_analysis(X, y, scores, {"model": .5, "dummy": .5}, tmp_path)
    groups = tables["error_groups"]
    assert set(groups.feature) == set(FEATURES)
    for (_, _), group in groups.groupby(["model", "feature"]):
        assert group.n.sum() == len(X)
        assert group.positives.sum() == 3
        assert group.negatives.sum() == 3
        assert group[["tn", "fp", "fn", "tp"]].to_numpy().sum() == len(X)
    selected = groups[groups.model == "model"].set_index(["feature", "group"])
    assert selected.loc[("age", "30–39"), "n"] == 2
    assert selected.loc[("balance", "0"), "fp"] == 1
    assert selected.loc[("balance", "1–999"), "n"] == 2
    assert selected.loc[("balance", ">=5000"), "tp"] == 1
    assert selected.loc[("previous", "2–4"), "n"] == 2
    assert selected.loc[("previous", ">=5"), "n"] == 2
    assert selected.loc[("pdays", "Prior contact"), "n"] == 3
    assert selected.loc[("job", "unknown"), "n"] == 1
    assert np.isnan(selected.loc[("age", "<30"), "recall"])
    assert np.isnan(selected.loc[("age", ">=60"), "false_positive_rate"])
    assert groups.loc[groups.model == "dummy", "precision"].isna().all()
    assert groups.small_positive_denominator.all()
    assert groups.small_negative_denominator.all()
    rows = tables["row_errors"].query("model == 'model'").set_index("row_id")
    assert rows.loc[103, "outcome"] == "FP"
    assert rows.loc[106, "outcome"] == "TP"
    assert rows.loc[108, "outcome"] == "FN"
    assert list(rows.index) == list(X.index)
    pair = tables["model_disagreements"].iloc[0]
    assert pair.disagreement_n == 4
    assert pair.a_correct_b_wrong == pair.b_correct_a_wrong == 2
    assert pair.both_false_negative == 1
    assert pair.a_only_false_positive == pair.b_only_false_negative == 2
    assert pair.both_wrong == pair.both_false_positive + pair.both_false_negative
    assert (tmp_path / "figures/error_groups.png").is_file()


def test_error_analysis_rejects_misaligned_or_duplicate_row_ids(tmp_path):
    X = _validation_frame()
    y = pd.Series([0, 0, 1, 1, 0, 1], index=X.index)
    with pytest.raises(ValueError, match="identical unique"):
        write_error_analysis(X, y.iloc[::-1], {"model": np.full(6, .5)}, {"model": .5}, tmp_path)
    X.index = [0, 0, 2, 3, 4, 5]
    y.index = X.index
    with pytest.raises(ValueError, match="identical unique"):
        write_error_analysis(X, y, {"model": np.full(6, .5)}, {"model": .5}, tmp_path)


def test_permutation_uses_fitted_models_is_repeatable_and_has_correct_sd(tmp_path):
    X = _validation_frame()
    y = pd.Series([0, 0, 0, 1, 1, 1], index=X.index)
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    models = {"model": AlreadyFittedRule(), "dummy": AlreadyFittedRule(constant=True)}
    first = write_permutation_importance(models, X, y, tmp_path / "first", n_repeats=5, random_state=42)
    second = write_permutation_importance(models, X, y, tmp_path / "second", n_repeats=5, random_state=42)
    pd.testing.assert_frame_equal(first["permutation_repeats"], second["permutation_repeats"])
    pd.testing.assert_frame_equal(first["permutation_summary"], second["permutation_summary"])
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
    summary = first["permutation_summary"].set_index(["model", "feature"])
    assert summary.loc[("model", "age"), "importance_mean"] > 0
    assert (summary.loc["dummy", "importance_mean"] == 0).all()
    assert (summary.loc["dummy", "importance_std"] == 0).all()
    assert (summary.loc["model"].drop("age").importance_mean == 0).all()
    assert summary.loc[("model", "pdays"), "interpretation"] == "prior-contact flag only"
    for (name, feature), repeats in first["permutation_repeats"].groupby(["model", "feature"]):
        row = summary.loc[(name, feature)]
        assert row.importance_mean == pytest.approx(repeats.ap_decrease.mean())
        assert row.importance_std == pytest.approx(repeats.ap_decrease.std(ddof=0))
        assert row.std_ddof == 0
        assert row.baseline_ap == (1 if name == "model" else .5)
        np.testing.assert_allclose(repeats.permuted_ap + repeats.ap_decrease, repeats.baseline_ap)
    assert (tmp_path / "first/figures/permutation_importance.png").is_file()


@pytest.mark.parametrize("change, message", [("classes", "classes_"), ("repeats", "repetitions"), ("seed", "seed")])
def test_invalid_permutation_contract_fails_before_output(tmp_path, change, message):
    X = _validation_frame()
    y = pd.Series([0, 0, 0, 1, 1, 1], index=X.index)
    model = AlreadyFittedRule()
    kwargs = {}
    if change == "classes":
        model.classes_ = np.array([1, 0])
    if change == "repeats":
        kwargs["n_repeats"] = 1
    if change == "seed":
        kwargs["random_state"] = None
    with pytest.raises(ValueError, match=message):
        write_permutation_importance({"model": model}, X, y, tmp_path / "not_created", **kwargs)
    assert not (tmp_path / "not_created").exists()
