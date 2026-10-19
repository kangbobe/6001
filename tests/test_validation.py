"""Member D validation checks; every observation here is synthetic.

The final-gate checks inject synthetic data and model loaders. They never call
the command-line unlock, deserialize handed-off pipelines, or read raw data.
"""
from fractions import Fraction
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.final_evaluation import EvaluationRefused, MODEL_SET, run_final_evaluation
from src.validation import select_threshold, threshold_sweep, validate_predictions
import src.validation as validation


def test_threshold_equality_is_inclusive_and_curve_counts_are_consistent():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 0.5, 0.5, 1.0])
    sweep = threshold_sweep(y, p)
    assert {'threshold', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
            'f1', 'selected_n'}.issubset(sweep.columns)
    for row in sweep.itertuples(index=False):
        predicted = p >= row.threshold
        expected = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        np.testing.assert_array_equal([row.tn, row.fp, row.fn, row.tp], expected)
        assert row.selected_n == predicted.sum()
        assert row.f1 == pytest.approx(f1_score(y, predicted, zero_division=0))
        assert row.tn + row.fp + row.fn + row.tp == len(y)
    at_one = sweep.loc[sweep.threshold == 1].iloc[0]
    assert (at_one.tp, at_one.fn, at_one.fp, at_one.tn) == (1, 1, 0, 2)


def test_equal_optimal_f1_uses_higher_threshold_without_splitting_score_ties():
    # Predicting all four or only the last observation both yields F1 = 2/3.
    y = [1, 0, 0, 1]
    p = [.1, .2, .3, .4]
    threshold = select_threshold(y, p)
    assert threshold == pytest.approx(.35)
    np.testing.assert_array_equal(np.asarray(p) >= threshold, [False, False, False, True])


def test_repeated_scores_move_together():
    y = [0, 1, 0, 1, 1, 0]
    p = [.1, .1, .6, .6, .6, .9]
    sweep = threshold_sweep(y, p)
    assert set(sweep.selected_n) == {0, 1, 4, 6}
    for threshold in sweep.threshold:
        predicted = np.asarray(p) >= threshold
        assert predicted[0] == predicted[1]
        assert predicted[2] == predicted[3] == predicted[4]


def test_adjacent_float_midpoint_uses_right_endpoint_when_midpoint_rounds_left():
    left = .5
    right = np.nextafter(left, 1.0)
    assert left + (right - left) / 2 == left
    threshold = select_threshold([0, 1], [left, right])
    assert threshold == right
    assert f1_score([0, 1], np.asarray([left, right]) >= threshold) == 1


@pytest.mark.parametrize('score', [0.0, .117, 1.0])
def test_constant_scores_select_all_rows_and_remain_in_allowed_range(score):
    y = [0, 0, 1, 1, 0]
    p = np.full(len(y), score)
    threshold = select_threshold(y, p)
    assert 0 <= threshold <= 1
    assert (p >= threshold).all()
    # At score 1, both candidates select everybody: the higher threshold wins.
    assert threshold == (1.0 if score == 1.0 else 0.0)


def test_threshold_search_matches_independent_bruteforce_and_all_attainable_decisions():
    rng = np.random.default_rng(20261019)
    for n in (4, 9, 31):
        for _ in range(12):
            y = rng.integers(0, 2, size=n)
            y[:2] = [0, 1]
            p = rng.choice([0, .1, .3, .6, .9, 1], size=n)
            sweep = threshold_sweep(y, p)
            score_levels = np.unique(p)
            # Observed scores induce every nonempty attainable positive set.
            observed_cutoffs = np.unique(np.r_[0., score_levels, 1.])
            attainable = {tuple(p >= cutoff) for cutoff in observed_cutoffs}
            actual = {tuple(p >= cutoff) for cutoff in sweep.threshold}
            assert actual == attainable
            independent = []
            for cutoff in sweep.threshold:
                predicted = p >= cutoff
                tp = int(np.sum((y == 1) & predicted))
                fp = int(np.sum((y == 0) & predicted))
                fn = int(np.sum((y == 1) & ~predicted))
                independent.append((Fraction(2 * tp, 2 * tp + fp + fn), float(cutoff)))
            assert select_threshold(y, p) == max(independent)[1]


@pytest.mark.parametrize('y,p', [
    ([], []), ([0, 1], [.2]), ([0, 0], [.1, .2]),
    ([0, 2], [.1, .9]), ([0, 1], [.1, np.nan]),
    ([0, 1], [.1, np.inf]), ([0, 1], [-.1, .9]),
    ([0, 1], [.1, 1.1]), ([0, 1], [[.1], [.9]]),
])
def test_threshold_selection_rejects_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        select_threshold(y, p)


def _predictions():
    return pd.DataFrame({
        'row_id': [42, 7, 90, 13],
        'split': ['val'] * 4,
        'model': ['mlp'] * 4,
        'y_true': [0, 1, 0, 1],
        'p_yes': [.1, .5, .8, .9],
        'prediction_at_0_5': [0, 1, 1, 1],
    })


def test_valid_predictions_preserve_frozen_non_sorted_row_order():
    frame = _predictions()
    y = pd.Series(frame.y_true.to_numpy(), index=frame.row_id)
    p = validate_predictions(frame, 'mlp', frame.row_id.to_numpy(), y)
    assert isinstance(p, np.ndarray)
    np.testing.assert_array_equal(p, frame.p_yes.to_numpy())
    np.testing.assert_array_equal(
        validate_predictions(frame, 'mlp', frame.row_id.to_numpy()), p)


@pytest.mark.parametrize('mutate', [
    lambda f: f.iloc[::-1].reset_index(drop=True),
    lambda f: f.iloc[:-1].copy(),
    lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True),
    lambda f: f.assign(row_id=[42, 7, 90, 90]),
    lambda f: f.assign(row_id=[42, 7, 90, 14]),
    lambda f: f.assign(split=['val', 'val', 'train', 'val']),
    lambda f: f.assign(model=['mlp', 'mlp', 'dummy', 'mlp']),
    lambda f: f.assign(y_true=[0, 1, 0, 2]),
    lambda f: f.assign(y_true=[0, 0, 0, 0]),
    lambda f: f.assign(p_yes=[.1, .5, np.nan, .9]),
    lambda f: f.assign(p_yes=[.1, .5, np.inf, .9]),
    lambda f: f.assign(p_yes=[-.1, .5, .8, .9]),
    lambda f: f.assign(p_yes=[.1, .5, .8, 1.1]),
    lambda f: f.assign(prediction_at_0_5=[0, 0, 1, 1]),
    lambda f: f.drop(columns='split'),
    lambda f: f.assign(unexpected='extra'),
])
def test_validation_predictions_reject_malformed_or_misaligned_records(mutate):
    expected = _predictions()
    with pytest.raises(ValueError):
        validate_predictions(mutate(expected.copy()), 'mlp', expected.row_id.to_numpy())


def test_validation_predictions_reject_disagreement_with_validation_labels():
    frame = _predictions()
    with pytest.raises(ValueError):
        validate_predictions(frame, 'mlp', frame.row_id.to_numpy(), np.array([1, 0, 0, 1]))


def test_validation_predictions_reject_misindexed_series_even_when_labels_coincide():
    frame = _predictions()
    y = pd.Series(frame.y_true.to_numpy(), index=[90, 13, 42, 7])
    with pytest.raises(ValueError, match='index|alignment|aligned'):
        validate_predictions(frame, 'mlp', frame.row_id.to_numpy(), y)


def test_validation_reader_decodes_only_selected_rows_in_frozen_order(tmp_path, monkeypatch):
    columns = [*validation.FEATURES, 'y']
    root = tmp_path / 'synthetic_data'
    (root / 'data/raw').mkdir(parents=True)
    (root / 'config').mkdir()
    (root / 'config/dataset_lock.json').write_text(json.dumps({'columns': columns}), encoding='utf-8')
    first = '20;admin.;single;secondary;no;100;no;no;-1;0;unknown;no\n'.encode()
    third = '60;retired;married;primary;no;500;yes;no;20;1;success;yes\n'.encode()
    # A full CSV parse would fail on this deliberately invalid UTF-8 record.
    excluded = b'\xff\xfe;unparseable;excluded;record\n'
    raw = (root / 'data/raw/bank-full.csv')
    raw.write_bytes((';'.join(columns) + '\n').encode() + first + excluded + third)
    identities_called = []
    monkeypatch.setattr(validation, 'verify_input_identities',
                        lambda supplied: identities_called.append(supplied))
    monkeypatch.setattr(validation, 'load_split_indices', lambda supplied: {
        'val': np.array([2, 0]), 'test': np.array([1]), 'train': np.array([], dtype=int),
    })
    X, y = validation.load_validation(root)
    assert identities_called == [root]
    assert X.index.tolist() == y.index.tolist() == [2, 0]
    assert X.age.tolist() == [60, 20]
    assert y.tolist() == [1, 0]
    assert list(X.columns) == validation.FEATURES


def test_input_identity_change_refuses_before_selective_row_loading(tmp_path):
    relative = 'data/splits/synthetic_indices.json'
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    original = b'original frozen input'
    target.write_bytes(b'changed input bytes')
    manifest = tmp_path / 'results/experiment/post_training_review/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'inputs': [{
        'path': relative, 'bytes': len(original),
        'sha256': hashlib.sha256(original).hexdigest(),
    }]}), encoding='utf-8')
    with pytest.raises(ValueError, match='identity mismatch'):
        validation.load_validation(tmp_path)


def test_fitting_guard_blocks_pipeline_and_nested_estimators_and_restores_methods():
    model = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression())])
    X, y = [[0.], [1.]], [0, 1]
    original_fit = Pipeline.fit
    original_transform_fit = StandardScaler.fit_transform
    original_partial_fit = StandardScaler.partial_fit
    with validation.forbid_fitting({'synthetic': model}):
        for operation in (
            lambda: model.fit(X, y),
            lambda: model.named_steps['model'].fit(X, y),
            lambda: model.named_steps['scale'].fit_transform(X),
            lambda: model.named_steps['scale'].partial_fit(X),
        ):
            with pytest.raises(RuntimeError, match='never fit or retrain'):
                operation()
    assert Pipeline.fit is original_fit
    assert StandardScaler.fit_transform is original_transform_fit
    assert StandardScaler.partial_fit is original_partial_fit


def test_fitting_guard_restores_methods_after_exception():
    original_fit = LogisticRegression.fit
    with pytest.raises(RuntimeError, match='synthetic inference failure'):
        with validation.forbid_fitting({'synthetic': LogisticRegression()}):
            raise RuntimeError('synthetic inference failure')
    assert LogisticRegression.fit is original_fit


def test_existing_stage_refused_before_reading_configuration_or_models(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    stage = output / 'evaluation'
    stage.mkdir(parents=True)
    sentinel = stage / 'previous.txt'
    sentinel.write_text('preserve this evidence', encoding='utf-8')
    monkeypatch.setattr(validation, 'load_handoff',
                        lambda root: pytest.fail('Must refuse before loading handoff'))
    with pytest.raises(FileExistsError, match='Preserve previous stage'):
        validation.run_validation(stage='evaluation', output=output, root=tmp_path)
    assert sentinel.read_text(encoding='utf-8') == 'preserve this evidence'


@pytest.mark.parametrize('protected', ['data', 'results/experiment'])
@pytest.mark.parametrize('destination_kind', ['stage', 'protocol'])
def test_validation_refuses_destinations_inside_frozen_directories_before_any_loading(
        tmp_path, monkeypatch, protected, destination_kind):
    root = tmp_path / 'synthetic_project'
    forbidden = root / protected
    forbidden.mkdir(parents=True)
    kwargs = {'stage': 'evaluation', 'root': root, 'output': tmp_path / 'safe_output'}
    if destination_kind == 'stage':
        kwargs['output'] = forbidden / 'new_analysis'
    else:
        kwargs['protocol_output'] = forbidden / 'new_protocol.json'
    monkeypatch.setattr(validation, 'load_handoff',
                        lambda supplied: pytest.fail('Unsafe paths must fail before loading'))
    with pytest.raises(ValueError):
        validation.run_validation(**kwargs)
    assert list(forbidden.iterdir()) == []


def test_validation_resolves_symlink_before_rejecting_frozen_output_path(tmp_path):
    root = tmp_path / 'synthetic_project'
    frozen = root / 'results/experiment'
    frozen.mkdir(parents=True)
    alias = tmp_path / 'apparently_safe_output'
    alias.symlink_to(frozen, target_is_directory=True)
    with pytest.raises(ValueError):
        validation.run_validation(stage='evaluation', output=alias, root=root)
    assert list(frozen.iterdir()) == []


class _ReadOnlyModel(BaseEstimator):
    """Inference-only synthetic estimator: no training or artifact deserialization."""
    def predict_proba(self, X):
        p = np.array([.1, .8, .3, .9])[:len(X)]
        return np.column_stack([1 - p, p])


class _InvalidProbabilityModel(BaseEstimator):
    classes_ = np.array([0, 1])

    def __init__(self, invalid_row=0, delta=-1e-13):
        self.invalid_row = invalid_row
        self.delta = delta

    def predict_proba(self, X):
        p = np.array([0., .5, .8, 1.])
        p[self.invalid_row] += self.delta
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize('row,delta', [(0, -1e-13), (3, 1e-13)])
def test_handoff_rejects_inferred_probability_outside_range_even_within_reproduction_tolerance(
        tmp_path, monkeypatch, row, delta):
    frame = _predictions().assign(p_yes=[0., .5, .8, 1.])
    X = pd.DataFrame({'age': [20, 30, 40, 50]}, index=frame.row_id)
    y = pd.Series(frame.y_true.to_numpy(), index=X.index)
    metrics = validation.binary_metrics(y, frame.p_yes)
    summary = pd.DataFrame([{'model': name, **metrics} for name in MODEL_SET])
    model = _InvalidProbabilityModel(row, delta)
    # These weak checks pass; the explicit bounds check must still reject.
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], frame.p_yes, atol=1e-12)

    def fake_read_json(path):
        if path.name == 'posthoc_verification.json':
            return {'project_files': [], 'experiment_files': []}
        if path.name == 'model_metadata.json':
            return {'input_features': validation.FEATURES, 'fit_n': 27126,
                    'config_id': 0, 'validation_metrics_at_0_5': metrics}
        return {}

    def fake_read_csv(path, **kwargs):
        if path.name == 'development_summary.csv':
            return summary.copy()
        if path.name == 'cv_candidates.csv':
            return pd.DataFrame({'config_id': [0], 'mean_average_precision': [.5]})
        return frame.assign(model=path.parent.name)

    monkeypatch.setattr(validation, 'read_json', fake_read_json)
    monkeypatch.setattr(validation, '_validate_experiment', lambda *args: None)
    monkeypatch.setattr(validation, 'verify_input_identities', lambda *args: {})
    monkeypatch.setattr(validation, 'load_validation', lambda *args: (X, y))
    monkeypatch.setattr(validation.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(validation.joblib, 'load', lambda path: model)
    with pytest.raises(ValueError, match='probability'):
        validation.load_handoff(tmp_path)


def test_stage_preserves_failure_and_refuses_success_receipt_if_inputs_change(tmp_path, monkeypatch):
    import src.validation_analysis as analysis

    root, output = tmp_path / 'project', tmp_path / 'output'
    (root / 'config').mkdir(parents=True)
    config = {
        'partition': 'val', 'test_evaluated': False, 'retraining_allowed': False,
        'permutation': {'scoring': 'average_precision', 'n_repeats': 10,
                        'random_state': 42, 'n_jobs': 1, 'max_samples': 1.0, 'std_ddof': 0},
    }
    config_file = root / 'config/member_d_analysis.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')
    previous = output / 'evaluation'
    previous.mkdir(parents=True)
    before = {'synthetic_protected_input': 'old_digest'}
    (previous / 'receipt.json').write_text(json.dumps({
        'protected_files_before': before,
        'analysis_config_sha256': validation.sha256(config_file),
        'thresholds': {'synthetic': .5},
    }), encoding='utf-8')
    snapshots = iter([before, {'synthetic_protected_input': 'new_digest'}])
    monkeypatch.setattr(validation, 'protected_identities', lambda supplied: next(snapshots))
    X = pd.DataFrame({'age': [20, 30, 40, 50]}, index=[11, 12, 13, 14])
    y = pd.Series([0, 1, 0, 1], index=X.index)
    model = _ReadOnlyModel()
    monkeypatch.setattr(validation, 'load_handoff', lambda supplied: (
        X, y, {'synthetic': model.predict_proba(X)[:, 1]}, {'synthetic': model}, []))

    def write_synthetic_errors(X, y, predictions, thresholds, destination):
        (destination / 'partial_output.txt').write_text('retain failure evidence', encoding='utf-8')

    monkeypatch.setattr(analysis, 'write_error_analysis', write_synthetic_errors)
    with pytest.raises(RuntimeError, match='Protected A/B/C files changed'):
        validation.run_validation(stage='errors', output=output, root=root)
    stage = output / 'errors'
    assert (stage / 'partial_output.txt').read_text(encoding='utf-8') == 'retain failure evidence'
    assert not (stage / 'receipt.json').exists()
    with pytest.raises(FileExistsError, match='Preserve previous stage'):
        validation.run_validation(stage='errors', output=output, root=root)


def _synthetic_final_fixture(tmp_path):
    experiment = tmp_path / 'synthetic_experiment'
    experiment.mkdir()
    hashes = {}
    for name in MODEL_SET:
        model_dir = experiment / name
        model_dir.mkdir()
        # The injected model loader below never deserializes these fixture bytes.
        payload = f'synthetic pipeline identity for {name}'.encode()
        (model_dir / 'pipeline.joblib').write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (model_dir / 'model_metadata.json').write_text(json.dumps({
            'model': name, 'refit_partition': 'train', 'test_evaluated': False,
            'classes': [0, 1],
        }), encoding='utf-8')
    (experiment / 'completion.json').write_text(json.dumps({
        'status': 'complete', 'test_evaluated': False, 'models': MODEL_SET,
    }), encoding='utf-8')
    (experiment / 'run_protocol.json').write_text(json.dumps({
        'test_evaluated': False,
    }), encoding='utf-8')
    protocol = tmp_path / 'synthetic_protocol.json'
    protocol.write_text(json.dumps({
        'frozen': True, 'owner': 'synthetic fixture',
        'frozen_at': '2026-10-19T00:00:00+00:00',
        'decision_rule': 'Synthetic gate check only.',
        'feature_policy': 'pre_campaign_history_flag_v1',
        'models': MODEL_SET, 'thresholds': {name: .5 for name in MODEL_SET},
        'pipeline_sha256': hashes,
    }), encoding='utf-8')
    X = pd.DataFrame({'age': [20, 30, 40, 50]}, index=[11, 12, 13, 14])
    y = pd.Series([0, 1, 0, 1], index=X.index)
    return experiment, protocol, X, y


def test_final_gate_rejects_label_misalignment_before_any_model_inference(tmp_path):
    experiment, protocol, X, y = _synthetic_final_fixture(tmp_path)
    model_calls = []
    with pytest.raises(ValueError, match='aligned'):
        run_final_evaluation(
            unlock_test=True, protocol_path=protocol, experiment_path=experiment,
            load_partition_fn=lambda *args, **kwargs: (X, y.iloc[::-1]),
            features=['age'], model_loader=lambda path: model_calls.append(path))
    assert model_calls == []
    assert (experiment / 'final_test').is_dir()


def test_final_gate_rejects_wrong_class_order_and_preserves_failed_attempt(tmp_path):
    experiment, protocol, X, y = _synthetic_final_fixture(tmp_path)

    class ReversedClasses:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            raise AssertionError('Class guard must reject before inference')

    kwargs = dict(
        unlock_test=True, protocol_path=protocol, experiment_path=experiment,
        load_partition_fn=lambda *args, **kw: (X, y), features=['age'],
        model_loader=lambda path: ReversedClasses())
    with pytest.raises(ValueError, match='classes'):
        run_final_evaluation(**kwargs)
    output = experiment / 'final_test'
    assert output.is_dir()
    assert not (output / 'evaluation_receipt.json').exists()
    # Failure leaves the fixed destination in place: a later attempt cannot hide it.
    with pytest.raises(EvaluationRefused, match='refusing to overwrite'):
        run_final_evaluation(**kwargs)
