import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

import src.training as training
from src.data import FEATURES


def _bank_frame():
    return pd.DataFrame({
        'age': [20, 25, 35, 40, 45, 55, 10_000, 60],
        'job': ['admin.', 'admin.', 'technician', 'services', 'admin.', 'services', 'unseen_job', 'admin.'],
        'marital': ['single', 'married', 'single', 'married', 'single', 'married', 'single', 'married'],
        'education': ['secondary', 'primary', 'tertiary', 'secondary', 'primary', 'tertiary', 'unseen_education', 'secondary'],
        'default': ['no'] * 8,
        'balance': [-50, 0, 50, 100, 150, 200, 10_000_000, 250],
        'housing': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
        'loan': ['no', 'no', 'yes', 'no', 'yes', 'no', 'unseen_loan', 'no'],
        'pdays': [-1, 0, 20, -1, 10, 50, 3, 7],
        'previous': [0, 1, 0, 2, 0, 1, 5, 0],
        'poutcome': ['unknown', 'success', 'failure', 'unknown', 'success', 'failure', 'unseen_outcome', 'unknown'],
    }, columns=FEATURES)


class _CaptureClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.fit_y_ = np.asarray(y).copy()
        self.sample_weight_ = np.asarray(sample_weight).copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


def test_score_fold_weights_use_only_fit_labels_even_when_heldout_labels_change(monkeypatch):
    X = pd.DataFrame({'x': np.arange(10)})
    y = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    fit_ids = np.array([0, 1, 2, 3, 4, 5])
    score_ids = np.array([6, 7, 8, 9])
    fitted_estimators = []

    def build_capture_model(*args, **kwargs):
        estimator = _CaptureClassifier()
        fitted_estimators.append(estimator)
        return Pipeline([('model', estimator)])

    monkeypatch.setattr(training, 'build_model', build_capture_model)
    config = {'weight': 'balanced'}
    training.score_fold('logistic_regression', config, 0, 0, fit_ids, score_ids, X, y)
    altered_y = y.copy()
    altered_y.iloc[score_ids] = [0, 1, 1, 1]
    training.score_fold('logistic_regression', config, 0, 0, fit_ids, score_ids, X, altered_y)

    expected_weights = compute_sample_weight('balanced', y.iloc[fit_ids])
    full_data_weights = compute_sample_weight('balanced', y)
    assert len(fitted_estimators) == 2
    for estimator in fitted_estimators:
        np.testing.assert_array_equal(estimator.fit_y_, y.iloc[fit_ids])
        np.testing.assert_allclose(estimator.sample_weight_, expected_weights)
    np.testing.assert_allclose(fitted_estimators[0].sample_weight_, fitted_estimators[1].sample_weight_)
    assert not np.allclose(expected_weights, full_data_weights[fit_ids])


def test_logistic_pipeline_fits_scaler_and_encoder_on_training_fold_and_predicts_unknowns():
    X = _bank_frame()
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    fitted, _ = training.fit_candidate(
        'logistic_regression', {'C': 1.0, 'weight': None}, X.iloc[:6], y_train)

    preprocess = fitted.named_steps['preprocess']
    history = preprocess.named_steps['history']
    columns = preprocess.named_steps['columns']
    transformed_train = history.transform(X.iloc[:6])
    numeric_columns = ['age', 'balance', 'previous', 'has_previous_contact']
    scaler = columns.named_transformers_['numeric']
    np.testing.assert_allclose(scaler.mean_, transformed_train[numeric_columns].mean().to_numpy())

    encoder = columns.named_transformers_['categorical']
    assert 'unseen_job' not in encoder.categories_[0]
    assert 'unseen_education' not in encoder.categories_[2]
    probabilities = fitted.predict_proba(X.iloc[[6]])
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_mlp_at_iteration_cap_without_convergence_warning_does_not_retry(monkeypatch):
    instances = []

    def fake_fit(self, X, y, **kwargs):
        instances.append(self)
        self.n_iter_ = self.max_iter
        self.loss_curve_ = []
        return self

    monkeypatch.setattr(MLPClassifier, 'fit', fake_fit)
    fitted, attempts = training.fit_candidate(
        'mlp', {'hidden_layer_sizes': (2,), 'alpha': 0.01, 'weight': None},
        _bank_frame().iloc[:6], pd.Series([0, 0, 0, 1, 1, 1]))

    assert len(instances) == 1
    assert instances[0] is fitted.named_steps['model']
    assert instances[0].max_iter == 300
    assert instances[0].warm_start is False
    assert len(attempts) == 1
    assert attempts[0]['n_iter'] == 300
    assert attempts[0]['convergence_warning'] is False
    assert attempts[0]['warnings'] == []


def test_mlp_cap_warning_retries_once_from_scratch_at_600_and_records_final_warning(monkeypatch):
    instances = []

    def fake_fit(self, X, y, **kwargs):
        instances.append(self)
        self.n_iter_ = self.max_iter
        self.loss_curve_ = []
        warnings.warn(f'synthetic convergence warning at {self.max_iter}', ConvergenceWarning)
        return self

    monkeypatch.setattr(MLPClassifier, 'fit', fake_fit)
    fitted, attempts = training.fit_candidate(
        'mlp', {'hidden_layer_sizes': (2,), 'alpha': 0.01, 'weight': None},
        _bank_frame().iloc[:6], pd.Series([0, 0, 0, 1, 1, 1]))

    assert [estimator.max_iter for estimator in instances] == [300, 600]
    assert instances[0] is not instances[1]
    assert all(estimator.warm_start is False for estimator in instances)
    assert fitted.named_steps['model'] is instances[1]
    assert [attempt['max_iter'] for attempt in attempts] == [300, 600]
    assert [attempt['n_iter'] for attempt in attempts] == [300, 600]
    assert [attempt['convergence_warning'] for attempt in attempts] == [True, True]
    assert len(attempts) == 2
    assert 'synthetic convergence warning at 600' in attempts[-1]['warnings'][0]


class _SelectionPreprocessor:
    def get_feature_names_out(self):
        return np.array(['synthetic_feature'])


class _SelectionPipeline:
    def __init__(self, config_id):
        self.config_id = config_id
        self.classes_ = np.array([0, 1])
        self.named_steps = {'preprocess': _SelectionPreprocessor()}

    def predict_proba(self, X):
        probabilities = {
            0: [0.1, 0.2, 0.8, 0.9],
            1: [0.9, 0.8, 0.2, 0.1],
            2: [0.1, 0.2, 0.8, 0.9],
        }[self.config_id][:len(X)]
        positive = np.asarray(probabilities)
        return np.column_stack([1 - positive, positive])


def test_train_family_selects_highest_mean_cv_ap_then_lowest_config_id(tmp_path, monkeypatch):
    configs = [{'id': config_id, 'weight': None} for config_id in range(3)]
    mean_cv_ap = {0: 0.8, 1: 0.9, 2: 0.9}
    mean_train_ap = {0: 0.99, 1: 0.01, 2: 0.99}
    refit_configs = []

    monkeypatch.setattr(training, 'search_configs', lambda name: configs)

    def fake_score_fold(name, config, config_id, fold_id, fit_ids, score_ids, X, y):
        return ({
            'model': name, 'config_id': config_id, 'fold_id': fold_id,
            'params': '{}', 'fit_n': len(fit_ids), 'fit_seconds': 0.01,
            'attempt_count': 1, 'n_iter': None, 'final_convergence_warning': False,
            'warnings': '[]', 'train_ap': mean_train_ap[config_id],
            'average_precision': mean_cv_ap[config_id], 'roc_auc': 0.5,
            'f1': 0.5, 'precision': 0.5, 'recall': 0.5, 'accuracy': 0.5,
            'balanced_accuracy': 0.5,
        }, {'config_id': config_id, 'fold_id': fold_id, 'attempts': []})

    def fake_fit_candidate(name, config, X, y):
        refit_configs.append(config.copy())
        return _SelectionPipeline(config['id']), [{
            'fit_seconds': 0.01, 'n_iter': None, 'convergence_warning': False,
            'warnings': [],
        }]

    monkeypatch.setattr(training, 'score_fold', fake_score_fold)
    monkeypatch.setattr(training, 'fit_candidate', fake_fit_candidate)
    X = pd.DataFrame({'x': [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    folds = [(np.array([0, 1]), np.array([2, 3])),
             (np.array([2, 3]), np.array([0, 1]))]

    metadata = training.train_family(
        'logistic_regression', X, y, X, y, folds, tmp_path, jobs=1)

    assert metadata['config_id'] == 1
    assert refit_configs == [configs[1]]
    assert metadata['cv_average_precision'] == 0.9
    assert metadata['selected_params'] == configs[1]
    assert metadata['validation_metrics_at_0_5']['average_precision'] < average_precision_score(
        y, _SelectionPipeline(2).predict_proba(X)[:, 1])
    candidates = pd.read_csv(tmp_path / 'logistic_regression' / 'cv_candidates.csv')
    np.testing.assert_allclose(candidates.loc[candidates['config_id'] == 1, 'mean_average_precision'], [0.9])
    np.testing.assert_allclose(candidates.loc[candidates['config_id'] == 2, 'mean_average_precision'], [0.9])
    np.testing.assert_allclose(candidates.loc[candidates['config_id'] == 1, 'mean_train_ap'], [0.01])
