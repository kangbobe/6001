"""Frozen, small CPU search spaces. Configuration order breaks exact AP ties."""
from itertools import product

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

MODEL_NAMES = ['logistic_regression', 'decision_tree', 'random_forest', 'mlp']


def search_configs(name):
    if name == 'logistic_regression':
        return [dict(C=c, weight=w) for c, w in product([.01, .1, 1., 10.], [None, 'balanced'])]
    if name == 'decision_tree':
        return [dict(max_depth=d, min_samples_leaf=l, weight=w)
                for d, l, w in product([4, 8, None], [100, 20], [None, 'balanced'])]
    if name == 'random_forest':
        return [dict(max_depth=d, min_samples_leaf=l, weight=w)
                for d, l, w in product([10, None], [20, 5], [None, 'balanced'])]
    if name == 'mlp':
        return [dict(hidden_layer_sizes=h, alpha=a, weight=w)
                for h, a, w in product([(32,), (64, 32)], [.01, .0001], [None, 'balanced'])]
    if name == 'dummy':
        return [dict(weight=None)]
    raise ValueError(f'Unknown model: {name}')


def build_model(name, config, *, iteration_cap=None):
    params = {k: v for k, v in config.items() if k != 'weight'}
    if name == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42, **params)
    elif name == 'decision_tree':
        model = DecisionTreeClassifier(criterion='gini', random_state=42, **params)
    elif name == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, max_features='sqrt', bootstrap=True,
                                       random_state=42, n_jobs=1, **params)
    elif name == 'mlp':
        model = MLPClassifier(activation='relu', solver='adam', learning_rate_init=.001,
                              batch_size=256, early_stopping=False, max_iter=iteration_cap or 300,
                              tol=1e-4, n_iter_no_change=20, random_state=42, **params)
    elif name == 'dummy':
        model = DummyClassifier(strategy='prior')
    else:
        raise ValueError(name)
    return Pipeline([('preprocess', build_preprocessor(scale_numeric=name in {'logistic_regression', 'mlp'})),
                     ('model', model)])
