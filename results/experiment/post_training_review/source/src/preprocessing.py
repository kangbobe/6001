"""Importable, fold-fitted preprocessing shared by all four model families."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data import FEATURES

CATEGORICAL = ['job','marital','education','default','housing','loan','poutcome']
NUMERIC = ['age','balance','previous','has_previous_contact']
DERIVED_FEATURES = [c if c != 'pdays' else 'has_previous_contact' for c in FEATURES]


class HistoryFeatures(TransformerMixin, BaseEstimator):
    """Replace elapsed days by prior-contact status; no distribution learning."""
    def fit(self, X, y=None):
        self._validate(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != FEATURES:
            raise ValueError('Expected exactly the 11 frozen source features in order, excluding y and row_id')
        if X.isna().any().any():
            raise ValueError('Unexpected missing cell; literal unknown is the supported category')
        if not pd.api.types.is_numeric_dtype(X['pdays']) or (X['pdays'] < -1).any():
            raise ValueError('Invalid pdays')

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')
        self._validate(X)
        out = X.copy()
        out['has_previous_contact'] = X['pdays'].ge(0).astype(float)
        return out.drop(columns='pdays').loc[:, DERIVED_FEATURES]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        if input_features is not None and list(input_features) != FEATURES:
            raise ValueError('Unexpected input feature names')
        return np.asarray(DERIVED_FEATURES, dtype=object)


def build_preprocessor(scale_numeric=True):
    columns = ColumnTransformer([
        ('numeric', StandardScaler() if scale_numeric else 'passthrough', NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], remainder='drop', sparse_threshold=0)
    return Pipeline([('history', HistoryFeatures()), ('columns', columns)])
