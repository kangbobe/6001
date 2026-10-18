import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data import FEATURES, load_partition, load_split_indices
from src.preprocessing import HistoryFeatures, build_preprocessor


def fixture_X():
    return pd.DataFrame([
        [20,'admin.','single','unknown','no',-20,'no','no',-1,0,'unknown'],
        [40,'student','married','primary','no',100,'yes','no',0,1,'success'],
        [60,'unknown','divorced','secondary','no',200,'no','yes',90,2,'failure'],
    ], columns=FEATURES)


def test_exact_split_coverage_and_partition_alignment():
    splits = load_split_indices()
    assert sum(len(ids) for ids in splits.values()) == 45211
    assert not set(splits['train']).intersection(splits['test'])
    X, y = load_partition('train')
    np.testing.assert_array_equal(X.index, splits['train'])
    assert X.index.equals(y.index)
    assert list(X.columns) == FEATURES and len(X) == 27126


def test_test_guard_precedes_any_read(tmp_path):
    with pytest.raises(ValueError, match='reserved'):
        load_partition('test', root=tmp_path)


def test_contact_sentinel_and_time_information_removed():
    X = fixture_X()
    transformer = HistoryFeatures().fit(X)
    out = transformer.transform(X)
    assert out.has_previous_contact.tolist() == [0,1,1]
    changed = X.copy(); changed.loc[2,'pdays'] = 900
    pd.testing.assert_frame_equal(out, transformer.transform(changed))
    assert 'pdays' not in out and out.loc[0,'balance'] == -20


def test_scaler_and_encoder_fit_only_training_values():
    X = fixture_X()
    pipe = clone(build_preprocessor()).fit(X)
    mean = pipe.named_steps['columns'].named_transformers_['numeric'].mean_.copy()
    altered = X.copy(); altered.loc[0,'balance'] = 1e9; altered.loc[0,'job'] = 'new_job'
    transformed = pipe.transform(altered)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(mean, pipe.named_steps['columns'].named_transformers_['numeric'].mean_)
    assert any('job_unknown' in name for name in pipe.get_feature_names_out())
    assert all('pdays' not in name and 'duration' not in name for name in pipe.get_feature_names_out())


def test_forbidden_columns_not_silently_accepted():
    X = fixture_X().assign(duration=100)
    with pytest.raises(ValueError, match='11 frozen'):
        build_preprocessor().fit(X)
