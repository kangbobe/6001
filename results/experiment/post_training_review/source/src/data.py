"""Version-locked source and row partitions; no learned preprocessing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'config/dataset_lock.json').read_text(encoding='utf-8'))
FEATURES = LOCK['allowed_features']
EXPECTED_COLUMNS = LOCK['columns']
EXCLUDED = LOCK['excluded_features']
NUMERIC = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORIES = {
    'job': ['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services'],
    'marital': ['married','divorced','single'], 'education': ['unknown','secondary','primary','tertiary'],
    'default': ['yes','no'], 'housing': ['yes','no'], 'loan': ['yes','no'],
    'contact': ['unknown','telephone','cellular'],
    'month': ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'poutcome': ['unknown','other','failure','success'], 'y': ['yes','no'],
}


def read_raw(root=ROOT):
    path = Path(root) / 'data/raw/bank-full.csv'
    content = path.read_bytes()
    expected = LOCK['artifacts']['data/raw/bank-full.csv']
    if hashlib.sha256(content).hexdigest() != expected['sha256'] or len(content) != expected['bytes']:
        raise ValueError('Raw data identity mismatch; expected the locked legacy bank-full.csv')
    raw = pd.read_csv(path, sep=';', keep_default_na=False)
    if raw.shape != tuple(LOCK['shape']) or list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError('Wrong source schema or dataset variant')
    if raw.isna().any().any():
        raise ValueError('Unexpected missing values')
    for column in NUMERIC:
        if not pd.api.types.is_integer_dtype(raw[column]):
            raise ValueError(f'Expected integer column: {column}')
    for column, allowed in CATEGORIES.items():
        if not raw[column].isin(allowed).all():
            raise ValueError(f'Unexpected category: {column}')
    if (raw['pdays'] < -1).any() or (raw['previous'] < 0).any():
        raise ValueError('Invalid contact history values')
    raw.index.name = 'row_id'
    return raw


def load_split_indices(root=ROOT):
    root = Path(root)
    saved = json.loads((root / 'data/splits/split_indices.json').read_text(encoding='utf-8'))
    expected_sizes = {'train': 27126, 'val': 9042, 'test': 9043}
    if set(saved) != set(expected_sizes):
        raise ValueError('Unexpected partition names')
    for name, values in saved.items():
        if not isinstance(values, list) or len(values) != expected_sizes[name]:
            raise ValueError(f'Unexpected partition size: {name}')
        if any(type(v) is not int or not 0 <= v < 45211 for v in values):
            raise ValueError(f'Invalid row_id in {name}')
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in saved.items()}
    all_ids = np.concatenate(list(arrays.values()))
    if not np.array_equal(np.sort(all_ids), np.arange(45211)):
        raise ValueError('Partition membership overlaps or does not cover every row')
    manifest = pd.read_csv(root / 'data/splits/split_manifest.csv', keep_default_na=False)
    if list(manifest.columns) != ['row_id','split'] or len(manifest) != 45211 or not manifest.row_id.is_unique:
        raise ValueError('Invalid split manifest')
    expected = pd.Series('', index=pd.RangeIndex(45211, name='row_id'), name='split')
    for name, ids in arrays.items():
        expected.loc[ids] = name
    pd.testing.assert_frame_equal(manifest.sort_values('row_id').reset_index(drop=True), expected.reset_index())
    return arrays


def load_partition(name, *, allow_test=False, root=ROOT):
    if name not in {'train','val','test'}:
        raise ValueError('Expected train, val, or test')
    if name == 'test' and not allow_test:
        raise ValueError('Test is reserved for D after protocol freeze; explicit allow_test=True required')
    ids = load_split_indices(root)[name]
    frame = read_raw(root).loc[ids]
    X = frame[FEATURES].copy()
    y = frame['y'].map({'no': 0, 'yes': 1}).astype('int64')
    assert X.index.equals(y.index)
    return X, y


def audit_data(root=ROOT):
    raw, splits = read_raw(root), load_split_indices(root)
    train = raw.loc[splits['train']]
    # Only structural full-data checks; label relationships and distributions use train.
    full_input = raw.drop(columns='y')
    pair_counts = {}
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        left = pd.MultiIndex.from_frame(full_input.loc[splits[a]])
        right = pd.MultiIndex.from_frame(full_input.loc[splits[b]])
        pair_counts[f'{a}_{b}_exact_input_profiles'] = len(left.unique().intersection(right.unique()))
    return {
        'source': 'UCI Bank Marketing legacy bank-full.csv', 'raw_shape': list(raw.shape),
        'raw_sha256': LOCK['artifacts']['data/raw/bank-full.csv']['sha256'],
        'raw_missing_cells': int(raw.isna().sum().sum()),
        'full_data_exact_17_column_duplicate_rows': int(raw.duplicated().sum()),
        'partition_sizes': {k: len(v) for k,v in splits.items()},
        'cross_partition_unlabelled_profile_collisions': pair_counts,
        'train': {
            'rows': len(train), 'target_counts': {k:int(v) for k,v in train.y.value_counts().items()},
            'positive_rate': float(train.y.eq('yes').mean()),
            'exact_16_input_duplicates': int(train.drop(columns='y').duplicated().sum()),
            'exact_17_column_duplicates': int(train.duplicated().sum()),
            '11_source_feature_duplicates': int(train[FEATURES].duplicated().sum()),
            'negative_balance': int(train.balance.lt(0).sum()),
            'pdays_never_contacted': int(train.pdays.eq(-1).sum()),
            'pdays_zero_valid_contact': int(train.pdays.eq(0).sum()),
            'never_contacted_but_previous_positive': int((train.pdays.eq(-1)&train.previous.gt(0)).sum()),
            'contacted_but_previous_zero': int((train.pdays.ge(0)&train.previous.eq(0)).sum()),
            'never_contacted_but_outcome_known': int((train.pdays.eq(-1)&train.poutcome.ne('unknown')).sum()),
            'contacted_but_outcome_unknown': int((train.pdays.ge(0)&train.poutcome.eq('unknown')).sum()),
        },
        'policy': 'Keep all records; unknown remains a category; no clipping or deduplication',
        'limitation': 'No customer ID; matching profiles are not proof of duplicate customers. Random record splits do not prove future generalization.',
        'test_used_for_modeling_or_target_association': False,
    }
