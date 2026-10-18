import json

import pandas as pd

from src import data
from src.eda import run_eda


FEATURES = ['age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'pdays', 'previous', 'poutcome']


def _synthetic_raw():
    index = pd.Index(range(100, 112), name='row_id')
    raw = pd.DataFrame({
        'age': [21, 21, 34, 46, 53, 61, 25, 29, 42, 50, 38, 66],
        'job': ['admin.', 'admin.', 'unknown', 'retired', 'technician', 'student',
                'services', 'management', 'blue-collar', 'unknown', 'retired', 'student'],
        'marital': ['single', 'single', 'married', 'married', 'divorced', 'single',
                    'married', 'single', 'married', 'single', 'divorced', 'married'],
        'education': ['secondary', 'secondary', 'unknown', 'primary', 'tertiary', 'secondary',
                      'primary', 'tertiary', 'secondary', 'unknown', 'primary', 'tertiary'],
        'default': ['no'] * 12,
        'balance': [-300, -300, 0, 200, 1000, 900000, 400, 20, 100, -5, 700, 80],
        'housing': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'loan': ['no'] * 12,
        'pdays': [5, 9, 5, 9, -1, 0, 4, -1, 20, -1, 5, -1],
        'previous': [1, 1, 1, 1, 0, 2, 1, 0, 3, 0, 1, 0],
        'poutcome': ['unknown', 'unknown', 'success', 'failure', 'unknown', 'other',
                     'success', 'unknown', 'failure', 'unknown', 'other', 'unknown'],
        'y': ['no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no'],
    }, index=index)
    return raw


def test_run_eda_reads_only_train_and_emits_train_scoped_outputs(tmp_path, monkeypatch):
    raw = _synthetic_raw()
    train_ids = raw.index.to_numpy()
    X = raw[FEATURES].copy()
    y = raw['y'].map({'no': 0, 'yes': 1}).astype('int64')
    calls = []

    monkeypatch.setattr(data, 'read_raw', lambda root: raw.copy())
    monkeypatch.setattr(data, 'load_split_indices', lambda root: {
        'train': train_ids, 'val': train_ids[:0], 'test': train_ids[:0],
    })

    def fake_load_partition(name, *, root):
        calls.append(name)
        assert name == 'train'
        return X.copy(), y.copy()

    audit = {'raw_shape': [12, 12], 'test_used_for_modeling_or_target_association': False}
    monkeypatch.setattr(data, 'load_partition', fake_load_partition)
    monkeypatch.setattr(data, 'audit_data', lambda root: audit)
    out = run_eda(root=tmp_path, output_dir=tmp_path / 'eda')

    assert calls == ['train']
    manifest = json.loads((out / 'eda_manifest.json').read_text(encoding='utf-8'))
    assert manifest['partition'] == 'train'
    assert manifest['n'] == len(train_ids)
    assert manifest['test_target_relationships_computed'] is False
    assert len(list(out.glob('fig*.png'))) == 5
    assert pd.read_csv(out / 'target_distribution.csv')['n'].sum() == len(train_ids)

    unknown = pd.read_csv(out / 'unknown_frequency.csv').set_index('feature')
    assert unknown.loc['job', 'unknown_n'] == 2
    assert (out / 'numeric_profile.csv').exists()
    assert (out / 'contact_history_crosstabs.csv').exists()

    duplicates = json.loads((out / 'duplicate_profile_summary.json').read_text(encoding='utf-8'))
    assert duplicates['raw_16_inputs']['duplicate_excess_rows'] == 0
    assert duplicates['policy_11_profile_pdays_replaced_by_flag']['duplicate_excess_rows'] >= 1

    rates = pd.read_csv(out / 'category_subscription_rates.csv')
    unknown_poutcome = rates.loc[(rates.feature == 'poutcome') & (rates.category == 'unknown')].iloc[0]
    assert unknown_poutcome['n'] == int(raw.poutcome.eq('unknown').sum())

    numeric = pd.read_csv(out / 'numeric_profile.csv').set_index('feature')
    assert numeric.loc['balance', 'min'] == -300
    assert numeric.loc['balance', 'max'] == 900000
