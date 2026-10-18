"""Train-only exploratory analysis and report-ready figures for CA6001."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_POLICY = 'pre_campaign_history_flag_v1'
NUMERIC_PROFILE_COLUMNS = ['age', 'balance', 'previous', 'pdays']
CATEGORY_RATE_COLUMNS = ['poutcome', 'job']


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _duplicate_summary(frame: pd.DataFrame) -> dict:
    duplicate_mask = frame.duplicated(keep=False)
    return {
        'columns': list(frame.columns),
        'n_rows': int(len(frame)),
        'n_unique_profiles': int(frame.drop_duplicates().shape[0]),
        'duplicate_excess_rows': int(frame.duplicated().sum()),
        'rows_in_duplicate_groups': int(duplicate_mask.sum()),
        'duplicate_group_count': int(frame.loc[duplicate_mask].drop_duplicates().shape[0]),
        'largest_duplicate_group': int(frame.value_counts(dropna=False).max()) if len(frame) else 0,
    }


def _target_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().reindex([0, 1], fill_value=0).sort_index()
    result = pd.DataFrame({
        'target': ['no', 'yes'],
        'target_value': [0, 1],
        'n': counts.to_numpy(dtype=int),
    })
    result['share'] = result['n'] / len(y)
    return result


def _numeric_profile(raw_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_PROFILE_COLUMNS:
        values = pd.to_numeric(raw_train[column], errors='raise')
        rows.append({
            'feature': column,
            'n': int(values.notna().sum()),
            'missing_n': int(values.isna().sum()),
            'mean': float(values.mean()),
            'std': float(values.std(ddof=1)),
            'min': float(values.min()),
            'p01': float(values.quantile(.01)),
            'p05': float(values.quantile(.05)),
            'p25': float(values.quantile(.25)),
            'median': float(values.median()),
            'p75': float(values.quantile(.75)),
            'p95': float(values.quantile(.95)),
            'p99': float(values.quantile(.99)),
            'max': float(values.max()),
            'negative_n': int(values.lt(0).sum()),
            'zero_n': int(values.eq(0).sum()),
        })
    return pd.DataFrame(rows)


def _unknown_frequency(raw_train: pd.DataFrame) -> pd.DataFrame:
    categorical = [column for column in raw_train.columns
                   if column != 'y' and not pd.api.types.is_numeric_dtype(raw_train[column])]
    rows = []
    for column in categorical:
        unknown_n = int(raw_train[column].eq('unknown').sum())
        rows.append({
            'feature': column,
            'unknown_n': unknown_n,
            'n': int(len(raw_train)),
            'unknown_share': unknown_n / len(raw_train),
        })
    return pd.DataFrame(rows).sort_values(['unknown_share', 'feature'], ascending=[False, True])


def _category_target_rates(raw_train: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for column in CATEGORY_RATE_COLUMNS:
        summary = pd.DataFrame({'category': raw_train[column], 'y': y.to_numpy()})
        grouped = summary.groupby('category', sort=True, observed=True)['y'].agg(['size', 'sum', 'mean'])
        for category, stats in grouped.iterrows():
            rows.append({
                'feature': column,
                'category': str(category),
                'n': int(stats['size']),
                'subscribed_n': int(stats['sum']),
                'subscription_rate': float(stats['mean']),
            })
    return pd.DataFrame(rows)


def _contact_history_tables(raw_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe cross-field counts without treating the relationships as hard errors."""
    pdays = pd.to_numeric(raw_train['pdays'], errors='raise')
    previous = pd.to_numeric(raw_train['previous'], errors='raise')
    outcome = raw_train['poutcome'].astype(str)
    contact_status = pd.Series(np.where(pdays.ge(0), 'pdays_nonnegative', 'pdays_negative'),
                               index=raw_train.index)
    previous_status = pd.Series(np.where(previous.gt(0), 'previous_positive', 'previous_zero'),
                                 index=raw_train.index)
    outcome_status = pd.Series(np.where(outcome.eq('unknown'), 'outcome_unknown', 'outcome_known'),
                               index=raw_train.index)

    pairs = [
        ('pdays_vs_previous', contact_status, previous_status),
        ('pdays_vs_poutcome', contact_status, outcome_status),
        ('previous_vs_poutcome', previous_status, outcome_status),
    ]
    rows = []
    for name, left, right in pairs:
        counts = pd.crosstab(left, right, dropna=False)
        for left_value in counts.index:
            for right_value in counts.columns:
                n = int(counts.loc[left_value, right_value])
                rows.append({
                    'comparison': name,
                    'field_a_status': str(left_value),
                    'field_b_status': str(right_value),
                    'n': n,
                    'share_of_train': n / len(raw_train),
                })
    summary = [
        ('pdays_negative', int(pdays.lt(0).sum()), 'pdays is the source used to derive the binary history flag'),
        ('pdays_nonnegative', int(pdays.ge(0).sum()), 'pdays is the source used to derive the binary history flag'),
        ('pdays_negative_and_previous_positive', int((pdays.lt(0) & previous.gt(0)).sum()), 'descriptive cross-field combination'),
        ('pdays_nonnegative_and_previous_zero', int((pdays.ge(0) & previous.eq(0)).sum()), 'descriptive cross-field combination'),
        ('pdays_negative_and_outcome_known', int((pdays.lt(0) & outcome.ne('unknown')).sum()), 'descriptive cross-field combination'),
        ('pdays_nonnegative_and_outcome_unknown', int((pdays.ge(0) & outcome.eq('unknown')).sum()), 'descriptive cross-field combination'),
        ('previous_zero_and_outcome_known', int((previous.eq(0) & outcome.ne('unknown')).sum()), 'descriptive cross-field combination'),
        ('previous_positive_and_outcome_unknown', int((previous.gt(0) & outcome.eq('unknown')).sum()), 'descriptive cross-field combination'),
    ]
    summary_frame = pd.DataFrame(summary, columns=['diagnostic', 'n', 'interpretation'])
    summary_frame['share_of_train'] = summary_frame['n'] / len(raw_train)
    return pd.DataFrame(rows), summary_frame


def _plot_target(distribution: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(distribution['target'], distribution['n'], color=['#6687a8', '#d57b54'])
    ax.set_title('Training target distribution')
    ax.set_xlabel('Subscription outcome')
    ax.set_ylabel('Records')
    ax.set_ylim(0, distribution['n'].max() * 1.25)
    ax.grid(axis='y', alpha=.25)
    for bar, n, share in zip(bars, distribution['n'], distribution['share']):
        ax.annotate(f'n={n:,}\n{share:.1%}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(output_dir / 'fig1_target_distribution.png', dpi=160)
    plt.close(fig)


def _plot_numeric(raw_train: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(raw_train['age'], bins=30, color='#4c8c83', edgecolor='white')
    axes[0].set_title('Age distribution (training partition)')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Records')
    axes[0].grid(axis='y', alpha=.2)

    balance = pd.to_numeric(raw_train['balance'], errors='raise')
    axes[1].ecdf(balance, color='#ba6a4b', linewidth=2)
    axes[1].set_xscale('symlog', linthresh=100)
    axes[1].set_title('Balance distribution (training partition)')
    axes[1].set_xlabel('Annual balance (EUR; symlog scale; negative values retained)')
    axes[1].set_ylabel('Cumulative share of records')
    axes[1].grid(alpha=.25, which='both')
    fig.tight_layout()
    fig.savefig(output_dir / 'fig2_age_balance_distribution.png', dpi=160)
    plt.close(fig)


def _plot_unknown(unknown: pd.DataFrame, output_dir: Path):
    plot_data = unknown.sort_values('unknown_share', ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(plot_data['feature'], plot_data['unknown_share'] * 100, color='#7a8f6a')
    ax.set_title('Literal "unknown" frequency by categorical field')
    ax.set_xlabel('Training records with value "unknown" (%)')
    ax.set_ylabel('Categorical field')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=.25)
    for bar, n in zip(bars, plot_data['unknown_n']):
        ax.annotate(f'n={n:,}', (bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(4, 0), textcoords='offset points', va='center')
    fig.tight_layout()
    fig.savefig(output_dir / 'fig3_unknown_frequency.png', dpi=160)
    plt.close(fig)


def _plot_category_rates(rates: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, field in zip(axes, CATEGORY_RATE_COLUMNS):
        data = rates.loc[rates['feature'].eq(field)].sort_values('subscription_rate')
        bars = ax.bar(data['category'], data['subscription_rate'] * 100,
                      color='#c57b54' if field == 'poutcome' else '#56877e')
        ax.set_title(f'Subscription rate by {field}')
        ax.set_xlabel(field.capitalize())
        ax.set_ylabel('Training records subscribed (%)')
        ax.set_ylim(0, data['subscription_rate'].max() * 100 * 1.35)
        ax.tick_params(axis='x', rotation=40)
        ax.grid(axis='y', alpha=.25)
        for bar, n in zip(bars, data['n']):
            ax.annotate(f'n={n:,}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=8, rotation=90 if len(data) > 8 else 0)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig4_category_subscription_rates.png', dpi=160)
    plt.close(fig)


def _plot_contact_history(raw_train: pd.DataFrame, output_dir: Path):
    pdays = pd.to_numeric(raw_train['pdays'], errors='raise')
    previous = pd.to_numeric(raw_train['previous'], errors='raise')
    outcome = raw_train['poutcome'].astype(str)
    frame = pd.DataFrame({
        'pdays status': np.where(pdays.ge(0), 'pdays >= 0', 'pdays < 0'),
        'previous status': np.where(previous.gt(0), 'previous > 0', 'previous = 0'),
        'poutcome status': np.where(outcome.eq('unknown'), 'unknown', 'known'),
    }, index=raw_train.index)
    comparisons = [
        ('pdays vs previous', 'pdays status', 'previous status'),
        ('pdays vs poutcome', 'pdays status', 'poutcome status'),
        ('previous vs poutcome', 'previous status', 'poutcome status'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (title, left, right) in zip(axes, comparisons):
        table = pd.crosstab(frame[left], frame[right], dropna=False)
        table.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', width=.68)
        ax.set_title(title)
        ax.set_xlabel(left)
        ax.set_ylabel('Training records')
        ax.set_ylim(0, table.sum(axis=1).max() * 1.15)
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title=right, fontsize=8, title_fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Prior-contact fields: descriptive cross-field counts')
    fig.tight_layout()
    fig.savefig(output_dir / 'fig5_contact_history_consistency.png', dpi=160)
    plt.close(fig)


def write_eda_outputs(raw_train: pd.DataFrame, train_X: pd.DataFrame, train_y: pd.Series,
                      *, audit_summary: dict, output_dir) -> Path:
    """Create report-ready outputs from one already-selected training partition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw_train.index.equals(train_X.index) or not raw_train.index.equals(train_y.index):
        raise ValueError('Raw rows, training features, and labels must align by row_id')
    if train_y.ndim != 1 or set(pd.unique(train_y)) != {0, 1}:
        raise ValueError('Training labels must be a binary Series containing both classes')
    if list(train_X.columns) != ['age', 'job', 'marital', 'education', 'default', 'balance',
                                 'housing', 'loan', 'pdays', 'previous', 'poutcome']:
        raise ValueError('Expected the frozen 11 raw source feature columns in order')
    required_raw = {'y', *train_X.columns}
    if not required_raw.issubset(raw_train.columns):
        raise ValueError(f'Raw training frame lacks expected columns: {sorted(required_raw-set(raw_train.columns))}')

    y = train_y.astype('int64')
    raw_y = raw_train['y'].map({'no': 0, 'yes': 1})
    if raw_y.isna().any() or not raw_y.astype('int64').equals(y):
        raise ValueError('Raw training labels do not align with load_partition("train") labels')

    # The transformed profile has the same 11-column width: pdays is replaced by
    # the fixed binary flag and never kept as an exact elapsed-day input.
    policy_profile = train_X.copy()
    policy_profile['has_previous_contact'] = policy_profile['pdays'].ge(0).astype('int64')
    policy_columns = ['has_previous_contact' if column == 'pdays' else column
                      for column in train_X.columns]
    policy_profile = policy_profile.drop(columns='pdays').loc[:, policy_columns]
    duplicate_profiles = {
        'raw_16_inputs': _duplicate_summary(raw_train.drop(columns=['y', 'row_id'], errors='ignore')),
        'raw_17_columns_including_target': _duplicate_summary(raw_train.drop(columns=['row_id'], errors='ignore')),
        'raw_11_source_features_including_pdays': _duplicate_summary(train_X),
        'policy_11_profile_pdays_replaced_by_flag': _duplicate_summary(policy_profile),
        'interpretation': 'Repeated profiles are row-level matches, not evidence of repeated customers; no rows are removed.',
        'feature_policy': FEATURE_POLICY,
    }

    distribution = _target_distribution(y)
    numeric = _numeric_profile(raw_train)
    unknown = _unknown_frequency(raw_train)
    rates = _category_target_rates(raw_train, y)
    consistency, consistency_summary = _contact_history_tables(raw_train)

    distribution.to_csv(output_dir / 'target_distribution.csv', index=False)
    numeric.to_csv(output_dir / 'numeric_profile.csv', index=False)
    unknown.to_csv(output_dir / 'unknown_frequency.csv', index=False)
    rates.to_csv(output_dir / 'category_subscription_rates.csv', index=False)
    consistency.to_csv(output_dir / 'contact_history_crosstabs.csv', index=False)
    consistency_summary.to_csv(output_dir / 'contact_history_diagnostics.csv', index=False)
    _write_json(output_dir / 'duplicate_profile_summary.json', duplicate_profiles)
    _write_json(output_dir / 'source_audit.json', audit_summary)
    _write_json(output_dir / 'eda_manifest.json', {
        'partition': 'train',
        'n': int(len(y)),
        'row_id_source': 'raw.index / load_partition("train").index',
        'source_features': list(train_X.columns),
        'policy_features': list(policy_profile.columns),
        'feature_policy': FEATURE_POLICY,
        'target_encoding': {'no': 0, 'yes': 1},
        'unknown_values_retained': True,
        'negative_and_extreme_values_retained': True,
        'test_target_relationships_computed': False,
        'figures': [
            'fig1_target_distribution.png',
            'fig2_age_balance_distribution.png',
            'fig3_unknown_frequency.png',
            'fig4_category_subscription_rates.png',
            'fig5_contact_history_consistency.png',
        ],
    })

    _plot_target(distribution, output_dir)
    _plot_numeric(raw_train, output_dir)
    _plot_unknown(unknown, output_dir)
    _plot_category_rates(rates, output_dir)
    _plot_contact_history(raw_train, output_dir)
    return output_dir


def run_eda(root=None, output_dir=None) -> Path:
    """Load only the train partition for EDA; audit_data is structure-only."""
    from src.data import ROOT, audit_data, load_partition, load_split_indices, read_raw

    root = ROOT if root is None else Path(root)
    raw = read_raw(root=root)
    split_indices = load_split_indices(root=root)
    train_ids = split_indices['train']
    raw_train = raw.loc[train_ids]
    train_X, train_y = load_partition('train', root=root)
    if not raw_train.index.equals(train_X.index):
        raise ValueError('The split manifest and train partition row_id order disagree')
    summary = audit_data(root=root)
    output_dir = Path(output_dir) if output_dir is not None else Path(root) / 'results/eda'
    return write_eda_outputs(raw_train, train_X, train_y, audit_summary=summary,
                             output_dir=output_dir)
