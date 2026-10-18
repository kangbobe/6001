"""Recompute validation-only handoff checks and write an honest posthoc receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_partition
from src.metrics import binary_metrics
from src.models import MODEL_NAMES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path, base):
    content = path.read_bytes()
    return {'path': path.relative_to(base).as_posix(), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}


def verify(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve the prior receipt; choose a new --output path')
    protocol = read_json(experiment/'run_protocol.json')
    completion = read_json(experiment/'completion.json')
    assert completion['status'] == 'complete' and completion['test_evaluated'] is False
    X, y = load_partition('val')
    checks = []
    artifacts = ['run_protocol.json', 'cv_manifest.csv', 'completion.json', 'development_summary.csv']
    for name in ['dummy'] + MODEL_NAMES:
        folder = experiment/name
        meta = read_json(folder/'model_metadata.json')
        assert meta['refit_partition'] == 'train' and meta['fit_n'] == protocol['train_n']
        assert meta['test_evaluated'] is False
        candidates = pd.read_csv(folder/'cv_candidates.csv')
        selected = candidates.sort_values(['mean_average_precision','config_id'], ascending=[False,True]).iloc[0]
        assert int(selected.config_id) == meta['config_id']
        model = joblib.load(folder/'pipeline.joblib')
        assert list(model.classes_) == [0, 1]
        with threadpool_limits(limits=1):
            probability = model.predict_proba(X)[:, 1]
        saved = pd.read_csv(folder/'validation_predictions.csv')
        np.testing.assert_array_equal(saved.row_id, X.index)
        np.testing.assert_array_equal(saved.y_true, y)
        assert saved['model'].eq(name).all() and saved['split'].eq('val').all()
        np.testing.assert_allclose(saved.p_yes, probability, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(saved.prediction_at_0_5, (probability >= .5).astype(int))
        metrics = binary_metrics(y, probability)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, meta['validation_metrics_at_0_5'][key], atol=1e-12, rtol=1e-12)
        checks.append({'model':name, 'rows':len(X), 'selected_config_verified':int(selected.config_id),
                       'max_abs_probability_error':float(np.max(np.abs(saved.p_yes-probability))),
                       'validation_metrics_recomputed':True})
        artifacts += [f'{name}/{file}' for file in ['pipeline.joblib','model_metadata.json',
            'validation_predictions.csv','cv_candidates.csv','cv_folds.csv','fit_histories.json',
            'feature_names.json','search_configs.json']]
    source_paths = list((ROOT/'src').glob('*.py')) + [ROOT/p for p in [
        'scripts/train.py','scripts/verify_development.py','config/dataset_lock.json',
        'config/feature_policy.csv','requirements.lock.txt','data/raw/bank-full.csv',
        'data/splits/split_indices.json','data/splits/split_manifest.csv']]
    snapshot = experiment/'source_snapshot/manifest.json'
    receipt = {
        'verified_at_utc':datetime.now(timezone.utc).isoformat(), 'verification_kind':'posthoc',
        'training_time_source_snapshot_available':snapshot.exists() and
            read_json(snapshot)['phase'] == 'before_model_fitting',
        'limitation':'Current identities and validation reproduction do not prove historical training-time source bytes.',
        'test_predictions_or_metrics_computed':False, 'validation_checks':checks,
        'project_files':[identity(p, ROOT) for p in sorted(source_paths)],
        'experiment_files':[identity(experiment/p, experiment) for p in artifacts],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Validation-only verification passed for {len(checks)} models; receipt: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT/'results/experiment')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify(args.experiment, args.output or args.experiment/'posthoc_verification.json')
