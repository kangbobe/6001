from datetime import datetime, timezone
import hashlib
import json

import pytest

from src.provenance import capture_provenance


_INPUT_PATHS = (
    'data/raw/bank-full.csv',
    'data/splits/split_indices.json',
    'data/splits/split_manifest.csv',
)


def _write(root, relative, contents):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _project(root):
    files = {
        'src/__init__.py': b'# package\n',
        'src/training.py': b'def train(): return 1\n',
        'src/nested/ignored.py': b'# not part of src/*.py\n',
        'scripts/train.py': b'from src.training import train\n',
        'config/dataset_lock.json': b'{"dataset":"fixture"}\n',
        'config/feature_policy.csv': b'feature,policy\nage,keep\n',
        'requirements.lock.txt': b'numpy==0.0\n',
        _INPUT_PATHS[0]: b'synthetic raw input\n',
        _INPUT_PATHS[1]: b'{"train":[0],"val":[1]}\n',
        _INPUT_PATHS[2]: b'row_id,split\n0,train\n',
    }
    for relative, contents in files.items():
        _write(root, relative, contents)
    return files


def test_before_capture_snapshots_fixed_sources_and_records_input_identities(tmp_path):
    root = tmp_path / 'project'
    files = _project(root)
    manifest_path = capture_provenance(
        tmp_path / 'capture', phase='before_model_fitting', root=root)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    captured_at = datetime.fromisoformat(manifest['captured_at_utc'])
    assert captured_at.tzinfo is not None
    assert captured_at.utcoffset() == timezone.utc.utcoffset(captured_at)
    assert manifest['phase'] == 'before_model_fitting'
    assert manifest['training_time_provenance'] is True
    assert any('caller-declared' in item for item in manifest['limitations'])

    expected_source_paths = sorted(
        set(files) - set(_INPUT_PATHS) - {'src/nested/ignored.py'})
    records = {record['path']: record for record in manifest['source_files']}
    assert sorted(records) == expected_source_paths
    for relative in expected_source_paths:
        contents = files[relative]
        record = records[relative]
        assert record['bytes'] == len(contents)
        assert record['sha256'] == hashlib.sha256(contents).hexdigest()
        assert (manifest_path.parent / 'source' / relative).read_bytes() == contents

    input_records = {record['path']: record for record in manifest['inputs']}
    assert set(input_records) == set(_INPUT_PATHS)
    for relative in _INPUT_PATHS:
        contents = files[relative]
        assert input_records[relative]['bytes'] == len(contents)
        assert input_records[relative]['sha256'] == hashlib.sha256(contents).hexdigest()
        assert not (manifest_path.parent / 'source' / relative).exists()


def test_post_training_capture_is_marked_as_not_training_time_provenance(tmp_path):
    root = tmp_path / 'project'
    _project(root)
    manifest_path = capture_provenance(
        tmp_path / 'review', phase='post_training_review', root=root)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    assert manifest['phase'] == 'post_training_review'
    assert manifest['training_time_provenance'] is False
    assert any('cannot prove which source bytes were used during training' in item
               for item in manifest['limitations'])


def test_input_content_changes_are_reflected_by_hash_without_copying_input(tmp_path):
    root = tmp_path / 'project'
    _project(root)
    first_path = capture_provenance(
        tmp_path / 'first', phase='before_model_fitting', root=root)
    input_path = root / _INPUT_PATHS[0]
    input_path.write_bytes(b'changed synthetic raw input\n')
    second_path = capture_provenance(
        tmp_path / 'second', phase='before_model_fitting', root=root)

    first = json.loads(first_path.read_text(encoding='utf-8'))
    second = json.loads(second_path.read_text(encoding='utf-8'))
    first_inputs = {record['path']: record for record in first['inputs']}
    second_inputs = {record['path']: record for record in second['inputs']}
    assert first_inputs[_INPUT_PATHS[0]]['sha256'] != second_inputs[_INPUT_PATHS[0]]['sha256']
    assert not (first_path.parent / 'source' / _INPUT_PATHS[0]).exists()
    assert not (second_path.parent / 'source' / _INPUT_PATHS[0]).exists()


def test_capture_rejects_existing_output_and_preserves_its_contents(tmp_path):
    root = tmp_path / 'project'
    _project(root)
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')

    with pytest.raises(FileExistsError):
        capture_provenance(output, phase='before_model_fitting', root=root)

    assert sentinel.read_text(encoding='utf-8') == 'keep'
    assert sorted(path.name for path in output.iterdir()) == ['keep.txt']


def test_capture_rejects_unknown_phase_before_creating_output(tmp_path):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='phase must be'):
        capture_provenance(output, phase='during_training', root=tmp_path)
    assert not output.exists()


def test_run_experiment_captures_provenance_before_any_family_fit(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    import src.training as training

    root = tmp_path / 'project'
    source_files = _project(root)
    output = tmp_path / 'experiment'
    monkeypatch.setattr(training, 'ROOT', root)
    partitions = []
    families = []
    train_X = pd.DataFrame({'synthetic_feature': np.arange(12)})
    train_y = pd.Series([0, 1] * 6)
    val_X = pd.DataFrame({'synthetic_feature': np.arange(4)}, index=[100, 101, 102, 103])
    val_y = pd.Series([0, 1, 0, 1], index=val_X.index)

    def fake_load_partition(split):
        partitions.append(split)
        if split == 'train':
            return train_X.copy(), train_y.copy()
        if split == 'val':
            return val_X.copy(), val_y.copy()
        raise AssertionError(f'run_experiment must not load {split!r}')

    def fake_train_family(name, X, y, X_val, y_val, folds, experiment_dir, jobs=2):
        families.append(name)
        snapshot_dir = experiment_dir / 'source_snapshot'
        manifest_path = snapshot_dir / 'manifest.json'
        protocol_path = experiment_dir / 'run_protocol.json'
        assert (experiment_dir / 'cv_manifest.csv').is_file()
        assert manifest_path.is_file()
        assert protocol_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        assert manifest['phase'] == 'before_model_fitting'
        assert manifest['training_time_provenance'] is True
        assert protocol['source_provenance'] == 'source_snapshot/manifest.json'
        assert protocol['test_evaluated'] is False
        assert (snapshot_dir / 'source' / 'src' / 'training.py').read_bytes() == source_files['src/training.py']
        assert all(split != 'test' for split in partitions)

        return {
            'config_id': 0,
            'cv_average_precision': 0.5,
            'cv_ap_std': 0.0,
            'refit_attempts': [{'n_iter': 1, 'convergence_warning': False}],
            'validation_metrics_at_0_5': {
                'average_precision': 0.5, 'roc_auc': 0.5, 'f1': 0.5,
                'precision': 0.5, 'recall': 0.5, 'accuracy': 0.5,
            },
        }

    monkeypatch.setattr(training, 'load_partition', fake_load_partition)
    monkeypatch.setattr(training, 'train_family', fake_train_family)
    training.run_experiment(output, jobs=1)

    assert partitions == ['train', 'val']
    assert families == ['dummy', *training.MODEL_NAMES]
    completion = json.loads((output / 'completion.json').read_text(encoding='utf-8'))
    assert completion['test_evaluated'] is False
