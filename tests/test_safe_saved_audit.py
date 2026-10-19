"""Delivery safety checks using temporary copies of saved results, never inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import audit_saved_results as safe


AUDIT_INPUTS = (
    'results/experiment/final_test/frozen_protocol.json',
    'results/experiment/final_test/evaluation_receipt.json',
    'results/experiment/final_test/test_metrics.csv',
    'results/experiment/final_test/test_predictions.csv',
    'results/member_d/evaluation/metrics.csv',
    'data/splits/split_indices.json',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved_copy(tmp_path):
    root = tmp_path / 'saved_copy'
    for relative in AUDIT_INPUTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe.REPOSITORY / relative, destination)
    # Deliberately no dataset, pipelines, src, or evaluation script in this root.
    return root


def test_success_uses_saved_copies_and_preserves_all_input_bytes(saved_copy, tmp_path, monkeypatch):
    before = {relative: digest(saved_copy / relative) for relative in AUDIT_INPUTS}
    historical_json = safe.HISTORICAL_SCRIPT.with_suffix('.json')
    historical_before = digest(historical_json)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != 'bank-full.csv' and path.suffix != '.joblib', 'Forbidden raw/model access'
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    output = tmp_path / 'new' / 'audit.json'
    assert safe.main(['--root', str(saved_copy), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['status'] == 'passed'
    assert result['n'] == 9043
    assert set(result['inputs_sha256']) == set(AUDIT_INPUTS)
    assert result['inputs_sha256'] == before
    assert {relative: digest(saved_copy / relative) for relative in AUDIT_INPUTS} == before
    assert digest(historical_json) == historical_before
    assert all(item['max_absolute_difference_from_saved_metrics'] <= 1e-13
               for item in result['models'].values())


def refuse_before_loading(monkeypatch):
    def forbidden():
        pytest.fail('Output refusal must happen before loading the historical audit or its inputs')
    monkeypatch.setattr(safe, '_load_historical_audit', forbidden)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    refuse_before_loading(monkeypatch)
    output = tmp_path / 'existing.json'
    output.write_bytes(b'original evidence\n')
    assert safe.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    assert output.read_bytes() == b'original evidence\n'


@pytest.mark.parametrize('directory', safe.PROTECTED_DIRECTORIES)
def test_protected_input_copy_paths_refused_before_read(tmp_path, monkeypatch, directory):
    refuse_before_loading(monkeypatch)
    output = tmp_path / directory / 'new-audit.json'
    assert safe.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    assert not output.exists()


def test_alternate_root_cannot_bypass_protection_of_original_repository(tmp_path, monkeypatch):
    refuse_before_loading(monkeypatch)
    output = safe.REPOSITORY / 'results/experiment/final_test/new-audit.json'
    assert safe.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    assert not output.exists()


def test_parent_symlink_into_frozen_directory_refused(tmp_path, monkeypatch):
    refuse_before_loading(monkeypatch)
    root = tmp_path / 'project'
    frozen = root / 'results/experiment/final_test'
    frozen.mkdir(parents=True)
    alias = tmp_path / 'alias'
    alias.symlink_to(frozen, target_is_directory=True)
    assert safe.main(['--root', str(root), '--output', str(alias / 'audit.json')]) == 1
    assert not (frozen / 'audit.json').exists()


def test_dangling_output_symlink_refused(tmp_path, monkeypatch):
    refuse_before_loading(monkeypatch)
    target = tmp_path / 'absent.json'
    output = tmp_path / 'alias.json'
    output.symlink_to(target)
    assert safe.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    assert output.is_symlink() and not target.exists()


def test_output_is_explicitly_required(tmp_path):
    with pytest.raises(SystemExit) as error:
        safe.main(['--root', str(tmp_path)])
    assert error.value.code == 2


def test_exclusive_creation_refuses_race_at_open(tmp_path, monkeypatch):
    output = tmp_path / 'audit.json'
    original_open = Path.open

    def race_open(path, mode='r', *args, **kwargs):
        if path == output and mode == 'x':
            with original_open(path, 'w') as stream:
                stream.write('concurrent evidence')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', race_open)
    with pytest.raises(FileExistsError):
        safe.ExclusiveOutput(output, tmp_path).write_text('replacement')
    assert output.read_text() == 'concurrent evidence'


def test_changed_historical_implementation_refused(tmp_path, monkeypatch):
    historical = tmp_path / 'changed.py'
    historical.write_text('raise RuntimeError("must not execute")\n')
    monkeypatch.setattr(safe, 'HISTORICAL_SCRIPT', historical)
    output = tmp_path / 'audit.json'
    assert safe.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    assert not output.exists()


def test_loading_preserved_script_does_not_write_bytecode(tmp_path, monkeypatch):
    historical = tmp_path / 'historical.py'
    shutil.copyfile(safe.HISTORICAL_SCRIPT, historical)
    monkeypatch.setattr(safe, 'HISTORICAL_SCRIPT', historical)
    assert callable(safe._load_historical_audit().main)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['historical.py']


@pytest.mark.parametrize('directory,alias', [('results', 'RESULTS'), ('data', 'DaTa')])
def test_case_insensitive_frozen_alias_refused(tmp_path, monkeypatch, directory, alias):
    refuse_before_loading(monkeypatch)
    root = tmp_path / 'project'
    (root / directory).mkdir(parents=True)
    if not (root / alias).exists():
        pytest.skip('Case-sensitive filesystem does not alias this path')
    assert safe.main(['--root', str(root), '--output', str(root / alias / 'audit.json')]) == 1
    assert not (root / directory / 'audit.json').exists()


def test_failed_saved_metric_check_creates_no_report(saved_copy, tmp_path):
    metrics = saved_copy / 'results/experiment/final_test/test_metrics.csv'
    metrics.write_text(metrics.read_text().replace('9043', '9044'))
    output = tmp_path / 'audit.json'
    assert safe.main(['--root', str(saved_copy), '--output', str(output)]) == 1
    assert not output.exists()


def test_optimized_python_refused_before_reading_inputs(tmp_path):
    output = tmp_path / 'audit.json'
    result = subprocess.run([sys.executable, '-O', str(safe.REPOSITORY / 'scripts/audit_saved_results.py'),
                             '--root', str(tmp_path), '--output', str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert 'assertions must remain enabled' in result.stderr
    assert not output.exists()
