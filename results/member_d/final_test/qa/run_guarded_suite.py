"""Audit-only harness: repository tests with real raw/model read access blocked."""
import builtins
import io
import json
from pathlib import Path
import sys

# Run from the repository root, or pass its path as the first argument.
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
if not (ROOT / 'src/final_evaluation.py').is_file():
    raise SystemExit('Run from the repository root or supply its path as the first argument')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import joblib
import pytest

denied = []
raw = (ROOT / 'data/raw/bank-full.csv').resolve()
models = {(ROOT / 'results/experiment' / name / 'pipeline.joblib').resolve()
          for name in ['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'mlp']}

def path_of(value):
    try:
        return Path(value).resolve()
    except (TypeError, ValueError):
        return None

def wrap_open(original):
    def guarded(file, *args, **kwargs):
        target = path_of(file)
        if target == raw or target in models:
            denied.append(str(target))
            raise AssertionError(f'Audit harness refused access to real raw data or model artifact: {target}')
        return original(file, *args, **kwargs)
    return guarded

builtins.open = wrap_open(builtins.open)
io.open = wrap_open(io.open)
result = pytest.main(['-q', '-p', 'no:cacheprovider', '-k',
                      'not test_exact_split_coverage_and_partition_alignment',
                      f'--junitxml={OUT / "guarded-suite.xml"}', 'tests'])
(OUT / 'guarded-suite-access.json').write_text(json.dumps({
    'pytest_exit_code': result, 'blocked_access_attempts': denied,
    'protection': 'builtins.open and io.open reject real raw bank-full.csv and five real pipeline.joblib paths',
    'excluded': 'tests/test_data.py::test_exact_split_coverage_and_partition_alignment',
}, indent=2) + '\n')
sys.exit(result)
