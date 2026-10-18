"""Capture a bounded source snapshot and input-file identities for an experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal


Phase = Literal['before_model_fitting', 'post_training_review']
_PHASES = {'before_model_fitting', 'post_training_review'}
_FIXED_SOURCE_PATHS = (
    Path('scripts/train.py'),
    Path('config/dataset_lock.json'),
    Path('config/feature_policy.csv'),
    Path('requirements.lock.txt'),
)
_INPUT_PATHS = (
    Path('data/raw/bank-full.csv'),
    Path('data/splits/split_indices.json'),
    Path('data/splits/split_manifest.csv'),
)


def _record(path: Path, contents: bytes) -> dict[str, str | int]:
    return {
        'path': path.as_posix(),
        'bytes': len(contents),
        'sha256': hashlib.sha256(contents).hexdigest(),
    }


def capture_provenance(
    output_dir: str | Path,
    *,
    phase: Phase,
    root: str | Path | None = None,
) -> Path:
    """Write fixed source snapshots and input identities; return manifest path.

    Input data bytes are hashed but never copied. ``phase`` is caller-declared.
    """
    if phase not in _PHASES:
        raise ValueError("phase must be 'before_model_fitting' or 'post_training_review'")

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    output = Path(output_dir)
    source_paths = sorted((project_root / 'src').glob('*.py'))
    source_paths.extend(project_root / relative for relative in _FIXED_SOURCE_PATHS)
    source_contents = [
        (path.relative_to(project_root), path.read_bytes())
        for path in sorted(source_paths, key=lambda item: item.relative_to(project_root).as_posix())
    ]
    input_contents = [(relative, (project_root / relative).read_bytes()) for relative in _INPUT_PATHS]

    # mkdir with exist_ok=False is the final collision check and refuses to merge
    # a snapshot into any pre-existing output directory.
    output.mkdir(parents=True, exist_ok=False)
    source_records = []
    for relative, contents in source_contents:
        snapshot_path = output / 'source' / relative
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(contents)
        source_records.append(_record(relative, contents))

    limitations = ['The phase is caller-declared and is not independently verified.']
    if phase == 'post_training_review':
        limitations.append(
            'A post-training capture cannot prove which source bytes were used during training.'
        )
    manifest = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'phase': phase,
        'training_time_provenance': phase == 'before_model_fitting',
        'limitations': limitations,
        'source_files': source_records,
        'inputs': [_record(relative, contents) for relative, contents in input_contents],
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return manifest_path
