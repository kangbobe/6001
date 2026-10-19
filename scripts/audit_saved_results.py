#!/usr/bin/env python3
"""Safely repeat the historical arithmetic audit of already saved predictions.

This entry point never loads a pipeline or raw dataset. It preserves the original
audit implementation and evidence, verifies the implementation's byte identity,
and redirects its output to an explicitly chosen, exclusively created file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType


REPOSITORY = Path(__file__).resolve().parents[1]
HISTORICAL_SCRIPT = REPOSITORY / 'results/member_d/final_test/qa/independent_metrics_audit.py'
HISTORICAL_SHA256 = '61723df994a573b2dbcd3a30d12e7d32600ee0e31ab5275a313a13eefae7383b'
# Protect all frozen inputs/results, plus code and historical documentation. The
# same rule applies to the implementation checkout AND an alternate input copy.
PROTECTED_DIRECTORIES = ('data', 'config', 'results', 'src', 'scripts', 'tests', 'notes', '.git')


class AuditRefused(ValueError):
    """Unsafe output destination or unsupported audit implementation."""


def validate_output(output: Path, root: Path) -> Path:
    """Resolve parent symlinks and refuse an existing or protected destination."""
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise AuditRefused(f'Output already exists; refusing to overwrite: {output}')
    resolved = output.resolve()
    for repository in {REPOSITORY.resolve(), Path(root).resolve()}:
        if resolved == repository:
            raise AuditRefused(f'Output is a repository directory: {resolved}')
        for name in PROTECTED_DIRECTORIES:
            protected = (repository / name).resolve()
            # Path equality alone misses case aliases on case-insensitive file
            # systems (e.g. macOS RESULTS vs results). Compare existing ancestor
            # identities as well; nonexistent descendants remain safe to check.
            aliases_protected = protected.exists() and any(
                parent.exists() and parent.samefile(protected) for parent in resolved.parents
            )
            if resolved == protected or protected in resolved.parents or aliases_protected:
                raise AuditRefused(f'Output is inside a protected/frozen directory: {protected}')
    return resolved


class ExclusiveOutput:
    """Minimal adapter for the preserved audit's one write_text call."""

    def __init__(self, path: Path, root: Path):
        self.path = path
        self.root = root

    def write_text(self, content: str) -> int:
        destination = validate_output(self.path, self.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after parent creation; 'x' also rejects a file created between
        # validation and open, including a final-component symlink.
        destination = validate_output(destination, self.root)
        with destination.open('x', encoding='utf-8') as stream:
            return stream.write(content)

    def __str__(self) -> str:
        return str(self.path)


def _load_historical_audit():
    source = HISTORICAL_SCRIPT.read_bytes()
    if hashlib.sha256(source).hexdigest() != HISTORICAL_SHA256:
        raise AuditRefused('Historical audit implementation SHA256 mismatch')
    # Execute exactly the checked bytes, without an import loader that would
    # create __pycache__ inside the preserved historical evidence directory.
    module = ModuleType('_preserved_member_d_audit')
    module.__file__ = str(HISTORICAL_SCRIPT)
    exec(compile(source, str(HISTORICAL_SCRIPT), 'exec'), module.__dict__)
    return module


def audit_saved_results(root: Path, output: Path) -> Path:
    if not __debug__:
        raise AuditRefused('Python -O is unsupported: historical assertions must remain enabled')
    root = Path(root).resolve()
    destination = validate_output(output, root)  # Before reading audit inputs.
    audit = _load_historical_audit()
    audit.REPO = root
    audit.FINAL = root / 'results/experiment/final_test'
    audit.OUT = ExclusiveOutput(destination, root)
    audit.main()
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPOSITORY,
                        help='Repository or temporary copy of the six saved audit inputs')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output file outside protected directories; never overwritten')
    args = parser.parse_args(argv)
    try:
        audit_saved_results(args.root, args.output)
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f'Audit refused: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
