"""Reproduce Member D validation stages without any test unlock option."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation import run_validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['evaluation', 'errors', 'importance'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/member_d')
    parser.add_argument('--protocol-output', type=Path,
                        help='New approval-pending proposal path (evaluation stage only)')
    args = parser.parse_args(argv)
    result = run_validation(stage=args.stage, output=args.output, protocol_output=args.protocol_output)
    print(f'Validation {args.stage} complete: {result}', flush=True)
    return result


if __name__ == '__main__':
    main()
