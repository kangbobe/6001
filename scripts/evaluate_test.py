"""Evaluate frozen A/B/C pipelines on test only after an explicit protocol unlock."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.final_evaluation import run_final_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unlock-test', action='store_true',
                        help='explicitly authorize the one-shot test evaluation')
    parser.add_argument('--protocol', type=Path, required=True,
                        help='D-frozen protocol JSON with thresholds and pipeline hashes')
    parser.add_argument('--experiment', type=Path, default=None,
                        help='experiment directory (default: ROOT/results/experiment)')
    args = parser.parse_args(argv)
    if not args.unlock_test:
        parser.error('--unlock-test is required to evaluate the test partition')
    result = run_final_evaluation(unlock_test=True, protocol_path=args.protocol,
                                  experiment_path=args.experiment)
    print(f'Final test evaluation complete: {result}')
    return result


if __name__ == '__main__':
    main()
