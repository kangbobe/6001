"""Generate train-only EDA tables and figures for the CA6001 report."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT
from src.eda import run_eda


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/eda',
                        help='directory for CSV, JSON, and PNG outputs')
    args = parser.parse_args(argv)
    output = run_eda(root=ROOT, output_dir=args.output)
    print(f'Train-only EDA complete: {output}')
    return output


if __name__ == '__main__':
    main()
