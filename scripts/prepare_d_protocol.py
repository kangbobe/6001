"""Prepare a draft with current model identities; does not open test or freeze decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT
from src.final_evaluation import MODEL_SET

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'results/experiment')
    parser.add_argument('--output', type=Path, default=ROOT / 'config/final_protocol.draft.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Draft already exists; choose another output so D decisions are preserved')
    protocol = json.loads((ROOT/'config/final_protocol.example.json').read_text(encoding='utf-8'))
    completion = json.loads((args.experiment/'completion.json').read_text(encoding='utf-8'))
    if completion.get('status') != 'complete':
        parser.error('Training must be complete before creating the draft')
    protocol['pipeline_sha256'] = {name: hashlib.sha256((args.experiment/name/'pipeline.joblib').read_bytes()).hexdigest() for name in MODEL_SET}
    protocol['notes'] = 'Current pipeline identities are populated. Thresholds are unselected 0.5 placeholders. D must set owner, frozen_at, decision_rule and thresholds, then explicitly freeze. No test data have been accessed by this script.'
    args.output.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(args.output)
