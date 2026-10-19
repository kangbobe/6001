# Member D reproducibility and approval handoff

## Delivered state

Base commit `bf53eab1ef5f1fb5729af38660b76af54497bdab`; isolated branch `member-d/validation-integration`. No remote push or final test run is part of this delivery. Original A/B/C code, data, models, splits and experiment outputs are preserved; README receives an additive pointer to D materials.

Analysis uses Python 3.12.14 on macOS arm64 with all **68 exact versions** in the original `requirements.lock.txt`. A/B/C used Python 3.12.10 on Windows. The patch-version/platform difference is disclosed, not represented as an identical historical environment. All five fitted pipelines reproduce the saved validation scores to maximum absolute difference **2.220446049250313e-16**, original metrics agree within 1e-12, and all selected-threshold classifications agree exactly. No dependency lock is changed. `pip check` reports no broken requirements; its cache-permission warning is unrelated to package compatibility.

## Reproduce in a fresh Python 3.12 environment

From the repository root, install the original dependency lock into a local environment (for example `python3.12 -m venv .venv`, then the environment's `python -m pip install -r requirements.lock.txt`). On Windows use `.venv\Scripts\python.exe` in place of `python`; on macOS/Linux activate `.venv` or use `.venv/bin/python`.

The delivered result directories deliberately refuse overwrite. Use a new output directory and a new proposal path for every reproduction:

```bash
python -m pytest -q -k 'not test_exact_split_coverage_and_partition_alignment'
python scripts/evaluate_validation.py --stage evaluation --output work/d-reproduction --protocol-output work/d-reproduction-proposed.json
python scripts/evaluate_validation.py --stage errors --output work/d-reproduction
python scripts/evaluate_validation.py --stage importance --output work/d-reproduction
```

No training script or final-test CLI is needed. The validation CLI has no test-unlock option. If a stage fails after its directory is created, inspect and retain that directory and its failure log, then use a new output root; do not delete a failed attempt to pretend it was the first run.

The one deselected legacy test calls A's full-table loader. D's stricter task boundary excludes that call because it materializes test rows before slicing train. All remaining original tests and the new tests were run. Synthetic legacy tests exercise training/final-evaluation control flow on tiny synthetic data; they do not retrain project pipelines or access real test records. D's new selective-loader tests separately cover row order, identity rejection, label alignment, and exclusion of invalid non-validation bytes from decoding/parsing.

## Actual checks performed

- Initial legacy checks: **29 passed, 1 deselected**.
- New tests: **63 passed** (51 validation/protocol/guard cases and 12 diagnostic cases).
- Final combined run: **92 passed, 1 deselected**.
- Every stage verified source/input/artifact identities, train-only metadata, CV selection IDs, classes `[0,1]`, row alignment, score reproduction, stored decisions, original metrics, and fresh selected decisions.
- Runtime guards rejected fitting methods; fitted-object hashes and protected-file hashes remained unchanged after every stage.
- Output reconciliation checked all row outcomes, all 11 group partitions per model, threshold-independent metrics, 550 permutation drops and 55 mean/SD summaries, and zero Dummy importance.
- A second complete validation run with identical settings reproduced all **15 numerical/figure/threshold output files byte-for-byte**. Receipt timestamps differ by design. This is computational reproducibility, not new independent statistical evidence.
- All 131 original tracked files other than the intentionally extended README retain their original SHA256; the original tracked inventory contains 132 files.
- Four generated figures were visually inspected for labels, legends, clipping and consistency with exported CSVs.
- No `config/final_protocol.json` or real `results/experiment/final_test` exists.

Machine-readable checks are in `results/member_d/qa/verification.json`; test/run logs and the initial refused preflight log are retained in the same directory. Per-stage `receipt.json` files record package versions, source hashes, all protected artifact hashes, output hashes, score differences, thresholds and non-test status. They are D-stage provenance, not proof of the original training-time source.

## Reproducibility observations and repaired issues

The first preflight compared current training source with the earlier `post_training_review` snapshot and refused the mismatch. Inspection found only the documented upstream addition of future-training source capture. The later original `posthoc_verification.json` matches all 18 delivered project-file identities. The final loader verifies that later receipt for project/experiment artifacts and the earlier manifest for frozen input identities. The failed preflight read no validation or test records and created no output stage. This was a D preflight assumption corrected after evidence, not an A/B/C model defect.

Before final output generation, review also added rejection of destinations within frozen data/experiment directories, including symlinks, and strict bounds for both inferred probability columns. Synthetic regressions cover both. The initial successful evaluation and proposal were retained under ignored `work/`; final outputs were regenerated using the same settings and unchanged thresholds. No model/feature/threshold rule was selected from repeat-run variation.

## Output map

- `evaluation/metrics.csv`: five models at 0.5 and selected thresholds; all rows are validation.
- `evaluation/threshold_sweep.csv`, `thresholds.json`: full decision-set search and selected cutoffs. Dummy sweep is descriptive; its cutoff remains fixed at 0.5.
- `evaluation/roc_points.csv`, `pr_points.csv`, `confusion_matrices.csv`, `figures/`: complete curves and both decision rules. PR terminal threshold is blank; the initial ROC all-negative threshold is infinity.
- `errors/row_errors.csv`, `error_groups.csv`, `model_disagreements.csv`, `figures/`: selected-threshold diagnostic counts/rates. Missing group rates mean zero denominator, not a fabricated value.
- `importance/permutation_repeats.csv`, `permutation_summary.csv`, `figures/`: all 10 AP drops for every feature/model and shuffle SDs (`ddof=0`).
- `config/final_protocol.proposed.json`: exact thresholds, unchanged pipeline hashes, input identities, primary/report metrics and validation-receipt identity. It remains `frozen=false` with blank owner/freeze time and explicit approval-pending status.

## Approval boundary and later integration

The proposal is ready for explicit review. F1 was selected as an academic precision/recall compromise without business costs; maximum validation F1 is not profit-optimal or an unbiased generalization estimate. RF's prior training-CV recommendation is preserved; all five models remain in the final comparison.

Only after explicit approval: enter the accountable Member D owner and actual timezone-aware freeze time, retain all exact thresholds/model identities, set the status to approved and `frozen=true`, and save a separate `config/final_protocol.json`. Verify all proposed data/configuration/source/receipt hashes against the delivered files before the existing final gate is used. Its current implementation enforces pipeline hashes but does not enforce the proposal's extra input/configuration hashes. These additional identities must therefore be checked in the approval preflight. The existing one-shot final evaluator must retain its success or failure directory. Do not alter hyperparameters, thresholds, features, model identities or included families based on any test outcome.

This delivery intentionally stops before those approval-dependent actions. The final ten-section, ten-page CA6001 report still needs group integration and approved test evidence; `MEMBER_D_REPORT.md` supplies validation-supported text, not a fabricated completed final report.
