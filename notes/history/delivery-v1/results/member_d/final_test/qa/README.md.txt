# Final-evaluation QA evidence

The pre-test records archive the clean hardening commit, 145 passing tests, the excluded legacy full-table-loader case, protected hashes, exact dependency versions and structural split checks. `verify_pretest.py` is the historical pre-test procedure and intentionally refuses an already-frozen/post-test checkout; it must not be used to rerun the experiment.

`protocol-freeze.json` records the explicit human authorization and unchanged thresholds before evaluation. `final-invocation.json` records the sole real evaluator invocation and exit code 0. Original one-shot outputs are preserved separately and must not be overwritten or reevaluated.

The independent audit reads saved predictions, metrics, protocol/receipt and split row IDs only. It can be reproduced with the locked environment from the repository root:

```bash
python results/member_d/final_test/qa/independent_metrics_audit.py .
python results/member_d/final_test/qa/smoke_reporting.py .
```

The second command uses synthetic saved outputs only. Reporting regeneration uses `scripts/report_final_results.py` and a new destination, without model inference. The final report and figures were independently reviewed; all numeric tables, source and derived-output hashes matched, and all three actual figures were visually inspected for clipping, axes, legends and consistency. A minor wording correction clarified that LR and MLP AP agree only at three-decimal display precision. No score or threshold was changed.
