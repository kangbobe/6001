# Final-evaluation QA evidence and current safe audit entry

These files are **historical evidence** of the completed final evaluation. The pre-test logs record the hardening commit, 145 passing tests, one excluded legacy full-table-loader case, protected hashes, dependency versions and structural split checks. Those tests were not rerun for this delivery repair. `verify_pretest.py` intentionally refuses an already-frozen/post-test checkout and is not a current execution instruction.

`protocol-freeze.json` records authorization and unchanged thresholds before evaluation. `final-invocation.json` records one real evaluator invocation and exit code 0; a retained log cannot prove the absence of unrecorded historical actions. The four one-shot outputs, protocol, predictions, metrics, original audit script/JSON and all other historical receipts remain unchanged. No Test inference may be repeated for this repair.

From the repository root, audit only the saved results into a **new** explicitly specified file:

```bash
python scripts/audit_saved_results.py --root . --output work/audit-check-001.json
```

The safe entry rejects existing output files and frozen destinations. It reads saved predictions, metrics, protocol/receipt and split row IDs, without raw-data loading, model deserialization, fitting or inference. **Do not execute `independent_metrics_audit.py` directly:** its historical output logic overwrites its adjacent JSON. Both historical files are retained byte-for-byte, and the safe entry reuses the preserved arithmetic without that output policy.

Optional presentation regeneration uses `scripts/report_final_results.py` and a new destination. This repair did not regenerate figures or validation analysis. Historical reporting smoke logs and visual-review statements remain evidence of earlier checks only; actual checks for the safe entry and delivery package are documented in [the repair record](../../../../notes/DELIVERY_REPAIR.md).

The old [SHA256 manifest](../SHA256SUMS.json) is not a current-directory manifest after documentation repair. Its document identities can be checked against the [byte-exact documentation snapshots](../../../../notes/history/delivery-v1/SNAPSHOT_MANIFEST.json). The new package has a separate manifest; old audit evidence is never rewritten to match new documentation. Current delivery scope is in [DELIVERY_README.md](../../../../DELIVERY_README.md).
