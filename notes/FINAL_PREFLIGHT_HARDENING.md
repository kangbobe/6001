> **历史阶段快照 / Historical stage snapshot.** 下方正文保留原样，描述当时的 validation 或 preflight 状态；“尚未最终测试”“待批准”及其中命令不是当前操作要求。最终 Test 现已完成，本轮只整理保存材料并修复安全审计入口，没有重跑该阶段。当前说明见 [交付入口](../DELIVERY_README.md)和[复现说明](MEMBER_D_REPRODUCIBILITY.md)；最终结果见 [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md)。原文的执行次数、测试与复现结论由当时日志支持，不是本轮新执行证据。

# Final evaluation preflight hardening

Date: 2026-10-19. Starting commit: `29657cf6d7bb2e8c3b65c13c1ab545c264b1fd2a` on `member-d/validation-integration`.

The independent audit's Medium preflight enforcement issue is addressed. The final evaluator now verifies the required experiment identities and recorded approval before creating its one-shot output directory, importing/calling the production partition loader, or deserializing any pipeline. No final evaluation was executed for this change. The existing proposed protocol remains byte-for-byte unchanged, unfrozen and pending explicit approval.

## Enforcement

| Required check | Enforcement before test loading |
|---|---|
| Raw dataset SHA256 | Hash the fixed `data/raw/bank-full.csv` path and compare with the protocol and validation receipt. |
| Split indices SHA256 | Hash the fixed `data/splits/split_indices.json` path; compare the same two identities. |
| Split manifest SHA256 | Hash the fixed `data/splits/split_manifest.csv` path; compare the same two identities. |
| Five pipeline SHA256 values | Retain existing actual-file hash checks for every model; also bind their approved hashes to the validation receipt. |
| Validation analysis configuration | Verify the exact bytes of `config/member_d_analysis.json` against both protocol and receipt. Require validation-only, yes=1, inclusive thresholds, unweighted F1 selection, AP ranking, and no retraining/test evaluation. |
| Validation evaluation receipt | Authenticate its exact bytes. Require a completed validation evaluation, unchanged artifacts/models, fitting blocked, and no test-row parsing or test performance computation. Bind completion/run-protocol/model metadata bytes to that receipt too. |
| Final protocol consistency | Require all experimental fields to match `config/final_protocol.proposed.json`; compare thresholds with both the receipt and authenticated `thresholds.json`; verify the complete metric schema and preserved CV family recommendation. |
| Explicit approval | Require `approval_status="approved"` as well as `frozen=true`, a nonblank owner, a timezone-aware freeze timestamp, and the existing explicit unlock gate. |

Missing files, missing required fields, invalid digests, inconsistent identities/settings, malformed JSON, invalid JSON text encoding, duplicate JSON keys, or unsupported protocol fields cause `EvaluationRefused`. The raw dataset is hashed as opaque bytes; preflight does not parse its rows or inspect labels. Fixed evidence paths prevent selecting alternative configuration/receipt files through protocol path fields.

The reviewed proposal is the local approval baseline. Freezing may change authorization fields (`frozen`, `owner`, `frozen_at`, `approval_status`) and notes; it must preserve the experimental settings and identities. Recorded approval is not a cryptographic authentication system: the reviewed proposal and local code remain trusted inputs. This implementation does not claim to establish historical training provenance or defend against concurrent adversarial modification of the working tree.

The original proposal's notes describe the gate before this change. This document supersedes their statement that additional hashes are not automatically enforced; the proposal itself is preserved as the audited record.

## One-shot behavior and thresholds

The fixed `results/experiment/final_test` directory still refuses overwrite. Preflight failure creates no directory. Once inference starts, any success or failure leaves the directory in place so a subsequent invocation cannot silently replace the attempt. No fitting method is added. The validated protocol bytes are now saved directly, rather than copying a file that may have been edited after validation. The eventual evaluation receipt will include the verified preflight identities.

All selected thresholds remain unchanged:

| Model | Threshold |
|---|---:|
| Dummy | 0.5 |
| Logistic regression | 0.16796284539462858 |
| Decision tree | 0.2586600685192234 |
| Random forest | 0.6112421038019304 |
| MLP | 0.1896097366729046 |

The actual proposal still has `frozen=false`, blank owner/freeze time, and `approval_status="pending_explicit_user_approval"`. Neither `config/final_protocol.json` nor the real `results/experiment/final_test` exists.

## Changes and tests

Changed implementation: `src/final_evaluation.py`. Added shared synthetic approval evidence in `tests/conftest.py` and 53 focused cases in `tests/test_final_preflight.py`. Updated only the synthetic fixture wiring in `tests/test_final_evaluation.py` and `tests/test_validation.py` so existing tests reach their intended gates. Added this note and updated README/reproduction guidance. No model, split, existing result, proposal, analysis configuration or dependency lock was edited.

Tests ran with Python 3.12.14 and the existing locked dependencies:

- Initial final-evaluator and validation regression run: **59 passed**.
- Final guarded combined suite: **145 passed, 1 deselected**. This includes all 92 previously retained tests and 53 new cases. No test failed.
- The guard rejected any attempt to open the real raw dataset or any of the five real pipeline files through `builtins.open`/`io.open`; **zero such attempts occurred**.
- Refusal cases independently corrupt each of the three inputs and five pipeline identities; exercise missing/malformed configuration, receipt, proposal and threshold evidence; alter approval/protocol settings; and break internal receipt links even after updating synthetic outer hashes. Every preflight refusal asserts zero partition-loader calls, zero model-loader calls and no final output directory.
- Synthetic success checks exact selected-threshold application, no fitting, the saved original protocol snapshot despite a later source-file edit, and refusal of a second evaluation. Existing alignment/class-order failure tests still verify retention of a failed attempt.
- A separate read-only compatibility check authenticated the real artifact/configuration/receipt chain, with CSV parsing and model deserialization blocked. The original pending proposal was then refused by the authorization gate. This was an identity check, not an evaluation or approval.

The excluded legacy case remains `tests/test_data.py::test_exact_split_coverage_and_partition_alignment`: it invokes the original full-table loader, which parses real test rows before selecting train. Its exclusion is unchanged and preserves the user's no-test-access boundary. Synthetic/selective-loader coverage remains active.

For the retained test scope, run from the repository root:

```bash
python -m pytest -q -p no:cacheprovider -k 'not test_exact_split_coverage_and_partition_alignment'
```

The delivery evidence includes the guarded harness, log, access report, JUnit results, read-only identity-check script/results, protected-file comparison, and exact change patch. All 99 protected files snapshotted before this task—data, original experiment artifacts, existing Member D outputs, proposal and analysis configuration—retain their SHA256 identities.

## Provenance and reproduction limits

Original training-time source bytes were not captured. The existing posthoc verification establishes delivered artifact/source identities, not which source bytes originally trained the models. No attempt was made to reconstruct, backfill, or strengthen that historical claim.

The historical handoff receipt includes the old `src/final_evaluation.py` source hash. The authorized gate change therefore intentionally causes the old strict validation reproduction check to refuse this checkout. Existing receipts remain unchanged; their checks are not weakened. Reproduce the original validation analysis in a separate checkout at `29657cf`, and use this hardened checkout for any subsequently approved final evaluation. This source-version distinction does not alter the validation evidence, selected thresholds or fitted pipelines.

Work stops here. Freezing a final protocol and executing the real one-time final test require the user's explicit approval. No unlock command, model retraining, GitHub push or final test was performed.
