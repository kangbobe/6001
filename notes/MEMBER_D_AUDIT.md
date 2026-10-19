> **历史阶段快照 / Historical stage snapshot.** 下方正文保留原样，描述当时的 validation 或 preflight 状态；“尚未最终测试”“待批准”及其中命令不是当前操作要求。最终 Test 现已完成，本轮只整理保存材料并修复安全审计入口，没有重跑该阶段。当前说明见 [交付入口](../DELIVERY_README.md)和[复现说明](MEMBER_D_REPRODUCIBILITY.md)；最终结果见 [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md)。原文的执行次数、测试与复现结论由当时日志支持，不是本轮新执行证据。

# Member D audit and implementation scope

Audited upstream commit: `bf53eab1ef5f1fb5729af38660b76af54497bdab`.
Working branch: `member-d/validation-integration`. Audit preceded project edits.

## Already complete and reused

A/B/C supplied the locked legacy Bank Marketing data, exact 27,126/9,042/9,043 train/validation/test membership and order, train-only three-fold tuning, four selected fitted pipelines plus Dummy, aligned validation score files, original metrics, experiment metadata, post-training verification, and a guarded one-shot final-test evaluator. No verified technical defect requires retraining.

Read README, HANDOFF_D, METHODOLOGY, PRO_REVIEW_RESOLUTION, draft protocol, development summary, shared metrics, final evaluator/CLI, all six existing test modules, preprocessing/model definitions, source provenance, and CA6001 template requirements.

`src/metrics.py` correctly computes unweighted yes=1 metrics, inclusive score >= threshold decisions, explicit confusion order, and sklearn Average Precision. Reuse unchanged. `src/final_evaluation.py` checks frozen status, owner, timezone-aware freeze time, model names/order, hashes, train-only metadata and explicit unlock; its output directory refuses overwrite even after an inference failure. Reuse unchanged.

## Member D additions

Validation threshold selection and comparison; ROC/PR curves; confusion matrices; feature-group FP/FN diagnostics and cross-model disagreement; AP permutation importance of all full pipelines; reproducibility and boundary tests; an approval-pending protocol; evidence-grounded report support.

New implementation: `src/validation.py`, `src/validation_analysis.py`, `scripts/evaluate_validation.py`.
New tests: `tests/test_validation.py`, `tests/test_validation_analysis.py`.
New configuration: `config/member_d_analysis.json`, `config/final_protocol.proposed.json`.
New documentation: this audit, MEMBER_D_METHODS, MEMBER_D_REPORT, MEMBER_D_REPRODUCIBILITY.
New generated results: `results/member_d/evaluation`, `errors`, `importance`, with stage receipts and figures. README receives an additive Member D entry.

## Preserved files

All original A/B/C source, scripts, tests, notebooks, report drafts, data bytes, split files, feature policy, dependency lock, original draft protocol, fitted model files, CV records, predictions, development metrics, snapshots and historical receipts are preserved. Before/after SHA256 checks cover original tracked files; each D stage also hashes protected input/artifact files and fitted in-memory objects. No training entry point is run. Synthetic tests may fit tiny synthetic estimators in the original suite; these are not project model retraining.

## Verified integration risks and resolution

1. A's `load_partition('val')` and verification script parse the full CSV before selecting validation. D's stricter reader scans/hashes raw bytes, but decodes/CSV-parses only validation lines. Neither A's loader nor its verification CLI is used on real data during D analysis. Byte-level identity checks and split membership metadata are not test performance analysis. Do not claim that no bytes from the shared raw file were read.
2. One legacy test calls A's full-table reader. It is excluded by name in this session; remaining legacy tests use synthetic data or a test-access refusal check. New tests cover validation row order and isolation separately. This is not an unqualified full-suite pass.
3. CSV round-tripping and cross-platform inference can differ at floating precision. Read saved scores with `float_precision='round_trip'`, select interior midpoint thresholds where representable, and require identical thresholded decisions from freshly loaded fitted models.
4. Balanced RF scores are not assumed calibrated; accuracy or a universal 0.5 cutoff is an incomplete comparison. Preserve AP as the primary ranking metric and present threshold decisions separately.
5. Validation threshold tuning makes selected F1 optimistic. Error groups are exploratory and do not authorize feature or model selection changes. Training-CV RF recommendation remains unchanged.
6. Original training-time source bytes were not captured. The existing honest posthoc limitation remains; D's new source identities establish D-run provenance only.
7. The final evaluator enforces pipeline hashes, but does not enforce additional data/configuration/source identities in the D proposal. Before an approved final run, check those identities and preserve every model result. Do not infer approval from merely having a proposal file.

## Stop boundary

No real test data frame, predictions or performance are produced. No `--unlock-test` command is run. The proposal remains `frozen=false`; no `config/final_protocol.json` is created. Explicit user approval, owner and actual freeze timestamp are required before the next phase.

## Provenance preflight finding

The first D preflight additionally compared delivered sources with the earlier post-training snapshot and refused `src/training.py`. Inspection showed exactly the already-documented upstream changes that add `capture_provenance` before future training and link its manifest in future run protocols. The later `posthoc_verification.json` matches all 18 delivered project-file identities. D therefore verifies delivered project/artifact identities against that later receipt, while the earlier snapshot supplies the frozen input identities. This is not a changed training result or a reason to retrain. The failed preflight log is retained.

A subsequent review added early rejection of output paths inside frozen data/experiment directories and strict [0,1] checks on both fresh probability columns. The first successful validation run and its proposal were retained under ignored `work/`; the final run uses the same predeclared settings and thresholds. No previous test exposure or results were involved.
