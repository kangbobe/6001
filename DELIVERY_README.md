# CA6001 Member D support delivery — v2

**最终 Test 已完成，仅可从保存结果复核或重新生成展示材料。** 本包整理已存在的 D 工作及配套仓库，补充安全审计入口；没有重新训练、推理、调参、校准、阈值选择、验证分析或最终评估。所有原模型、数据、划分、冻结协议、预测、指标和四个最终原始文件保持不变。

这是 **Member D 支持材料与完整配套仓库快照**，不是已经完成的小组总交付。完整小组报告、slides 和录音仍需团队整合；[课程要求](notes/TEMPLATE_REQUIREMENTS.md)仍适用。

## 先读与安全复核

阅读 [最终结果与讨论](notes/FINAL_TEST_REPORT.md)、[当前复现说明](notes/MEMBER_D_REPRODUCIBILITY.md)及[本轮修复记录](notes/DELIVERY_REPAIR.md)。在仓库根目录、原锁定依赖环境下，只向一个新文件运行保存结果审计：

```bash
python scripts/audit_saved_results.py --root . --output work/audit-check-001.json
```

`--output` 必须显式给出。已有文件、冻结产物路径及其符号链接目标均拒绝写入；重复检查时换用新文件名。旧 `results/member_d/final_test/qa/independent_metrics_audit.py` 有固定覆写输出行为，连同旧 JSON 保留作历史证据，**不可直接运行**。安全新入口只复核保存结果，完全不读取原始数据行或反序列化模型。

可选的展示材料派生也只使用保存结果，且必须选择新目录：

```bash
python scripts/report_final_results.py --root . --output work/final-report-reproduction
```

这只是当前可用说明，本轮没有执行该派生。不要再次运行最终 evaluator、解锁 Test 或重新运行验证分析。原 A/B/C [交接说明](notes/HANDOFF_D.md)及 [preflight 说明](notes/FINAL_PREFLIGHT_HARDENING.md)为历史阶段快照，其中旧命令不是当前步骤。

## 本包范围

| 材料 | 路径与用途 |
| --- | --- |
| 既有实现与测试 | `src/`、`scripts/`、`tests/`；保留 A/B/C 与 D 源码、预检及验证测试，新增安全审计入口及其聚焦测试 |
| 验证方法、报告、审计 | [MEMBER_D_METHODS](notes/MEMBER_D_METHODS.md)、[MEMBER_D_REPORT](notes/MEMBER_D_REPORT.md)、[MEMBER_D_AUDIT](notes/MEMBER_D_AUDIT.md)，保留原正文并标历史身份 |
| 五份验证预测 | `results/experiment/{dummy,logistic_regression,decision_tree,random_forest,mlp}/validation_predictions.csv`，完整保留既有字节 |
| 验证阈值与比较 | `results/member_d/evaluation/`：`threshold_sweep.csv`、`thresholds.json`、指标、ROC/PR、混淆矩阵、图和 receipt |
| 验证错误分析 | `results/member_d/errors/`：逐行错误、分组率、模型差异、图和 receipt |
| 验证置换重要性 | `results/member_d/importance/`：逐次 AP drop、汇总 CSV、图和 receipt；仍为 validation 证据 |
| 原始最终 Test | `results/experiment/final_test/` 中四个原始文件，保存不变 |
| 最终展示及历史 QA | `results/member_d/final_test/`：CSV、PNG/SVG、派生 receipt、审计、日志与旧清单 |
| 配套复核资源 | 原数据、固定划分、五个模型、配置、依赖锁、notebook、引用和许可；不含 `.git`、虚拟环境及临时工作目录 |
| 文档原字节快照 | [SNAPSHOT_MANIFEST.json](notes/history/delivery-v1/SNAPSHOT_MANIFEST.json) 记录修改前八份文档的路径与 SHA256；旧清单与旧 receipt 不回填 |

上述验证材料只是整理和纳入完整交付，未重新计算。五份验证预测的实际目录名称以清单为准。每个包内文件的相对路径、大小和 SHA256 由新独立 `DELIVERY_MANIFEST.json` 记录；清单自身排除，避免自引用。旧 `results/member_d/final_test/SHA256SUMS.json` 是上一阶段快照，修改前文档字节可在历史文档快照中核对，不能当作当前全目录清单。

## 包名、版本和旧子集

新完整配套包名：`CA6001-Member-D-support-delivery-v2.zip`。包内 `PACKAGE_INFO.json` 记录对应本地 Git 提交；ZIP 旁的 `DELIVERY_INDEX.json` 记录完整包文件名、SHA256、提交和清单身份。ZIP 自身 SHA256 放在包外，避免把自身哈希写进自身造成循环依赖。核查时应同时保留该外部索引。本轮未 push 或 merge；不能假设 GitHub 已有此本地交付版本。

旧 `CA6001-final-evaluation.zip` 原样保留，SHA256 为 `4321c43de21d7d57e62b1907d30c77cf0270cb7608f16bec8dc4e0daf31b20b6`。它仅供最终结果阅读与局部保存结果复核，是 **结果子集**，不是独立完整项目。它缺少完整 D 源码/测试、验证方法与报告、五份 validation 预测、完整阈值扫描、错误分析、置换重要性的图/底层 CSV/receipt，以及部分交接、许可和仓库配套材料；其中某些相对链接依赖完整仓库。请使用上述 v2 完整配套包及其外部索引取得这些缺项。旧 ZIP、旧解压目录、旧入口文本和旧清单都不被本轮改写。

## 解释与可证明范围

AP 是按召回增量加权的 precision，不是梯形 PR 面积。RF 排序指标较强并不证明概率质量好；保存的 Test Brier/log loss 劣于 Dummy，未作校准。阈值由 validation F1 选取，因此 validation F1 存在选择乐观性；Test 差异不能直接归因于时间漂移或声称显著优劣。缺少客户 ID 和外部时间检验，行号互斥不能证明客户隔离或未来推广效果。

置换重要性和特征分组错误始终标为 **validation**。十次 permutation 的标准差是 shuffle SD（`ddof=0`），**不是置信区间**；重要性不建立因果关系。

历史日志记载 validation 分析、145 项通过/1 项排除的 pre-test 套件、冻结时间和一次最终调用；本轮未重跑这些历史步骤。当前检查以本轮修复记录为准，不用历史日志冒充当前执行。保留记录不能证明没有未记录的历史执行；原训练时源码字节未被捕获，后验身份校验不能补足该证据缺口，本轮不重建或回填历史 provenance。
