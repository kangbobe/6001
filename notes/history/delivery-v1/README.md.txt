# CA6001 — Bank Marketing Prediction

本项目完成 **A 数据分析/预处理、B 逻辑回归/决策树、C 随机森林/MLP**，并提供 D 可接手的模型和预测。原 CA6003 仓库保持只读。GitHub 仓库：https://github.com/raederhans/6001 。

研究问题：使用客户属性和历史营销信息，进行**营销启动前筛选的离线模拟**。不是营销因果效果估计，也不承诺真实部署时的字段可得性。数据采用官方 legacy `bank-full.csv`，45,211 条记录；只使用11个原始来源字段，`pdays` 进一步替换为“是否曾联系”的标志。

## 从哪里开始

**D 同学先读 [交接说明](notes/HANDOFF_D.md)**。仓库已包含原始数据、固定划分、五个已拟合管线、validation 预测及完整开发记录；接手不需要重新下载数据或重新训练。

1. `notes/TEMPLATE_REQUIREMENTS.md`：模板要求、10页预算和分工。
2. `notebooks/01_A_data.ipynb`：数据质量、EDA、预处理。
3. `notebooks/02_B_baselines.ipynb`、`03_C_advanced.ipynb`：模型依据、真实调参结果、验证集概率复核及诊断。相邻 `.html` 可直接阅读。
4. `notes/CA6001_ABC_Technical_Draft.docx`：便于小组整合的英文技术草稿，尚不是完整最终报告。
5. `notes/HANDOFF_D.md`：D 成员的输入、剩余工作和最终评估命令。

`results/experiment/development_summary.csv` 是开发结果，**不是最终测试结果**。默认所有脚本和 notebook 都不评估 test。最终结果和结论由 D 冻结阈值后完成。

## 本次已运行的结果

| 模型 | 训练3折平均AP | 验证集AP | 验证集ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.3419 | 0.3589 | 0.7210 |
| Decision Tree | 0.3358 | 0.3698 | 0.7248 |
| Random Forest | 0.3588 | 0.3865 | 0.7438 |
| MLP | 0.3327 | 0.3772 | 0.7376 |

Dummy 验证集AP为0.1170。随机森林是本轮训练CV推荐的家族，最终结论仍需D完成。AP是 Average Precision，不是准确率。

四模型36组配置的108次基础CV拟合已完成；另有Dummy的3折、5个完整train重拟合和21次预先规定的MLP补跑。MLP有6/24个候选折在600轮后仍警告；所选配置的3折最终均无警告（其中1折补跑），完整train重拟合185轮且无警告。全部记录保留，不以重试掩盖失败。

复查后30项行为检查通过；3本notebook已完整执行，重新加载四个模型及Dummy后，全部9,042行validation的行号、标签、概率和指标均与交付记录一致。图表与HTML阅读版已做显示检查。Word文件是A/B/C技术草稿；合并D内容后需再核对最终页数。

## Pro复查后的改进

完整处置见 `notes/PRO_REVIEW_RESOLUTION.md`。已统一原始11字段的575条与工程后582条重复画像口径，澄清字面值unknown和预测时未见类别的区别，补全报告训练协议，并收紧权重、概率与MLP收敛的解释。模型和原始训练成绩未替换，D仍负责最终阈值与test。

本次 `results/experiment/posthoc_verification.json` 是事后核验：记录当前源码/数据划分/依赖/冻结产物身份，并重算validation预测与指标。`post_training_review/` 保存审查时源码，不能证明训练当时的源码字节。未来训练会在拟合前保存 `source_snapshot/`，原实验协议未被回填。

## 本地运行

首次接手时，在 PowerShell 中克隆仓库并使用 Python 3.12 新建环境（本地虚拟环境不上传）：

```powershell
git clone https://github.com/raederhans/6001.git
cd 6001
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

读取数据无需网络，原始CSV和精确行号划分已包含。源数据SHA只用于确认版本身份。依赖完整锁定，scikit-learn=1.8.0。

先复核现有模型即可，不需要运行训练命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/verify_development.py --output work/validation-reproduction.json
```

上述复核只计算 validation 预测和指标；同名核验文件存在时会拒绝覆盖，重复运行时请换一个输出文件名。D 随后在 validation 上确定统一的评估及阈值规则，完成错误分析和特征重要性，再按 [HANDOFF_D.md](notes/HANDOFF_D.md) 冻结协议并运行一次最终 test。当前 `config/final_protocol.draft.json` 的 `frozen=false`，0.5 阈值为占位值。

以下为开发或重新生成材料的可选命令：

```powershell
# 数据审计和训练集图表
.venv\Scripts\python.exe scripts/run_eda.py
# 重新进行完整调参；新目录可保留随包交付的原始实验
.venv\Scripts\python.exe scripts/train.py --output results/rerun --jobs 2
# 生成并执行配套notebook，核对随包experiment结果（不重新调参）
.venv\Scripts\python.exe scripts/build_notebooks.py --execute
# 数据边界、指标和合成test入口测试；不读取真实test做评估
.venv\Scripts\python.exe -m pytest -q
# 再次独立复核validation；选择新收据名，保留原核验记录
.venv\Scripts\python.exe scripts/verify_development.py --output results/experiment/posthoc_verification-rerun.json
```

训练会先冻结 search spaces 和精确3折清单。四模型36配置、108次基础CV拟合；另有Dummy、最终重拟合和统一规定的MLP补跑。CPU训练耗时取决于硬件与收敛情况。输出目录存在时拒绝覆盖，不把旧、新结果混在一起。

## 实验原则

- train/validation/test = 27,126/9,042/9,043，复用6003精确行号。三个训练CV折供所有模型共用。
- 编码和缩放在各拟合折内部学习。unknown、负余额和极端值保留；不盲目去重、补成多数类或SMOTE。
- 每模型比较不加权和拟合折内balanced sample weights。按训练CV AP选配置；默认0.5阈值指标只是开发诊断。
- MLP不使用内部accuracy早停；300轮触顶且警告则相同配置/种子重跑600，保留每次日志。
- 已拟合管线在完整train训练后交接，validation可供D选阈值，默认不train+validation重训。
- 数据按历史记录划分；没有客户ID，无法证明客户隔离。时间外推未验证。完整规则见 `notes/METHODOLOGY.md`。

## 文件地图

| 目录 | 用途 |
|---|---|
| `data/raw`、`data/source` | 未改动CSV、原始说明、来源记录 |
| `data/splits`、`config` | 行号划分、数据锁、字段政策、D协议示例 |
| `src`、`scripts` | 可复用实现和命令行入口 |
| `results/eda` | 训练集审计、5张图和底层CSV |
| `results/experiment` | 每模型配置、逐折指标、损失/警告、模型、validation概率和开发汇总 |
| `notebooks` | 已执行配套实验本及HTML阅读版 |
| `notes`、`references` | 方法、交接、技术草稿、Pro原文、课程模板 |
| `tests` | 与数据泄漏、对齐、指标及最终入口相关的行为检查 |

复查结论和修改记录见 [PRO_REVIEW_RESOLUTION.md](notes/PRO_REVIEW_RESOLUTION.md)。模型建议不是训练证据，报告中的数字来自本地CSV/JSON。Pro 原始对话、截图、本地环境和临时文件不上传；方法、修订和本地验证证据已保留。数据引用和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。`.gitattributes` 保留文件字节，避免跨系统换行转换破坏数据锁和实验文件校验。

## Member D — validation evaluation and approval handoff (2026-10-19)

Member D's validation-only work is in [report support](notes/MEMBER_D_REPORT.md),
[methods](notes/MEMBER_D_METHODS.md), [audit](notes/MEMBER_D_AUDIT.md), and
[reproduction instructions](notes/MEMBER_D_REPRODUCIBILITY.md).
The validation delivery at `29657cf` preserved the original A/B/C artifacts and source. `results/member_d/` contains
threshold searches, model comparisons, ROC/PR curves, confusion matrices, feature-group
errors, repeated AP permutation importance, and identity/QA receipts.

`config/final_protocol.proposed.json` contains the actual proposed thresholds but remains
**frozen=false** as the preserved historical proposal. Its approved, frozen counterpart
is now `config/final_protocol.json`; the authorized final test is complete.
The original `final_protocol.draft.json` is preserved as the A/B/C handoff record.
Validation results must not be described as final-test performance.

The D validation CLI uses a selective reader and never parses real test records.
Its fresh reproduction commands and the explicitly excluded legacy full-table-loader
test are documented in `notes/MEMBER_D_REPRODUCIBILITY.md`.

The subsequent [final preflight hardening](notes/FINAL_PREFLIGHT_HARDENING.md) automatically
checks the approved data, splits, five pipelines, validation configuration/receipt,
and protocol consistency before any test loading. It requires explicit recorded
approval and preserves the one-shot mechanism. The historical proposal and selected thresholds
remain unchanged. Historical validation reproduction uses a separate
checkout at `29657cf`; historical receipts have not been rewritten.

## Member D — authorized final evaluation (2026-10-19)

The [final evaluation report](notes/FINAL_TEST_REPORT.md) records the explicit approval,
local preflight commit `ab46dfa`, 145 passing pre-test checks (one legacy case excluded),
and exactly one successful final evaluation. On 9,043 test records, the previously
recommended Random Forest achieved AP 0.373566, ROC-AUC 0.736447 and F1 0.401537.
Its recommendation, all five pipelines and all selected thresholds were retained.
No retraining, test-driven model selection, push or merge occurred.

Immutable one-shot outputs are under `results/experiment/final_test/`; derived figures,
validation/test comparisons and QA evidence are under `results/member_d/final_test/`.
Use `scripts/report_final_results.py` only to reproduce figures from saved predictions
in a new output directory. The real final evaluation must not be rerun.
