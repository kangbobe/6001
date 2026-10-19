# Member D 有限交付修复记录 — v2

本轮只整理已保存材料、修复审计输出行为和文档入口。未执行训练、调参、校准、模型或阈值选择、validation 分析、绘图重生成、`evaluate_test.py` 或真实 Test 推理。原结果不替换；本包是 D 支持材料，完整小组报告、slides、录音仍待团队整合。

## 修改与纳入范围

以本轮开始的文件字节为基准，实际修改的八份文档如下。修改前字节已保存于 [文档快照清单](history/delivery-v1/SNAPSHOT_MANIFEST.json)，未改写任何旧 receipt、旧审计脚本/JSON 或旧清单。

| 文件 | 修复 |
| --- | --- |
| `README.md` | 顶部明确最终 Test 已完成；保留旧 A/B/C 正文并标为历史；当前步骤仅指向保存结果复核 |
| `notes/MEMBER_D_REPRODUCIBILITY.md` | 新安全入口；原复现正文保留于历史折叠区 |
| `notes/FINAL_TEST_REPORT.md` | 保留结果和统计解释；区分历史执行证据与本轮检查，更新安全命令 |
| `results/member_d/final_test/qa/README.md` | 更新当前 QA 入口，禁止直接执行会覆写 JSON 的历史入口 |
| `notes/FINAL_PREFLIGHT_HARDENING.md` | 仅新增历史快照说明，原正文保留 |
| `notes/MEMBER_D_METHODS.md` | 仅新增历史快照说明，原正文保留 |
| `notes/MEMBER_D_REPORT.md` | 仅新增历史快照说明，原正文保留 |
| `notes/MEMBER_D_AUDIT.md` | 仅新增历史快照说明，原正文保留 |

新增 `scripts/audit_saved_results.py`、`tests/test_safe_saved_audit.py`、根目录 `DELIVERY_README.md`、本记录、八份 `.txt` 文档快照及其独立清单，以及 `results/member_d/delivery_repair/` 中的本轮测试、完整性和范围检查记录。

本轮开始时最终交付尚有未跟踪文件。将已有 `config/final_protocol.json`、四个 `results/experiment/final_test/` 文件、`results/member_d/final_test/` 原结果/图/QA/清单、`scripts/report_final_results.py` 和最终报告纳入本地版本管理，不代表本轮重新生成这些实验产物。起始状态保存在 [git-status-before.txt](../results/member_d/delivery_repair/git-status-before.txt)。其中最终报告和 QA README 只有上述文档更新，其余字节不变。

已有 D 源码、测试、验证方法/报告/审计、五份 `validation_predictions.csv`、`threshold_sweep.csv`、验证错误分析、置换重要性图和底层 CSV/receipt 已在分支中，本轮完整保留并打包。具体路径及类别见 [scope-inventory.json](../results/member_d/delivery_repair/scope-inventory.json)。所有 validation 标记保留，没有把它们改称 Test 分析。

## 安全入口与实际执行检查

新入口要求显式 `--output`，在读取审计输入前拒绝已有文件、悬空输出链接及受保护目录。检查同时覆盖输入副本和实现所在仓库，解析父目录符号链接，并使用目录身份比较防止 macOS 大小写别名绕过。最终使用排他创建模式，拒绝验证后出现的同名文件。

历史审计脚本和 JSON 保持原字节。新入口验证历史脚本 SHA256，再直接执行同一份已验证源码字节的计算函数；不使用会在历史目录生成 `__pycache__` 的导入加载器。历史断言必须开启，因此拒绝 Python `-O`。复用的计算只读六份保存结果/划分文件，无模型反序列化或原始数据行解析。

本轮实际执行：

```bash
../venv/bin/python -m pytest tests/test_safe_saved_audit.py -q --junitxml=../delivery-repair-v2/safe-audit-tests.xml
```

最终结果为 **21 passed in 0.32s**。见 [文本日志](../results/member_d/delivery_repair/safe-audit-tests.log)和 [JUnit XML](../results/member_d/delivery_repair/safe-audit-tests.xml)。运行环境为现有 Python 3.12.14 虚拟环境；本轮没有重新安装依赖或声称重新验证全部 68 个锁定版本。

成功用例复制六份已保存文件到临时目录，执行保存结果算术审计；该目录没有原始数据、模型或推理代码。输入副本和历史 JSON 字节不变，保存指标复算误差不超过 `1e-13`。这不是新的模型性能实验。其余用例覆盖已有文件拒绝、八类受保护目录、alternate-root 保护、符号链接、显式输出参数、排他写入竞争、历史源码身份不符、禁止字节码写入、大小写别名、损坏指标拒绝和 `-O` 拒绝。大小写敏感文件系统上两项别名测试会明确跳过；本机两项均实际通过。

首轮 18 项测试通过后，静态复核发现历史导入器的字节码副作用，以及本机大小写路径比较缺口；两处均修正并补测，最终运行是上述 21 项。没有失败测试被隐藏或删除，也没有为通过测试放宽冻结结果保护。

另实际执行了文件字节与 SHA256 检查、相对文件链接检查、范围清单核对和 `git diff --check`。详细机器记录见 [integrity-verification.json](../results/member_d/delivery_repair/integrity-verification.json)。本轮没有运行完整旧套件，避免将其中开发流程误当成本次修复的必要步骤。

完整暂存区的 `git diff --cached --check` 返回 2：三份既有 SVG 共 15,186 处行尾空格（混淆矩阵 971、ROC/PR 12,964、对比图 1,251）。这些是原生成文件的字节格式，不是本轮新增绘图问题；三份哈希均与修复前相同。为遵守结果保留约束，没有格式化或重写它们。排除这三份原文件后，其余暂存代码与文档的同项检查返回 0；诊断及处置记录于 `results/member_d/delivery_repair/whitespace-check.json`，不宣称全量格式检查通过。

## 完整性与历史证据

- 本轮开始清单记录的 157 个配置/数据/结果/依赖文件中，156 个字节不变；唯一差异是计划内的最终 QA README，其修改前字节另存快照。没有任何实验结果差异。
- 对历史 pre-test 清单中的 **99 个保护文件**重新计算当前 SHA256，全部一致。这是本轮实际完整性检查，不是重跑历史 preflight。
- 对原 main `bf53eab1ef5f1fb5729af38660b76af54497bdab` 的 **99 个 A/B/C 配套文件**逐字节核对：数据、配置、结果、notebook、references，以及原交接/方法/复查说明和依赖锁，全部一致。这组 99 与上一项是不同清单。
- 四个原始最终文件 SHA256 未变；配置协议副本也未变。五个模型、数据与划分、验证预测/阈值/指标、错误分析、重要性结果、所有历史 receipt 和历史审计脚本/JSON均未修改。
- 旧最终清单的 42 个条目仍可核对：38 项直接对应原文件，4 项旧文档哈希对应本轮保留的字节快照。旧清单自身未改写。
- 三个旧 ZIP 的 SHA256 全部与本轮开始值一致。八份文档原字节快照全部核对通过。新包独立生成清单，不回填旧证据。

历史日志支持的事项仍包括原 92/145 项测试、validation 重复分析、置换运行及其图表检查、冻结前访问保护、最终调用及执行时间。**本轮未重新执行这些步骤**。本轮保存结果测试可以验证已存指标与决定的内部一致性，不能证明从未有未记录的历史执行。原训练时源码字节证据缺失，不能靠后验校验补齐；本轮没有重建或回填 provenance。

AP 与梯形 PR 面积的区别、RF 概率质量限制、验证阈值乐观性、客户和时间泛化限制均保留。置换重要性仍是 validation 结果，其十次 shuffle 标准差不是置信区间，也不支持因果解释。

## 新交付与团队审核

完整配套仓库目录为 `member-d-delivery-v2/`，新 ZIP 为 `CA6001-Member-D-support-delivery-v2.zip`；包内 `PACKAGE_INFO.json` 记录本地 Git 提交，`DELIVERY_MANIFEST.json` 独立列出每个载荷文件的路径、大小和 SHA256，排除清单自身。包外 `DELIVERY_INDEX.json` 记录 ZIP、清单身份和打包核查结果。此安排避免把 ZIP 自身哈希嵌入自身。包的提交身份和最终哈希以该索引为准。

新包包含完整跟踪的配套仓库文件，不包含 `.git`、虚拟环境、缓存、临时目录或本机讨论记录。旧结果子集用途、缺项和哈希见 [DELIVERY_README.md](../DELIVERY_README.md)。未创建新的结果子集，也未将旧子集称为独立完整项目。

建议 PR 标题：**Complete Member D delivery and protect saved-result audit evidence**

建议 PR 描述：

> The previous result-only delivery omitted existing Member D validation support, and directly rerunning the historical arithmetic audit could overwrite its JSON evidence. Include the existing D sources, tests, validation predictions and analyses, final outputs and documentation in the local delivery; add an explicit-output audit entry that refuses existing and protected destinations while preserving the historical script and JSON.
>
> Mark development/preflight instructions as historical, retain byte-exact document snapshots, and provide a separate complete companion repository package and manifest. Keep all reported metrics, thresholds, validation identities and methodological limitations unchanged; the group report, slides and recording still require integration.
>
> Validation: 21 focused tests passed using temporary saved-result copies; current SHA256 checks passed for both 99-file protection inventories, the four original final files and all three existing ZIPs. All package files and relative file links are verified in the delivery index. No training, model inference, analysis regeneration, test evaluation, push or merge was performed.

以上为供队友审核的文本；本轮没有创建 GitHub PR、push 或 merge。
