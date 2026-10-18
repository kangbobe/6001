# 给 D 成员的交接

A/B/C 提供可复现开发实验和完整已拟合管线。**本次没有计算真实 test 性能，也没有替你选最终阈值、完成错误分析或特征重要性。**

2026-10-18 已完成网页端 Pro 复查及本地修订：30 项测试通过，五个管线在全部 9,042 行 validation 上复现原概率与指标，模型文件身份与原协议草稿一致。详情见 `notes/PRO_REVIEW_RESOLUTION.md`。原训练时源码快照缺失仍如实保留，新增记录明确属于事后核验；未重训，阈值仍待你决定和冻结。

## 你收到什么

- `results/experiment/development_summary.csv`：每家族所选配置的训练CV AP均值/标准差，以及validation在0.5阈值下的完整指标。AP越高越好；Brier/log loss越低越好。CV标准差不是置信区间。
- `results/experiment/<model>/cv_candidates.csv`、`cv_folds.csv`：所有候选和逐折记录。`fit_histories.json` 保留尝试次数、损失和收敛警告。
- `pipeline.joblib`：整个特征派生、预处理和分类器。模型只在27,126条train上拟合，`classes_=[0,1]`；`predict_proba(X)[:,1]` 是yes分数。加权模型分数不默认等于校准概率。
- `validation_predictions.csv`：原始记录的0-based `row_id`，split=val，真实0/1标签、`p_yes`、0.5阈值预测。所有模型必须保持相同9,042行及顺序。
- `feature_names.json` 与 `model_metadata.json`：编码后的列名、最佳参数、refit记录、序列化复核结果。代码中 `src.data.FEATURES` 定义11个输入来源字段，row_id只用于对齐。
- `data/splits/split_manifest.csv` 和实验中的 `cv_manifest.csv`：精确划分。不要重新随机切分或把测试集并回开发。
- `notes/CA6001_ABC_Technical_Draft.docx`、三本notebook及图表：报告技术输入。模板正文最多10页，不含封面，最终排版需全组整合后核对。

## 建议的 D 流程

1. 阅读方法和开发结果，保持已冻结的模型/特征策略。跨家族推荐依据在 `completion.json` 的 `cv_recommended_family`，来自训练CV AP；不是“最终最好”的结论。
2. 在validation上统一选择四模型的阈值并写明依据。没有真实营销成本时，可采用F1最大且并列选较高阈值作为课程约定，不能称利润最优。也可以保持0.5，但要明确说明。
3. 做validation错误分组、原始字段的 permutation importance（完整pipeline、AP、固定种子，例如5次重复）。这是模型依赖，不是因果效应。重要性不稳定/共线性、小群体样本量、age等属性差异需有对应解释。
4. 冻结模型身份、阈值和报告指标。不要train+validation重训现有管线，否则现有阈值与概率映射会改变，必须另定协议。
5. 完成冻结后统一运行一次test，保留五模型全部结果和失败记录。不要根据test重新调参、换特征、改阈值或挑选更好看的种子。

## 冻结协议与最终入口

`config/final_protocol.draft.json` 会在训练完成后填入当前管线SHA256，但仍 `frozen=false`，owner/time留空，阈值0.5为未选择占位值。填入你的负责人标识、带时区的冻结时间、实际决策规则和各模型阈值后，另存 `config/final_protocol.json` 并设置 `frozen=true`。Dummy固定0.5。

如果重新训练了另一目录，先生成对应的新草稿：

```powershell
.venv\Scripts\python.exe scripts/prepare_d_protocol.py --experiment results/rerun --output config/final_protocol.rerun.draft.json
```

只有冻结后，由你运行下面命令；普通notebook“全部运行”不会调用它：

```powershell
.venv\Scripts\python.exe scripts/evaluate_test.py --unlock-test --protocol config/final_protocol.json --experiment results/experiment
```

成功时输出 `results/experiment/final_test/test_metrics.csv`、`test_predictions.csv`、`frozen_protocol.json` 和 `evaluation_receipt.json`。已有final_test目录时拒绝覆盖；若曾失败，先检查剩余文件和失败位置，不删除后假装是首次测试。

显式解锁是协作流程防误触机制，不是密码或访问控制；原始CSV仍含全部记录。joblib只加载本项目可信管线，并使用锁定的Python依赖。

## 报告措辞

目前可写“完成开发阶段四模型统一调参与验证”，不可写“最终测试证明随机森林最好”。三折最优配置分数存在选择偏差，validation参与阈值选择后也属于开发数据。最终test、错误分析和结论请据你实际产出补全。
