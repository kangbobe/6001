# CA6001 实验协议（A/B/C，2026-10-18）

本协议在四模型训练前确定。网页端 Pro 讨论记录保存在 `PRO_DISCUSSION_1.txt`；讨论提供方法建议，真实结果以本地运行文件为准。

## 研究问题与范围

使用客户属性和此前营销信息预测历史营销记录中的定期存款认购，作为**营销启动前筛选的离线模拟**。预测单位是数据中的记录，正类 `yes=1`。这是认购倾向预测，不是营销的因果增量效果；数据不包含可验证的客户标识或完整决策时点快照。

使用 UCI legacy `bank-full.csv`，45,211 条记录，16 个原始输入和标签。沿用 CA6003 原始字节和行号划分；6003 的代码和课程材料不被改写。数据来源、原始说明和许可在 `data/source`、`data/raw/bank-names.txt`、`THIRD_PARTY_NOTICES.md`。

## 特征与预处理

11 个原始来源字段：age、job、marital、education、default、balance、housing、loan、pdays、previous、poutcome。

- 排除 `duration`，因为通话结束前不可得；排除当前活动的 `campaign/contact/day/month`，避免把最终联系记录当作预先计划。
- `pdays=-1` 代表之前未联系。主实验将其替换为 `has_previous_contact=(pdays>=0)`，不保留精确天数，以减少不明确的时间基准假设。`poutcome=unknown` 保持原义，不被强行解释为“从未联系”。
- 保留负余额、极端值和 literal `unknown`。先审计，未找到业务依据时不删记录，不用全数据分布设截断点，不将相同画像认定为重复客户。
- 所有模型使用相同特征派生和 one-hot 编码；未知新类别用 `handle_unknown='ignore'`。LR/MLP 的数值列标准化，树模型不标准化。所有学习到的参数仅在当前训练折内拟合。
- 11 是原始字段来源数；最终编码维度以已拟合管线的 `feature_names.json` 为准。row_id 和 y 不进入输入。

## 划分、调参与阈值

保留 train 27,126 / validation 9,042 / test 9,043 的精确行号清单。原划分由分层、random_state=42 的两次切分生成。读取时验证身份、范围、互斥和覆盖，不重新决定划分。

`row_id` 是原始 CSV 去掉表头后的0-based行位置。复现时保留已有划分的行序和 `cv_manifest.csv`，不要自行排序后只凭seed重建。

训练集内部固定 `StratifiedKFold(3, shuffle=True, random_state=42)`；保存 `cv_manifest.csv`。按平均 Average Precision（sklearn `average_precision_score`，简称 AP）选择每个家族配置。AP 是召回增量加权的 precision，不能混称梯形积分 PR-AUC。并列取较低 config_id（顺序偏好较小架构、较强正则、较大叶节点）；跨家族并列按脚本中固定模型顺序。

搜索空间：

| 模型 | 参数 | 配置数 |
|---|---|---:|
| Logistic Regression | C=0.01/0.1/1/10 × weight=None/balanced | 8 |
| Decision Tree | max_depth=4/8/None × min_samples_leaf=100/20 × weight | 12 |
| Random Forest | max_depth=10/None × min_samples_leaf=20/5 × weight；200 trees，sqrt features | 8 |
| MLP | hidden=(32,)/(64,32) × alpha=0.01/0.0001 × weight；ReLU、Adam、batch=256、lr=0.001 | 8 |

四个家族共 36 配置、108 次基础 CV 拟合；另加 Dummy 的 3 折、5 个最终 train 重拟合及规定的 MLP 补跑。不是穷尽所有算法设置的性能上限实验。

四模型的 balanced 权重都由当前拟合子集的 y 计算，经 `model__sample_weight` 传入；不同时设置 class_weight。评估保持原类别分布且不加权。scikit-learn 固定为 1.8.0，已确认 MLP fit 支持 sample_weight。没有使用 SMOTE。

MLP `early_stopping=False, max_iter=300, tol=1e-4, n_iter_no_change=20`，按训练损失停止。任何触及上限且出现 ConvergenceWarning 的 MLP 拟合都以同种子从头重跑一次，唯一变化是 max_iter=600；若仍警告，保留并披露。不会根据分数选择性延长训练。完整警告、实际迭代、损失及耗时保留。

除 AP，记录 ROC-AUC、precision、recall、F1、accuracy、balanced accuracy、Brier、log loss、混淆计数；开发阶段类别预测统一 `p_yes>=0.5`。没有正预测时 precision/F1 设为0。三折标准差表示折间波动，不是置信区间。

每家族最佳配置仅在完整 train 上重拟合。validation 用于开发诊断和 **D 成员的阈值选择**，不用于再调超参。默认交接模型不做 train+validation 重训，确保阈值与原管线输出对应。D 冻结规则后再统一访问 test；当前任务不生成最终 test 指标。

## 解释边界与后续

随机主实验仅支持历史分布中的记录级泛化；不保证未来批次表现或客户隔离。UCI 明确说明文件按日期排序，可以做行序前向诊断；本次没有进行。未来若补充，应只在既有 train 内做，避免将原 test 混进开发。现有属性的 as-of 可得性仍是模拟假设。

本次不运行15特征情境实验或 duration 演示，不因增加实验数改变核心问题。D 负责最终比较、阈值、PR/ROC、错误分组与置换重要性。特征重要性解释为模型依赖，不能推断因果；年龄等个人属性的差异表现与数据年代/抽样偏差应在最终讨论中说明。

## 2026-10-18 复查补充（主实验完成后）

网页Pro与本地复核没有发现需要替换既有模型或重训的错误。原 `run_protocol.json`、模型、CV记录和validation预测保留。本次新增记录标为事后核验：`results/experiment/post_training_review/manifest.json` 保存当前源码快照和原始数据/划分身份，`posthoc_verification.json` 另外记录原协议、CV清单和已拟合产物的身份。它们不能倒推证明训练当时的源码字节，也不声称存在训练时Git提交。未来调用训练入口会在首次拟合前新建 `source_snapshot/manifest.json`。

重复画像定义统一为“总行数减唯一画像数”：原始11字段含精确pdays为575，替换为联系历史标志后为582，均在train内计算并保留所有行。字面值 `unknown` 正常编码；转换时本拟合折未见的类别才使用全零one-hot块，不删除记录。

全表只用于数据身份、模式及记录一致性检查（包括完整17列重复行的等值检查）。test未用于拟合、目标关联分析、选参或性能分析；不用“从未读取test标签”描述该流程。加权模型输出作为订购分数，不宣称已经校准。MLP无警告只说明满足配置的停止条件，不等于达到全局最优。

## 已核对的主要来源

- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank)：数据版本、字段和按日期排序说明。
- [scikit-learn 1.8 MLPClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPClassifier.html)：sample_weight、优化和停止参数。
- [Average Precision](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)：AP 定义。
- [Pipeline and leakage pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html)：训练与评估隔离。

网页 Pro 使用的在线 stable 文档当时为1.9.1；本地实现另核对并锁定1.8.0，不将其文档版本混作实际运行环境。
