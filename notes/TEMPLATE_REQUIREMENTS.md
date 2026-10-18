# CA6001 报告模板要求与页数建议

## 模板明确要求

- 正文最多 **10 页**，封面不计入这 10 页。模板将 References 列为第 10 节，因此参考文献页也计入正文页数；模板没有说明附录可以突破页数上限。
- 保留以下 10 个章节：Introduction、Problem Definition、Dataset Description、Algorithm Selection and Rationale、Methodology、Implementation、Results and Evaluation、Discussion & Insights、Conclusion、References。
- Introduction 说明问题领域、动机、目标和范围；Problem Definition 给出正式问题、任务类型、约束、假设及特殊考虑。
- Dataset Description 说明来源、关键变量、结构、规模、格式、预处理，并可加入精选探索分析表或图。
- Algorithm Selection and Rationale 简述理论基础、优缺点、对本任务的适用性和备选算法。
- Methodology 说明预处理、模型结构或算法逻辑、参数与超参数、评估策略、工具和库；可加入流程图、模型图或代码片段。
- Implementation 概括核心函数、关键实现决定、主要算法代码和重要输出；模板明确不要求粘贴完整代码，并注明 notebook 可选。
- Results and Evaluation 展示并解释与任务相符的指标，可包含表格、图、算法比较及错误分析。
- Discussion & Insights 讨论效果、困难、结果含义、局限和后续改进；Conclusion 总结主要结果、发现及算法选择和实现反思。
- References 使用 APA 格式，列出实际使用的数据集、论文、软件和工具。

模板**没有**规定各章节评分比例、必须达到的准确率或其他性能门槛，也没有指定必须使用的数据集版本。不要把建议页数、项目内部验收标准或模型分数写成课程评分规则。

## 项目建议页数（合计 10 页，封面另计）

下表是版面安排建议，不是模板的硬性比例。优先保留结果解释和可读性；图表与正文共用页数。

| 模板章节 | 建议页数 | 主要材料负责人 |
| --- | ---: | --- |
| 1. Introduction | 0.5 | A 提供问题背景、目标和范围；D 统一最终表述 |
| 2. Problem Definition | 0.5 | A 提供标签和预测任务定义；D 校准解释边界 |
| 3. Dataset Description | 1.0 | A 提供来源、结构、字段政策、审计和 train EDA |
| 4. Algorithm Selection and Rationale | 1.0 | B/C 提供四类模型的简要原理、适用性、优缺点和备选项 |
| 5. Methodology | 1.5 | A 提供数据处理；B/C 提供 CV 与训练流程；D 提供冻结阈值和 test 协议 |
| 6. Implementation | 1.0 | B/C 提供关键实现、运行结果和短代码示例 |
| 7. Results and Evaluation | 2.0 | D 汇总最终模型比较、test 指标、图表和错误分析 |
| 8. Discussion & Insights | 0.8 | D 结合全部结果讨论发现和限制 |
| 9. Conclusion | 0.5 | D 归纳最终发现 |
| 10. References | 1.2 | 全体提供实际来源；D 统一 APA 格式 |
| **合计** | **10.0** | 封面另计 |

## A/B/C/D 交接边界

- **A：数据与探索分析。** 维护来源和完整性说明、字段及预测时点政策、固定 split 行号、数据审计与预处理接口。EDA 的标签关系和分布图只使用 train。给正文提供精选图表，不提前解释模型结果。
- **B：Logistic Regression 与 Decision Tree。** 维护各自固定搜索配置的训练 CV 结果、参数、运行警告和完整 train 拟合管线，并导出 validation 概率及 row_id。
- **C：Random Forest、MLP 与 Dummy 基线。** 维护相同口径的 CV、最佳 train 管线和 validation 概率；另保留 MLP 训练损失、迭代和收敛/补跑信息。
- **D：冻结后的最终评估与整合。** 在不改变既定模型身份的前提下选择并冻结四个真实模型的阈值及评估规则，使用一次性入口解锁 test；完成最终比较、PR/ROC 展示、混淆矩阵、错误分组和置换重要性，并整合 Results、Discussion 与 Conclusion。不得将 test 结果用于重新选择特征、模型或阈值。

正文可以精简呈现；逐折结果、完整日志、管线和详细诊断保留为可复现交接材料。阈值、最终 test 结果、错误分析和特征重要性属于 D 的工作，不由 A/B/C 预先生成或代写结论。
