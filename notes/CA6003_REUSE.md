# 从 CA6003 复用的边界

来源：同组 CA6003 本地仓库，调查时 main，HEAD `b4c4ff1`。该仓库只读；原课程材料未修改。

原样复制：`data/raw/bank-full.csv`、`bank-names.txt`、`data/source/provenance.json`、`data/splits/split_indices.json`、`split_manifest.csv`、`config/dataset_lock.json`。许可说明在保留引用基础上改为描述实际 CA6001 打包范围。

沿用原60/20/20行级分层划分和11个原始来源字段。CA6001 新建数据读取、预处理、EDA、训练和交接代码；没有迁移 CA6003 其他课程报告、PPT 或 `.venv`。

实质微调：`pdays` 不再以精确天数进入模型，只派生是否有过联系；理由是无法验证精确间隔对应启动前时点。各算法的 one-hot/缩放学习在训练折内完成。四算法共用fold-local平衡权重实验。研究问题收窄为离线模拟，不承诺未来部署效果。

原始数据在2026-10-12由6003获取；2026-10-18核对官方 UCI 页并在6001校验原始字节身份。没有声称本次重新从UCI下载，也没有混用bank-additional版本。
