# COATING_AI

当前项目围绕“早期电化学时间信息辅助完整涂层耐久预测与配方设计”。两个配方输入为连续的复合填料添加量 `hybrid_wt_pct` 和 LDH/MXene 干基质量比 `ldh_mxene_ratio`。

## 数据与日常使用

所有必要输入、原始EIS复制件、未来测量空位和有效设置都在 `data/`。

先阅读 [数据填写与完整流程](data/README.md)，日常填写 `data/ML_data.xlsx`，然后：

```bash
conda activate AI
python -m coating_ai prepare-eis
python -m coating_ai train --data data/design_training.csv
python -m coating_ai recommend --run outputs/实际训练目录 --out-data data/round_02
```

早期预测使用单独的模型职责，但仍是同一套数据入口：

```bash
python -m coating_ai train --settings data/early_model.yaml --data data/early_training.csv
python -m coating_ai prepare-eis --new-data data/round_01
python -m coating_ai predict --settings data/early_model.yaml --run outputs/实际早期模型目录 --data data/early_prediction_inputs.csv
```

需将“实际训练目录”替换成终端输出。新实验未测时保留空白；只有早期输入完整才有待预测行，后期完整才进入再训练。

本次已经训练检查两个模型并生成 `data/round_01/`。可直接填写这一轮，不必重做；模型目录记录在 `data/current_runs.json`。上面的推荐示例使用round_02以避免覆盖现有记录。

项目不再要求 `data_kind` 或区分 real/demo，旧来源列仅作为额外元数据。原数学demo命令已撤下，主入口只有prepare-eis/init/validate/train/predict/recommend/merge。

默认设置已改为 `data/project.yaml`，由主工作簿“参数”重建；`settings/project.yaml` 保留为历史通用配置参考，不是当前默认。既有通用整数变量能力仍保留，但本研究两配方变量均为连续量。

## 当前数据和功能范围

工作簿六条曲线来自已有其他涂层的EIS，配方输入是流程演练假设；来源、重复谱和时间/幅值问题保留可查。不能当作MXene新体系实测结果或最优配方证据。

已经接通数据整理、配方与早期模型训练、分组验证、预测、连续配方推荐以及空白测量回填接口。现有target评分是目标偏差＋潜在不确定性惩罚，不是经过校准的可靠达标概率；独立重复模板留待真实测量。

数据和模型默认不推送Git，旧学习文档可继续参考理论；其历史接口说明以本README与data/README为准。

## 环境与测试

```bash
conda activate AI
python -m pip install -e .
python -m unittest discover -s tests -v
```

源码在 `src/coating_ai/`。新增时间数据整理在 `eis_workflow.py`，所有流程仍遵循未测值不作为训练标签、输入/目标检查、同源数据分组和不覆盖已有实验结果的原则。
