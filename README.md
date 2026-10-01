# COATING_AI

这是一个面向防腐涂层实验的教学型 AI 项目：用高斯过程学习“配方 → 性能”的关系，再用多目标贝叶斯优化推荐下一批实验。

## 在 PyCharm 中开始

项目目录固定为：

```text
/Users/william/Documents/ProjectPython/COATING_AI
```

1. 打开 **PyCharm → Settings → Project → Python Interpreter**。
2. 选择 **Add Interpreter → Add Local Interpreter → Conda → Existing environment**。
3. 解释器选择 `/opt/anaconda3/envs/AI/bin/python`。
4. 打开 PyCharm 底部 Terminal，执行：

```bash
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

`-e` 表示可编辑安装。修改 `src/coating_ai/` 后不需要重新安装项目。
项目配置已把 `src/` 标记为 Sources Root、`tests/` 标记为 Tests Root。

右上角运行配置中可以直接选择：

- `01 - 模拟闭环`：运行一轮20→25组的数学模拟。
- `02 - 预览初始DOE`：只预览20组初始配方，不写文件。

也可以在 Terminal 中运行：

```bash
python -m coating_ai demo --rounds 1
python -m coating_ai init
python -m coating_ai validate --data data/round_00.csv
python -m coating_ai suggest --data data/round_00.csv
```

## 项目结构

```text
COATING_AI/
├── pyproject.toml                 # 项目元数据、依赖、命令入口
├── requirements.txt              # 执行 pip install -e .
├── requirements-lock.txt         # 当前 Mac/AI 环境的精确版本快照
├── README.md                      # 安装与运行说明
├── src/
│   └── coating_ai/
│       ├── __init__.py            # 声明 Python 包
│       ├── __main__.py            # 命令行入口与完整闭环
│       ├── config.py              # 唯一配置源：路径、变量、边界、约束
│       ├── data.py                # 初始设计、CSV检查、模拟实验
│       ├── learning.py            # 高斯过程训练和留一验证
│       └── optimization.py        # qLogNEHVI、Pareto前沿、超体积
├── scripts/
│   └── generate_doe_excel.py      # 生成/预览初始DOE表
├── tests/
│   └── test_workflow.py           # 数据和模型流程测试
├── data/
│   └── initial_design.csv         # 20组初始实验模板
└── outputs/                       # 运行时自动创建，不提交Git
```

## 推荐的代码阅读顺序

1. `config.py`：先认识项目输入、目标、边界和约束。
2. `data.py`：看一组配方如何生成、变成表格、被验证和读取。
3. `learning.py`：看高斯过程如何拟合与验证。
4. `optimization.py`：看模型如何推荐下一批配方。
5. `__main__.py`：最后看上述模块怎样连接成完整循环。

详细的逐文件讲解见 [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)。

## 真实实验闭环

```text
生成初始配方 → 实验测量 → CSV质检 → 留一验证
      ↑                                  ↓
合并新测量 ← 实验下一批 ← qLogNEHVI推荐 ← 训练GP
```

生成模板：

```bash
python -m coating_ai init
```

程序默认拒绝覆盖已有文件。将 `data/initial_design.csv` 另存为 `data/round_00.csv`，完成实验后填写：

- `log_z`：阻抗模量的十进对数，例如 `10^8` 填 `8`。
- `adhesion_score`：反向划格等级，即 `5 - 原等级`。
- `healing_pct`：百分数，60%填 `60`。

验证模型：

```bash
python -m coating_ai validate --data data/round_00.csv
```

推荐下一批：

```bash
python -m coating_ai suggest --data data/round_00.csv
```

填写新批次测量值后合并：

```bash
python -m coating_ai merge \
  --data data/round_00.csv \
  --new data/batch_01_measured.csv \
  --out data/round_01.csv
```

模拟数据与真实数据由 `data_kind` 强制隔离。预测文件也和待填写的实验文件分开，避免把模型预测误当成实验结果。

## 实验前仍需确认

- 五个变量的含量基准和可制造精度。
- 自愈效率的统一测试时间和计算方法。
- PDA厚度与聚合时间在本实验体系中的标定关系。
- 超体积参考点 `[5.5, -0.5, -5.0]` 是否低于实际可接受性能。
- 每个配方的平行样、批次记录和测量噪声保存方式。

本项目是研究工作流，不会证明模拟函数具有化学意义。正式结论必须来自真实实验和独立验证。

