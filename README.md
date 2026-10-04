# COATING_AI

这是一个由设置文件驱动的实验 AI 项目。它用高斯过程学习“输入变量 → 目标变量”的关系，并提供两种下一批实验推荐模式：

- `target`：给定期望目标值、区间或阈值，反向推荐最合适的输入变量。
- `pareto`：在多个互相竞争的目标之间寻找 Pareto 前沿。

“目标变量”比“输出变量”更适合本项目：它既表示实验测量结果，也表示优化希望达到的目标。

## 1. 在 PyCharm 中准备环境

项目目录：

```text
/Users/william/Documents/ProjectPython/COATING_AI
```

在终端执行：

```bash
conda create -n AI python=3.11 -y
conda activate AI
cd /Users/william/Documents/ProjectPython/COATING_AI
python -m pip install -e .
python -m pip check
pytest -q
```

然后在 PyCharm 的 **Settings → Project → Python Interpreter** 中选择 Conda 环境 `AI` 的 Python。可编辑安装后，修改 `src/coating_ai/` 无需重复安装。

## 2. 先修改设置文件

所有项目级设置都集中在 [`settings/project.yaml`](settings/project.yaml)：

- 输入变量：名称、`continuous`/`integer` 类型、开关、上下界。
- 目标变量：名称、开关、Pareto 最大化/最小化方向、指定目标条件。
- 约束：总开关、每条线性约束的开关、系数和阈值。
- 推荐：`target`/`pareto` 模式、每批数量、候选池大小等。

切换模式只需修改：

```yaml
recommendation:
  mode: target   # 或 pareto
```

输入上下界有两种方式：

```yaml
bounds:
  enabled: true
  lower: 0
  upper: 10
```

或者关闭显式边界，改用训练数据的实际范围：

```yaml
bounds:
  enabled: false
  observed_margin: 0.10  # 在实际范围两端各扩展10%
```

注意：第一次生成初始实验时还没有训练数据，因此 `init` 所用的全部输入变量必须暂时启用显式上下界。获得数据后可关闭边界，让推荐范围自动从数据计算。

## 3. 完整使用流程

### 生成初始实验表

```bash
python -m coating_ai init
```

程序生成 `data/initial_design.csv`，其中目标变量保持空白。先另存一份工作数据，例如 `data/round_00.csv`，完成真实实验后填写目标变量。

也可以生成 Excel：

```bash
python scripts/generate_doe_excel.py --output data/initial_design.xlsx
```

### 检查数据和模型效果

```bash
python -m coating_ai validate --data data/round_00.csv
```

它会检查列、空值、整数变量、边界、约束和数据来源，再执行留一交叉验证。

### 训练并保存模型

```bash
python -m coating_ai train --data data/round_00.csv
```

命令会打印并生成类似 `outputs/train_20261002_...` 的目录。模型、训练数据快照、设置快照和验证指标会一起保存。快速调试时可添加 `--skip-cv`，正式判断模型质量时不要跳过验证。

### 推荐下一批实验

先在 `settings/project.yaml` 中设定推荐模式和目标，再执行：

```bash
python -m coating_ai recommend --run outputs/train_20261002_...
```

输出目录中两个文件用途不同：

- `candidates_to_measure.csv`：下一批待做实验；目标变量为空。
- `predictions_not_measurements.csv`：模型预测和不确定性，仅供判断，不得当作实验值回填。

目标设置可使用：

- `target`：接近期望值 `value`。
- `range`：进入 `[lower, upper]`。
- `at_least` / `at_most`：达到单边阈值 `value`。
- `maximize` / `minimize`：尽可能大/小。
- `ignore`：本次反向推荐忽略该目标。

多个目标通过 `weight` 调整相对重要性，`tolerance` 定义可接受误差尺度。

### 合并新实验

填写新推荐批次的真实目标值后：

```bash
python -m coating_ai merge \
  --data data/round_00.csv \
  --new data/batch_01_measured.csv \
  --out data/round_01.csv
```

然后用合并文件重新验证、训练和推荐，形成主动学习闭环。

### 运行纯模拟演示

```bash
python -m coating_ai demo --rounds 1
```

演示会读取当前设置中的模式，但模拟目标不具有真实材料意义。`data_kind` 会强制隔离 `real` 与 `demo`，防止误混。

## 4. 项目结构

```text
COATING_AI/
├── settings/project.yaml          # 用户主要修改的变量、目标、边界、约束和模式
├── src/coating_ai/
│   ├── config.py                  # 读取和验证设置，解析动态边界与约束
│   ├── data.py                    # 初始设计、候选池、数据质检、模拟函数
│   ├── learning.py                # 高斯过程训练、预测和交叉验证
│   ├── optimization.py            # Pareto优化与指定目标反向推荐
│   ├── artifacts.py               # 保存和恢复训练模型及快照
│   └── __main__.py                # 将各模块连接为命令行工作流
├── scripts/generate_doe_excel.py  # 可选的Excel初始实验表
├── tests/test_workflow.py         # 自动测试
├── data/initial_design.csv        # 可公开的空白示例模板
├── outputs/                       # 本地运行结果；Git忽略
└── docs/LEARNING_GUIDE.md         # 代码学习路线
```

真实实验数据、模型文件、输出结果、PyCharm本地设置和调试脚本 `src/Test.py` 都不会提交到 Git。Git 只管理可复现项目所需的源代码、设置示例、文档、测试和空白模板。
