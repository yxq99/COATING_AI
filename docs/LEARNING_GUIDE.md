# 代码学习指南

## 1. 先建立一张地图

整个项目只有两条主线：

- **主动学习闭环**负责重复执行“选择配方、做实验、更新数据、重新训练”。
- **贝叶斯优化**负责在每轮闭环中决定下一批最值得实验的配方。

运行 `python -m coating_ai ...` 时，Python会进入 `src/coating_ai/__main__.py` 的 `main()`。

## 2. config.py：所有模块共同使用的语言

关键对象：

- `PROJECT_ROOT`、`DATA_DIR`、`OUTPUT_DIR`：由文件位置推导，移动项目后仍然有效。
- `FEATURES`：5个输入列名。
- `TARGETS`：3个最大化目标列名。
- `BOUNDS`：每个输入的上下界。
- `REF_POINT`：计算超体积时固定使用的参考点。
- `feasible(x)`：批量检查配方是否满足边界和两条物理约束。
- `constraints()`：把同样的约束转换为BoTorch优化器所需的线性形式。

学习重点：配置只保留一份。采样、读数据和优化都引用这里，修改边界时不会遗漏另一份脚本。

## 3. data.py：数据从哪里来，能不能用

主要函数：

- `initial_points(n, seed)`：Sobol生成候选点，再过滤不可行配方。
- `frame(x, y)`：把PyTorch张量变成带列名的表格。
- `save(df, path)`：使用独占创建，防止覆盖实验文件。
- `load(path)`：检查列、编号、数据来源、空值、取值范围和物理约束，再转换成张量。
- `simulate(x)`：仅供学习程序流程的数学函数。

数据形状：

```text
x: [样本数, 5]  例如 [20, 5]
y: [样本数, 3]  例如 [20, 3]
```

建议在PyCharm中给 `initial_points()` 和 `load()` 打断点，观察DataFrame和Tensor之间的转换。

## 4. learning.py：模型怎样学习

`fit(x, y)`完成四件事：

1. 为三个输出构造独立的Matérn 5/2核。
2. 把五维输入归一化。
3. 把三个输出分别标准化。
4. 最大化边缘似然，学习核长度尺度、信号强度和噪声。

模型返回的后验包含：

- `posterior.mean`：配方的预测性能。
- `posterior.variance`：模型对潜在函数预测的不确定程度。

`cross_validate(x, y)`执行留一验证。每次移除一个配方、重新训练、预测被移除配方，最后计算MAE、RMSE和R²。这比查看训练集拟合更能反映小样本泛化能力。

## 5. optimization.py：下一组实验怎样选

`suggest(model, x, batch_size)`：

1. 用已有输入 `x` 作为 `X_baseline`。
2. 构建 `qLogNoisyExpectedHypervolumeImprovement`。
3. 用128个Sobol QMC样本近似采集函数期望。
4. 在边界和物理约束内优化采集函数。
5. 返回最多5个新配方，并再次检查可行性。

qLogNEHVI会同时考虑预测性能、不确定性、多目标Pareto改进和批次内候选点的联合价值。

`pareto_hv(y)`找出当前不被其他配方支配的结果，并计算它们相对固定参考点的观测超体积。

## 6. __main__.py：把模块连接起来

四个真实工作命令：

- `init`：生成初始空白实验表。
- `validate`：读取真实数据并执行留一验证。
- `suggest`：训练模型并生成下一批候选表与预测参考表。
- `merge`：验证并合并上一轮数据和新测量数据。

`demo`使用相同的训练和优化模块，但由 `simulate()` 代替真实实验。它只用于确认软件闭环。

## 7. 推荐的调试练习

在以下位置设置断点：

1. `__main__.py` 中 `args = parser.parse_args()`：观察命令参数。
2. `data.py` 中 `return torch.cat(blocks)[:n]`：查看20×5输入。
3. `learning.py` 中 `return model`：展开模型和核参数。
4. `optimization.py` 中 `candidates = candidates.detach()`：查看新配方。
5. `__main__.py` 中 `x, y = torch.cat(...)`：观察一轮后20组变成25组。

## 8. 修改代码时遵循的依赖方向

```text
config
  ↓
data      learning      optimization
   \         |          /
        __main__
```

底层模块不要反过来导入 `__main__`。实验变量优先放在 `config.py`，数据规则放在 `data.py`，模型只放在 `learning.py`，采样决策只放在 `optimization.py`。

每次修改后运行：

```bash
python -m unittest discover -s tests -v
python -m coating_ai demo --rounds 1
```

测试通过说明已覆盖的程序规则没有被破坏；它不代表科研假设或真实材料性能已经得到验证。

