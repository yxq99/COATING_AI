# COATING_AI 全部代码学习说明

编写日期：2026-10-04。对应目前已恢复的通用项目，代码基准为 `25194ea`。

项目目录：`/Users/william/Documents/ProjectPython/COATING_AI`。

本文解释当前主项目的全部 Python 源文件、配置、测试及安装文件，不包含已归档的 GNDs 代码。文档中的 A、B、C 和 X、Y、Z 是示例名称，不代表特定材料参数。本文不改变任何算法或设置。

## 阅读导航

1. 项目在做什么：机器学习、主动学习与贝叶斯优化的关系。
2. 文件结构与调用关系。
3. Python 语法、数据类型和张量形状。
4. `settings/project.yaml`：用户设置接口。
5. `config.py`：读取设置、计算边界、检查约束。
6. `data.py`：生成实验、读写和检查数据。
7. `learning.py`：高斯过程、预测与交叉验证。
8. `optimization.py`：Pareto 与指定目标反向推荐。
9. `artifacts.py`：保存和恢复模型。
10. `__main__.py`：六个命令如何串联整个项目。
11. 辅助脚本、包声明和调试文件。
12. `tests/test_workflow.py`：全部测试的意义。
13. 安装、依赖与 Git 文件。
14. 从第一次实验到再次训练的完整操作。
15. 当前版本限制、调试练习与修改位置。

建议第一次阅读先看第 1、2、4、14 节，掌握使用流程；第二次按第 5—10 节阅读代码；最后用第 12、15 节练习修改与验证。

## 1. 项目在做什么

### 1.1 学习输入与目标之间的关系

假设你可以控制输入变量 A、B、C，实验后测得目标变量 X、Y、Z。

```text
输入：A、B、C ──真实实验──> 目标：X、Y、Z
        │                       │
        └────历史数据训练模型────┘
                     │
             新输入的预测值和不确定性
```

模型学习的是 `输入 → 目标` 的正向关系。所谓反向推荐，不是再训练一个 `目标 → 输入` 的逆模型，而是在可行输入空间里生成候选，用正向模型预测，再寻找更符合要求的候选。

输入到目标可能是多对一，所以反向问题一般没有唯一解，也不一定存在精确解。

### 1.2 三个容易混淆的概念

| 概念 | 在本项目中的含义 |
|---|---|
| 机器学习 | 用已完成实验的数据训练高斯过程模型 |
| 主动学习 | 推荐新实验，测量，再加入数据重训的外层循环 |
| 贝叶斯优化 | 利用模型预测分布选择更有价值的下一批实验；Pareto 分支使用此方法 |

当前项目不是深度神经网络，没有卷积层、隐藏层数量或神经网络训练轮数等设置。

项目提供两种推荐方式：

- `pareto`：根据多个目标的最大化/最小化方向，寻找有望改善 Pareto 前沿的实验。
- `target`：根据用户指定的值、区间或阈值，在候选池里按目标偏差和风险排序。

这里的 `target` 是有限候选池上的目标匹配搜索，不是 Pareto 分支的 qLogNEHVI 算法。它倾向惩罚不确定性，没有一个独立的“优先探索最高不确定性”模式。

## 2. 文件结构与调用关系

### 2.1 当前需要学习的文件

```text
COATING_AI/
├── settings/project.yaml          用户设置
├── src/
│   ├── coating_ai/
│   │   ├── __init__.py            包说明和版本
│   │   ├── config.py              配置对象、边界、约束、目标方向
│   │   ├── data.py                候选生成、实验表、数据检查、模拟
│   │   ├── learning.py            模型构建、拟合、预测、留一验证
│   │   ├── optimization.py        两种推荐策略
│   │   ├── artifacts.py           运行目录、模型保存和恢复
│   │   └── __main__.py            命令入口
│   └── Test.py                    本地调试草稿；目前为空
├── scripts/
│   ├── __init__.py                辅助脚本包说明
│   └── generate_doe_excel.py      生成初始实验 Excel
├── tests/test_workflow.py         自动测试
├── data/initial_design.csv        已生成的空白目标示例表
├── outputs/                       运行结果，不是代码
├── pyproject.toml                 包、依赖与命令入口
├── requirements.txt              可编辑安装入口
├── requirements-lock.txt         环境版本记录
├── .gitignore                    Git 忽略规则
└── docs/                         学习文档
```

### 2.2 调用关系

```text
settings/project.yaml
         │
         ▼
      config.py ─────────────── 所有核心模块读取配置对象
         │
         ▼
       data.py ── x、y ──> learning.py
         │                   │
         │                 模型
         │                   ▼
         └──────────> optimization.py ──> 下一批输入
                              │
                        artifacts.py 保存/恢复

__main__.py：根据 init/validate/train/recommend/merge/demo 调用以上模块
generate_doe_excel.py：复用 config.py 与 data.py，不另写实验设计算法
```

其中 `config.py` 是规则层，`data.py` 是数据层，`learning.py` 是模型层，`optimization.py` 是决策层，`artifacts.py` 是存档层，`__main__.py` 是流程层。

## 3. 阅读代码前需要认识的语法与数据

### 3.1 常用 Python 写法

| 写法 | 含义与项目中的例子 |
|---|---|
| `from .config import ...` | 从当前包内的 `config.py` 导入对象；点表示包内相对导入 |
| `if __name__ == "__main__":` | 文件作为程序入口执行时调用 `main()`，普通导入时不调用 |
| `Path(...)` | 路径对象；用 `/` 拼接子目录，避免手工拼字符串 |
| `@dataclass(frozen=True)` | 自动生成保存字段的类，并禁止直接重新赋值字段 |
| `@property` | 把方法当作属性读取，例如 `config.input_names` |
| `list[str]` | 列表中保存字符串的类型提示，不是运行时强制转换 |
| `float \| None` | 允许数值或空值，如未填写的参考点 |
| `*` 出现在参数列表中 | 后面的参数必须按名字传入，例如 `allow_demo=True` |
| `*config.input_names` | 展开列表，将多个变量名插入另一个列表 |
| 列表推导式 | 例如 `[item.name for item in ... if item.enabled]` 筛选启用变量 |
| `mask` | 布尔数组；`x[mask]` 只保留对应位置为真的行 |
| `raise ValueError(...)` | 停止当前流程，并说明数据或设置错误 |
| `with ... as ...` | 在使用结束后自动关闭文件或释放临时资源 |
| `n or default` | `n` 为真值时使用它，否则使用默认值；与只判断 `None` 不完全相同 |

`frozen=True` 是浅层保护：不能直接写 `config.name = ...`，但其中的字典并不会因此变成完全不可修改对象。学习时不要把它理解为深层冻结。

`from __future__ import annotations` 让类型注解采用延后求值等兼容行为；它不负责训练算法。

### 3.2 表格和张量

`pandas.DataFrame` 是有行列名的表格，适合 CSV 读写。`torch.Tensor` 是适合数值运算和模型计算的数组。

设有 n 条实验、d 个启用输入、m 个启用目标：

| 对象 | 形状 | 含义 |
|---|---|---|
| `x` | `[n, d]` | 每行一组实验输入 |
| `y` | `[n, m]` | 每行对应实验目标 |
| `bounds` | `[2, d]` | 第一行下界，第二行上界 |
| `mean` | `[k, m]` | k 个候选的目标预测均值 |
| `std` | `[k, m]` | k 个候选的目标潜在函数标准差 |
| `mask` | `[n]` 或 `[k]` | 哪些行通过检查或属于 Pareto 前沿 |

当前例子的 `d=3、m=3`，但这些维数由配置决定，不写死在模型里。

张量默认使用 `torch.double`，即双精度数值。常见转换：

```python
torch.tensor(table[config.input_names].to_numpy(dtype=float), dtype=DTYPE)
```

含义依次是：按配置取输入列 → 转成浮点数组 → 转成双精度张量。

反向转换中：`detach()` 脱离梯度图，`cpu()` 移到 CPU，`numpy()` 转成数组，最后交给 pandas 写表。

## 4. `settings/project.yaml`：用户设置接口

源文件：[project.yaml](/Users/william/Documents/ProjectPython/COATING_AI/settings/project.yaml)。

### 4.1 `project`

- `name`：项目名字，主要用于保存说明，不决定算法。
- `initial_design_size`：初始实验数量，当前为 20。

### 4.2 `input_variables`

每个名称对应一个输入条件。当前 A、B 为连续数值，C 为整数；三者范围均为 0—10。

| 字段 | 含义 |
|---|---|
| `type` | `continuous` 或 `integer` |
| `enabled` | 是否进入当前模型；关闭会改变变量结构 |
| `bounds.enabled` | 是否使用明确指定的上下界 |
| `bounds.lower/upper` | 显式下界、上界；需要下界小于上界 |
| `bounds.observed_margin` | 关闭显式边界时，相对实际数据跨度向两端扩展的比例 |

例如实际数据范围为 2—8，扩展比例 0.1，则跨度为 6，推荐范围为 `1.4—8.6`。

关闭边界不等于无界搜索。`init` 没有历史数据，所以所有启用输入需要显式边界。

YAML 的缩进有意义；布尔值写 `true/false`，不要写成带引号的字符串 `"false"`。当前代码有些开关使用 `bool(...)` 转换，非空字符串并不会变成假。

### 4.3 `target_variables`

目标是实验后需要填写的数据。`enabled` 决定是否训练该目标。

`pareto_direction` 只定义 Pareto 计算方向，支持 `maximize` 和 `minimize`。`reference_value` 是 Pareto 超体积参考点的该目标坐标；留空时自动计算。

`goal` 是指定目标推荐的要求，与 `pareto_direction` 是不同接口。例如某目标的 Pareto 方向可以是越大越好，而此次 target 请求只想接近某个指定值。

| `goal.mode` | 含义 | 需要填写 |
|---|---|---|
| `target` | 接近指定值 | `value` |
| `range` | 落在指定区间 | `lower、upper` |
| `at_least` | 不低于阈值 | `value` |
| `at_most` | 不高于阈值 | `value` |
| `maximize` | 在候选池内尽量大 | 无额外值 |
| `minimize` | 在候选池内尽量小 | 无额外值 |
| `ignore` | 不参与此次 target 打分 | 无额外值 |

`weight` 越大，该目标对总分越重要；必须大于 0。

`tolerance` 是误差和不确定性的缩放尺度，必须大于 0。当前代码并不把它变成硬性达标约束。例如目标为 0.8、容差为 0.25，预测为 0.9 时该目标归一化偏差为 `0.1/0.25=0.4`，而不是在容差内就一律记 0 分。

未填 `tolerance` 时通常使用训练目标的标准差；`range` 模式的默认缩放尺度为区间宽度。

### 4.4 `constraints`

当前默认约束为 `A+B+C <= 20`。

- `constraints.enabled`：约束总开关。
- `rules`：线性约束列表。
- 每条规则的 `enabled`：单条开关。
- `coefficients`：变量系数。
- `operator`：只支持 `<=` 或 `>=`。
- `rhs`：不等式右端数值。

例如 `2A-B <= 5`，应设置 `coefficients: {A: 2, B: -1}`、`operator: <=`、`rhs: 5`。

这不是非线性约束接口。即使关闭某条约束，加载设置时仍会校验其结构和引用的变量，所以关闭输入变量后还需要同步整理相关规则。

### 4.5 `recommendation`

| 字段 | 当前值 | 用途 |
|---|---|---|
| `mode` | `target` | `target` 或 `pareto` |
| `batch_size` | 5 | 每批推荐几组实验 |
| `candidate_pool_size` | 2048 | 每次搜索的可行候选数量 |
| `seed` | 42 | Sobol 候选的随机种子等 |
| `risk_aversion` | 0.25 | target 模式的不确定性惩罚系数 |
| `diversity_min_distance` | 0.05 | target 模式在归一化输入空间中的批内距离要求 |
| `auto_reference_margin` | 0.10 | 自动 Pareto 参考点向最差观测值外扩的比例 |

`risk_aversion` 和 `diversity_min_distance` 不用于 Pareto 分支的目标打分。Pareto 分支通过采集函数本身处理预测不确定性及批次选择。

## 5. `config.py`：配置与可行域

源文件：[config.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/config.py)。

### 5.1 路径和常量

`PROJECT_ROOT = Path(__file__).resolve().parents[2]` 从当前文件位置定位项目根目录：`config.py → coating_ai → src → COATING_AI`。`parents[0]` 是 `coating_ai`，`parents[2]` 才是项目根目录。

`DATA_DIR、OUTPUT_DIR、DEFAULT_SETTINGS_PATH` 分别定位 data、outputs 和默认 YAML；因此不需要在各模块写死完整路径。

`DTYPE = torch.double` 统一数值精度。`RESERVED_COLUMNS` 保留 `sample_id、data_kind`，防止用户变量与系统列重名。

### 5.2 七个配置类

| 类 | 保存的内容 |
|---|---|
| `BoundsSpec` | 边界开关、上下界、观测扩展比例 |
| `InputVariable` | 输入名字、类型、开关和边界 |
| `GoalSpec` | 目标条件的模式、权重、值、区间、容差与开关 |
| `TargetVariable` | 目标名字、开关、Pareto 方向、参考值和 goal |
| `LinearConstraint` | 约束名字、开关、系数字典、运算符和阈值 |
| `RecommendationSpec` | 推荐模式、批量、候选数量、种子和风险参数 |
| `ProjectConfig` | 上述对象的总配置，以及源 YAML 路径 |

`ProjectConfig` 的五个属性：

- `input_names`：启用输入的名字，顺序与 YAML 一致。
- `target_names`：启用目标的名字。
- `active_inputs`：启用输入对象，而不只是名字。
- `active_targets`：启用目标对象。
- `integer_indices`：整数输入在当前启用输入列表中的位置；例如 C 是第 3 列，索引为 2。

代码取 CSV 列、构造张量和恢复模型都依赖这个顺序。

### 5.3 三个内部辅助函数

`_mapping(value, label)` 检查某项是否为字典；不符合时用字段名称解释错误。

`_number(value, label, optional=False)` 检查有限数值，明确排除布尔值。`optional=True` 时允许 `None`；返回浮点数或空值。

`_goal(name, raw)` 解析一项目标条件：检查模式是否合法、权重是否为正、指定值是否存在、区间上下界是否正确、容差是否为正，最后生成 `GoalSpec`。这里的 `_` 表示内部辅助函数，不是 Python 的强制私有机制。

### 5.4 `load_project_config(path)`

流程是：定位 YAML → `yaml.safe_load` → 检查根节点 → 逐项建立配置对象 → 返回 `ProjectConfig`。

主要检查包括：

1. 输入仅支持连续/整数，名字不能占用系统保留列。
2. 显式边界必须有限且 `lower < upper`；整数输入的边界也必须是整数。
3. 输入和目标不能重名；至少启用一个输入和一个目标。
4. 目标方向只能最大化/最小化；goal 模式、值、容差合法。
5. 约束引用的变量必须存在且启用，至少一个非零系数，运算符合法。
6. 推荐模式、候选数量、批量、风险及距离参数合法。
7. target 模式至少有一个启用且不是 ignore 的 goal。
8. 初始实验数量必须为正。

第 7 项在读取配置时执行，所以当前版本即使只是训练，也需要 target 设置已有有效 goal；不能等到推荐时才第一次补齐。

### 5.5 `resolve_search_bounds(config, observed_x=None)`

对每个启用输入计算上下界，返回 `[2,d]` 张量。

显式边界开启：直接读 lower/upper。关闭：从该列历史输入找最小值、最大值，根据跨度和扩展比例扩展；若所有观测值相同，改用 `max(abs(最小值),1)` 作为扩展基准。

当前版本用 `delta = max(observed_margin * base, 1e-9)`。连续变量因此至少扩展一个极小量；整数变量随后用 `floor/ceil` 向外取整。特别注意：整数观测端点恰好是整数、margin 为 0 时，也可能两端各多出一个整数。这是恢复版本的实际行为，不是“零扩展必然严格保持整数范围”。

没有观测数据而关闭边界，会报错。这也是初始实验阶段需要明确范围的原因。

### 5.6 `feasible(config, x, bounds)`

返回每行是否可行的布尔 mask。检查顺序：边界 → 整数要求 → 启用线性约束。

单个输入向量会先加一维，变成一行表。比较时允许 `1e-6` 数值误差，避免边界上的浮点舍入造成误判。

整数检查不是先偷偷把训练值四舍五入，而是检查其与最近整数是否足够接近。候选生成会先取整，但原实验数据不能任意填小数。

### 5.7 `target_signs(config)`

BoTorch 的 Pareto 计算以最大化为统一方向。最大化目标返回 +1，最小化目标返回 -1。通过 `y * signs`，把最小化目标取负后也变成“越大越好”。例如最小化成本 3、5，转成 -3、-5 后，-3 更优。

### 5.8 `resolve_reference_point(config, y)`

参考点是超体积计算的基准，并不是用户希望达到的 target。

用户填了 `reference_value`：按目标方向变换后使用。没填：在统一最大化空间中，取该目标最差观测值，再向更差方向减去 margin×跨度；跨度为 0 时使用备用尺度。

参考点会影响超体积与 Pareto 推荐。当前代码没有额外校验所有人工参考值是否都足够差，填写时需要自己保证物理和数值含义正确。

### 5.9 `schema_signature(config)`

返回输入名、输入类型、目标名三个列表，保存为模型的结构签名。推荐加载时需要完全匹配，包括顺序。

签名不包含目标值、方向、约束或边界，因此这些不是结构不匹配的直接原因；但修改边界/约束可能使历史数据无法通过当前数据检查，详见第 9、15 节。

## 6. `data.py`：实验数据

源文件：[data.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/data.py)。

### 6.1 `_stable_unique(values)`

把每行转换为 tuple，用 set 判断是否见过，再按原顺序保存第一次出现的行索引。返回去重后的张量。

为什么保持原顺序？如果先按数值排序再取前几个点，初始实验可能偏向搜索空间某一侧。

这是精确数值去重，不是“足够接近就算重复”。整数候选取整后更容易出现重复，因此需要它。

### 6.2 `candidate_pool(config, bounds, size, seed)`

1. 维数等于启用输入数量。
2. 建立加扰的 Sobol 序列，产生 `[0,1]` 内的多维点。
3. 用 `lower + unit * (upper-lower)` 映射到真实输入范围。
4. 整数输入列取整。
5. 使用 `feasible` 过滤边界、整数与线性约束。
6. 合并已有块并按顺序去重，数量达到 size 时返回前 size 个。
7. 最多尝试 100 次；每次至少抽 256 个，通常抽 `size*2` 个。

Sobol 的用途是较均匀地覆盖空间，不是在这里训练模型。`seed` 决定加扰序列的可复现性。

如果可行空间过小，或全部输入为整数、唯一组合数小于候选池要求，程序会报候选不足。增加抽样次数不能突破有限组合数。

### 6.3 `initial_points(config, n=None, seed=None)`

调用 `resolve_search_bounds` 和 `candidate_pool` 生成第一轮实验输入。数量默认用 `initial_design_size`，种子默认用 recommendation.seed。

这里只生成输入，没有目标，也没有训练。CLI 中 `--n` 可覆盖数量。

### 6.4 `frame(config, x, y=None, start=1, kind="real")`

把张量转为实验表。整数列转成整数显示；插入样本编号和来源；最后添加目标列。

- 有 y：填写目标值。
- 没有 y：目标列全部为 NaN，导出 CSV 后显示为空。
- 样本编号默认为 `real_0001、real_0002…` 或 `demo_0001…`。
- `start` 可控制起始编号。

`kind="real"` 是来源标记，不意味着空目标表已经完成实验。只有真实测量后填写完整，才能训练。

### 6.5 `save_csv(table, path)`

先创建父目录，再用 `open("x")` 独占创建新文件。已经存在的文件会报 `FileExistsError`，不会覆盖。

编码为 `utf-8-sig`，方便常见表格软件识别中文；`index=False` 防止额外写出 DataFrame 行索引。

当前已有 `data/initial_design.csv`，所以再执行不带输出路径的 init 会碰到防覆盖保护。应使用新文件名，而不是为了运行命令直接删除旧记录。

### 6.6 `load_data(config, path, allow_demo=False, min_rows=6)`

返回 `(table, x, y)`，并依次检查：

1. 必须包含 `sample_id、data_kind` 及全部启用输入、目标列。
2. 行数至少 min_rows，通常为 6。
3. 样本编号不能为空或重复。
4. 来源必须全为 real；若 `allow_demo=True`，则必须全为 demo。
5. 输入与目标能转成数值，并且没有空白、无穷值等。
6. 所有输入满足当前边界、整数规则和启用约束。

当前版本不会给只有输入/目标列的 CSV 自动补编号和来源，也不会自动清理列名两边的空格。列名必须与设置一致；额外列可以存在，但不会作为模型输入或目标。

“至少 6 行”只是程序门槛，不代表任何问题用 6 条数据就能达到可靠精度。

### 6.7 `simulate(config, x, bounds, noise=0.02)`

用于软件演示的数学函数，不代表真实涂层、材料或 GNDs 规律。

先把输入归一化到 unit 空间，再按目标序号改变相位与余弦权重；每个目标由加权输入项和正弦项组成，最后加随机正态噪声。

核心形式为：

```python
value = (unit * weights).sum(-1)
value += 0.25 * torch.sin(unit * torch.pi * phase).sum(-1)
```

`sum(-1)` 对每行的输入维度求和，得到一个目标值。对多个目标的结果做 `torch.stack(..., dim=-1)`，得到 `[n,m]`。

不要把模拟数据改成 real 以绕过来源隔离。

## 7. `learning.py`：模型训练与验证

源文件：[learning.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/learning.py)。

### 7.1 为什么是高斯过程

高斯过程不只给一个预测数值，还给该位置的预测分布。项目利用均值回答“可能测得多少”，利用不确定性辅助下一轮实验选择。

当前模型对不同目标分别建模，不显式学习目标之间的相关性。多目标优化不等于训练了一个带跨目标相关性的模型。

### 7.2 `build_model(x, y, model_bounds)`

此函数构建模型，但还没有执行拟合。

```python
input_count = x.shape[1]
target_count = y.shape[1]
batch_shape = torch.Size([target_count])
```

输入维数和目标数量从张量读取。batch_shape 为每个目标配置独立的一组核参数。

模型构件如下：

| 构件 | 作用 |
|---|---|
| `SingleTaskGP` | BoTorch 高斯过程回归模型；这里使用其独立多输出形式 |
| `Normalize(d=..., bounds=...)` | 按 model_bounds 缩放输入 |
| `Standardize(m=...)` | 对各目标分别标准化，预测时恢复原单位 |
| `MaternKernel(nu=2.5)` | Matérn 5/2 核，描述不同输入之间的函数相关性 |
| `ard_num_dims=input_count` | 每个输入维度有自己的长度尺度 |
| `ScaleKernel` | 为核添加信号幅度尺度 |

长度尺度可以理解为函数在某输入方向上变化的快慢尺度，不能直接当成经过严格验证的因果重要性排名。

归一化不是把预测限制在 0—1，也不是把超出边界的输入裁剪回来。最终预测仍是目标变量原单位。

### 7.3 `fit(x, y, model_bounds)`

先调用 build_model，然后用 `ExactMarginalLogLikelihood` 和 `fit_gpytorch_mll` 拟合模型参数，最后调用 `model.eval()` 进入评估状态。

训练学习的是核长度尺度、信号尺度、噪声等参数。当前没有手写神经网络式的 epoch 循环，拟合过程交给 BoTorch/GPyTorch 的优化实现。

`model.eval()` 不会重新训练，也不等于关闭一切梯度。

### 7.4 `predict(model, x)`

在 `torch.no_grad()` 中调用 `model.posterior(x)`，返回后验均值和 `sqrt(variance)`。

`clamp_min(0)` 先避免数值误差产生极小负方差，再开平方。

这里的 std 是潜在函数标准差，报告列也明确叫 `latent_std`。代码没有显式启用观测噪声预测，所以不能把它简单理解为未来实验测量误差的完整标准差，也不能把均值±某倍数 std 当作必然覆盖的保证。

### 7.5 `cross_validate(config, x, y, model_bounds)`

使用留一交叉验证：有 n 个样本，就做 n 次拟合。第 i 次把 i 留出，用其他样本拟合，再预测 i；最终对所有留出预测计算误差。

每次会重新建模和拟合目标标准化。当前 model_bounds 是外部传入并在各折共用的，不在折内重新从训练子集计算；当边界来自全体输入的实际范围时，验证流程已经利用了全体输入范围信息。这与严格折内重新计算所有预处理步骤不同，不能把留一结果当作独立测试集证据。

记误差 `e=预测值-实验值`，每个目标分别输出：

- `MAE = mean(abs(e))`：平均绝对误差，与目标单位相同。
- `RMSE = sqrt(mean(e²))`：均方根误差，对大误差更敏感。
- `R2 = 1 - sum(e²)/sum((y-mean(y))²)`：相对整体均值的预测表现；可能为负。

目标所有值一样时，R² 分母为 0，代码输出 NaN。若目标变化范围很小，即使绝对误差不大，R² 也可能较差。

验证不直接保存每个留出样本的预测表，仅返回按目标汇总的指标。`config` 在这里主要提供目标名称。

## 8. `optimization.py`：推荐决策

源文件：[optimization.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/optimization.py)。

### 8.1 `pareto_summary(config, y)`

先统一目标方向，再调用 `is_non_dominated` 找非支配点，然后计算超体积 Hypervolume，返回 `(mask, hv, reference)`。

例如目标为“强度最大、成本最小”：

| 方案 | 强度 | 成本 | 关系 |
|---|---|---|---|
| A | 1.0 | 1.0 | 被 C 超越：C 强度更高、成本更低 |
| B | 2.0 | 2.0 | 强度较好但成本较高 |
| C | 1.5 | 0.5 | 成本较好但强度低于 B |

B、C 之间互有优劣，因此属于 Pareto 前沿。Pareto 不会自动决定你最终应该选 B 还是 C。

Hypervolume 衡量前沿相对参考点覆盖的目标空间体积。不同目标方向、单位、参考点和数据集条件下的 HV 不宜直接横向比较。尤其自动参考点可能随数据改变，demo 的每轮 HV 不一定基于固定参考点。

### 8.2 `_prediction_report(config, candidates, mean, std)`

把候选输入和预测拼成表格：`candidate_id`、输入列、每个目标的 `*_predicted` 和 `*_latent_std`。整数输入按整数显示。

它只是报告，不是下一批真实实验目标数据。

### 8.3 `recommend_pareto(...)`

流程：

1. 生成满足边界和约束的 Sobol 候选池。
2. 从历史目标计算统一最大化空间中的参考点。
3. 使用 `WeightedMCMultiOutputObjective` 乘以方向符号 +1/-1。
4. 构建 `qLogNoisyExpectedHypervolumeImprovement`，即 qLogNEHVI。
5. 使用 `SobolQMCNormalSampler` 的 128 个后验样本辅助采集函数估计。
6. 在有限候选池上调用 `optimize_acqf_discrete` 选择一批实验。
7. 预测被选候选，生成报告和元数据。

q 表示批次实验；Noisy 表示考虑噪声场景；Hypervolume Improvement 表示有望改善多目标前沿的价值；Log 形式用于该采集函数的数值处理。保存的 acquisition_value 不是实验值，也不是达到要求的概率。

`X_baseline=train_x` 使用已有实验作为基线；`prune_baseline=True` 允许库裁剪不重要基线点。`unique=True` 避免批次内重复，`X_avoid=train_x` 要求避开已实验输入。`max_batch_size=1024` 控制库内评估块大小，不是推荐 1024 个点。

注意两个不同的 Sobol：data.py 的 Sobol 生成输入候选；这里的 SobolQMCNormalSampler 用于从模型预测分布采样。

此分支不是在整个连续空间上保证找到全局最优，而是在生成的有限候选里优化采集函数。goal.weight 不进入这里；目标方向由 pareto_direction 决定。

### 8.4 `_target_scores(config, mean, std, train_y)`

返回每个候选总分及各目标的归一化误差。分数越低越好。

先计算训练目标标准差作为默认尺度。标准差太小时，备用尺度为 `max(mean(abs(y)),1)*0.1`，防止除以零或极小值。

对第 j 个目标，记候选预测均值为 μ、潜在标准差为 σ、期望值为 v，尺度为 s：

| 模式 | 误差项 |
|---|---|
| `target` | `abs(μ-v)/s` |
| `range` | `(max(lower-μ,0)+max(μ-upper,0))/区间尺度` |
| `at_least` | `max(v-μ,0)/s` |
| `at_most` | `max(μ-v,0)/s` |
| `maximize` | `(候选池最大预测-μ)/候选池预测跨度` |
| `minimize` | `(μ-候选池最小预测)/候选池预测跨度` |

range 的区间尺度优先使用 tolerance，否则为区间宽度；其他点目标/阈值的 s 优先用 tolerance，否则用观测标准差。

每个目标的贡献：

```text
贡献_j = weight_j × (误差_j + risk_aversion × σ_j / scale_j)
总分 = 所有参与目标的贡献之和 / 所有参与目标的权重之和
```

maximize/minimize 的误差归一化用候选池预测跨度，但不确定性项仍使用前面选出的 tolerance 或观测尺度；两者不必相同。

例子：两个目标权重均为 1，归一化误差分别 0.4、0.2，归一化不确定性分别 0.1、0.3，risk_aversion=0.25，则总分为 `(0.4+0.025+0.2+0.075)/2=0.35`。

满足 range 或单边阈值时误差为 0，但不确定性惩罚可能仍大于 0。多个目标使用加权折中，所以选出的点可能不满足其中某个要求；当前没有硬性的目标可行性筛选。

components 保存 `*_normalized_error`，不包含单独的逐目标风险惩罚列。

### 8.5 `_select_diverse(scores, pool, train_x, bounds, count, minimum_distance)`

先按分数从低到高排序，再在按输入范围归一化的空间挑选。

1. 与已训练输入距离小于 `1e-9` 的候选跳过，避免精确重复。
2. 与已选候选距离小于 minimum_distance 的先放到 deferred。
3. 优先选目标较好且彼此较分散的点。
4. 若数量不足，按先前顺序从 deferred 补齐。
5. 最终仍不够时才报错。

因此 minimum_distance 是会被放宽的批内偏好，不是永远强制的硬约束；对训练点只排除极近的重复，并不要求与历史输入都相隔 minimum_distance。

### 8.6 `recommend_target(...)`

依次调用 candidate_pool → predict → _target_scores → _select_diverse → _prediction_report。

输出：候选张量、预测报告、`{"score_definition": "lower_is_better"}`。报告额外包含 `target_match_score`、各目标归一化误差和推荐模式。

当前没有连续局部优化步骤，也没有导出“所有目标已达标”的布尔判断；需要自己根据预测及真实复测结果判断。

### 8.7 `recommend_candidates(...)`

这是路由函数：mode 为 pareto 调用 recommend_pareto，否则调用 recommend_target。模式是否合法已经由 config.py 检查。

新增推荐策略时这里是分发位置，但还要同步配置校验、说明和测试。

## 9. `artifacts.py`：保存和恢复模型

源文件：[artifacts.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/artifacts.py)。

### 9.1 `create_run(label)`

创建 `outputs/标签_年月日_时分秒_微秒/`。用 `exist_ok=False` 避免复用旧结果目录。时间来自本机时钟，不是固定写死的日期。

### 9.2 `write_json(data, path)`

以 UTF-8、保留中文、缩进 2 空格写 JSON，方便人工查看。它不像 save_csv 那样采用独占创建；通常在新建时间戳目录内调用。

### 9.3 `save_training_run(...)`

保存以下文件：

| 文件 | 内容 |
|---|---|
| `training_snapshot.csv` | 本次模型实际使用的数据 |
| `settings_snapshot.yaml` | 当时的设置原文件 |
| `cv_metrics.csv` | 可选的验证指标；跳过 CV 时没有此文件 |
| `model.pt` | 变量签名、训练归一化边界和模型 state_dict |
| `training_metadata.json` | 项目名、行数、输入目标名称、模型文件名 |

`state_dict` 保存模型参数和相关缓冲量，而不是简单保存一个可随处使用的完整函数。恢复还需要训练数据和匹配的模型结构，所以应该整套保存训练目录，不能只留下 model.pt。

### 9.4 `load_training_run(config, run)`

1. 检查 training_snapshot.csv 和 model.pt 是否存在。
2. 用 torch.load 读取 checkpoint，优先使用 `weights_only=True`。
3. 与当前 schema_signature 比较，不一致则要求重新训练。
4. 检查快照来源全为 real 或全为 demo。
5. 使用当前配置调用 load_data 再检查快照。
6. 恢复保存的 model_bounds，按快照 x/y 重建模型。
7. 加载 state_dict，进入 eval 状态。
8. 返回模型、数据表、x、y、训练边界。

模型恢复不会调用 fit 重新优化参数。

结构签名相同不代表一定可以随意缩小当前范围：第 5 步仍会使用当前边界/约束检查历史输入。如果新范围排除历史点，就可能加载失败。建议保留兼容历史数据的配置；如需独立设置训练范围和推荐范围，要另行修改代码，当前并无这个独立接口。

这里能恢复 demo 模型供内部用途，但正式 recommend 命令会进一步拒绝 demo 来源。

模型文件只应从可信的本地来源加载；代码为了兼容旧版 PyTorch 有一个非 weights_only 的回退，不能把它视为对任意外部模型文件都安全。

## 10. `__main__.py`：命令如何执行

源文件：[__main__.py](/Users/william/Documents/ProjectPython/COATING_AI/src/coating_ai/__main__.py)。

### 10.1 启动方式

`python -m coating_ai` 会执行包中的 __main__.py。安装后也可以使用 pyproject.toml 声明的 `coating-ai` 命令。

不要直接运行包里的单个 __main__.py 来代替 `-m`，否则包内相对导入可能失败。PyCharm 可以把运行方式设为 Module name：`coating_ai`，再填写命令参数。

### 10.2 三个辅助函数

`_add_settings_argument(parser)` 为每个子命令添加 `--settings`，默认是项目 settings/project.yaml。

`_save_pareto_report(config, table, y, folder)` 计算历史观测的 Pareto 前沿并保存 observed_pareto.csv，返回前沿数量、HV、参考点。当前不论推荐模式是否为 target 都会执行；这是额外的观测报告，不表示 target 推荐也使用 Pareto 采集函数。

`_candidate_measurement_table(config, candidates, folder)` 用 frame 生成目标为空的真实实验待填写表，再将样本编号改为含运行目录名称的唯一编号，降低不同批次编号冲突风险。

### 10.3 `_command_init(args, config)`

调用 initial_points → frame → save_csv，输出下一步需要测量的初始输入。默认写 initial_design.csv，也可用 `--out` 指定新文件。

不会训练，不会生成真实目标值。默认实验数和 seed 可通过 `--n、--seed` 覆盖。

### 10.4 `_command_validate(args, config)`

调用 load_data → resolve_search_bounds → create_run → cross_validate → _save_pareto_report。

保存训练数据和设置快照、cv_metrics.csv、observed_pareto.csv、validation_metadata.json。此命令每折会训练临时模型用于验证，但不保存可供 recommend 使用的最终 model.pt。

### 10.5 `_command_train(args, config)`

读数据、计算训练边界、建立训练目录；默认先交叉验证，然后用全部数据拟合最终模型，保存训练套件，再保存 Pareto 观测报告和 pareto_metadata.json。

`--skip-cv` 只跳过验证，不跳过最终模型训练。正式判断可靠性时不要仅看全部训练数据上的拟合。

### 10.6 `_command_recommend(args, config)`

恢复训练模型 → 确认来源为 real → 用当前设置解析搜索范围 → 分发推荐策略 → 建立结果目录 → 保存两份性质不同的表。

| 文件 | 用途 |
|---|---|
| `candidates_to_measure.csv` | 真实实验待填写表，目标为空 |
| `predictions_not_measurements.csv` | 模型预测、潜在不确定性和推荐分数 |
| `settings_snapshot.yaml` | 此次推荐使用的设置 |
| `observed_pareto.csv` | 历史实测数据的非支配点 |
| `recommendation_metadata.json` | 推荐策略、模型来源及相关数值 |

两张表通过 sample_id 对应。把真实测量填写到 candidates_to_measure 的副本，再加入训练数据；不要复制预测值冒充测量值。

### 10.7 `_command_merge(args, config)`

旧表按默认至少 6 行检查，新表允许至少 1 行；拼接后检查 sample_id 冲突，独占创建新 CSV，并再次检查完整合并数据。

如果最后一次检查失败，会删除本次新创建的输出文件，不会删除原旧表和新表。此处 Exception 捕获用于回收不合格的合并产物。

当前只按 sample_id 排除重复编号，不会自动识别不同编号下相同的输入。重复实验若有独立实测目标，可以保留独立编号，但要自己管理其科学含义。

### 10.8 `_command_demo(args, config)`

设置随机种子，用 simulate 产生初始目标，然后重复：保存当前数据 → 记录 HV → 拟合 → 推荐 → 模拟测量 → 追加 x/y。

`--rounds 1` 表示在初始数据之后再推荐和模拟测量一批；循环处理第 0、1 轮。`--rounds 0` 只处理初始数据。

最后保存 history.csv、每轮数据/预测表、最终训练套件、observed_pareto.csv 和 demo_metadata.json。它验证软件闭环，并不证明真实材料模型准确。

### 10.9 `build_parser()`

用 argparse 定义六个命令：

| 命令 | 专属参数 |
|---|---|
| `init` | 可选 `--out、--n、--seed` |
| `validate` | 必填 `--data` |
| `train` | 必填 `--data`，可选 `--skip-cv` |
| `recommend` | 必填 `--run` |
| `merge` | 必填 `--data、--new、--out` |
| `demo` | 可选 `--rounds`，默认 1 |

每个命令都有 `--settings`。注意它属于子命令参数，应写在 init/train 等子命令后。

### 10.10 `main()`

解析参数 → 检查 n 和 rounds → 限制 torch 线程为 2 → 读取配置 → 通过 handlers 字典调用命令函数。

`rounds` 必须在 0—20，`--n` 必须大于 0。`torch.set_num_threads(2)` 控制 CPU 计算线程，不是两个模型，也不是两个实验批次。

`handlers[args.command](args, config)` 可以拆成两步理解：先从字典取函数，再调用这个函数。

## 11. 辅助脚本、包文件和调试文件

### 11.1 `scripts/generate_doe_excel.py`

源文件：[generate_doe_excel.py](/Users/william/Documents/ProjectPython/COATING_AI/scripts/generate_doe_excel.py)。

`build_parser()` 定义 `--settings、--output、--n、--seed`。

`main()`：读配置 → 判断输出是否已存在 → 复用 initial_points 和 frame → 用 pandas 写 xlsx → 用 openpyxl 做排版。

排版操作：

- `freeze_panes="C2"`：滚动时固定第一行和前两列。
- `auto_filter`：启用表头筛选。
- 表头加粗、浅蓝填充、居中。
- 根据内容调整列宽，限制在 12—24 之间。

它不读取新的实测数据、不训练模型、不执行推荐。文件已经存在时拒绝覆盖。

### 11.2 `src/coating_ai/__init__.py`

只有包说明和 `__version__="0.2.0"`。声明包及版本，不在导入时执行训练。

### 11.3 `scripts/__init__.py`

只有“项目辅助脚本”说明，标识辅助脚本目录的包身份。

### 11.4 `src/Test.py`

目前是空文件，没有需要解释的算法。用于你自己的临时练习，Git 忽略，不是正式运行入口。

## 12. `tests/test_workflow.py`：全部测试

源文件：[test_workflow.py](/Users/william/Documents/ProjectPython/COATING_AI/tests/test_workflow.py)。

### 12.1 测试辅助代码

`settings(mode="target", observed_bounds=False, pool_size=128)` 返回小型测试配置：temperature 连续、layers 整数、strength 最大化、cost 最小化，带线性约束。换一组与 A/B/C 不同的名称，是为了确认算法不依赖示例列名。

`Posterior.__init__(x)` 构造可预测的假均值和方差，均值为输入各列除以 10，方差为 0.01。

`PredictableModel.posterior(x)` 返回上述假后验。这样 target 推荐测试不必每次拟合真正 GP，结果更稳定；它不是正式模型。

`WorkflowTests.setUp()` 为每个测试创建临时目录；`tearDown()` 清理；`config(**kwargs)` 把测试设置保存到临时 YAML，再用正式 load_project_config 解析。

### 12.2 六项测试分别验证什么

| 测试函数 | 检查内容 |
|---|---|
| `test_dynamic_schema_integer_inputs_and_constraints` | 动态变量名称、整数列索引、初始点形状、整数性和可行性 |
| `test_disabled_bounds_use_observed_range_and_margin` | 2—8 扩展为 1.4—8.6；没有观测时关闭边界必须报错 |
| `test_real_data_must_be_complete_and_not_mixed_with_demo` | 数据正确读回；空白目标和 real/demo 混合必须报错 |
| `test_pareto_supports_mixed_directions` | 最大化强度与最小化成本的非支配 mask、HV、参考点形状 |
| `test_target_mode_returns_feasible_named_recommendations` | 推荐数量、约束、整数输入、预测字段和低分更好定义 |
| `test_saved_model_rejects_a_different_variable_schema` | 模型套件可恢复；改输入名称/顺序后必须拒绝旧模型 |

最后一项使用真实构建的 GP，但不调用 fit；它主要测试结构保存与恢复，不是科研预测精度测试。

### 12.3 如何运行与理解结果

在项目根目录、AI 环境中执行：

```bash
python -m unittest discover -s tests -v
```

该命令只需要 Python 自带 unittest，当前项目依赖没有显式包含 pytest。旧 README 中的 `pytest -q` 只有另外安装了 pytest 才能使用。

全部通过说明上述软件规则仍工作，不代表真实实验模型足够准确。当前测试也没有完整覆盖所有 goal 模式、所有命令或 Pareto 实际拟合与推荐全过程。

## 13. 安装、依赖与 Git 文件

### 13.1 `pyproject.toml`

`build-system` 指定用 setuptools 构建；`project` 声明名称 coating-ai、版本 0.2.0、Python 3.11 以及依赖；`project.scripts` 把命令 `coating-ai` 指向 `coating_ai.__main__:main`。

`package-dir={""="src"}` 和 packages.find 表明使用 src 布局：代码实际放在 src/coating_ai，不是项目根目录的 coating_ai。

主要库的分工：

| 库 | 用途 |
|---|---|
| PyTorch | 张量、模型状态和基础数值运算 |
| GPyTorch | 高斯过程核与边缘似然等 |
| BoTorch | 高斯过程接口、拟合、Pareto 采集函数和优化 |
| pandas | CSV 表格处理 |
| PyYAML | 读取 YAML 设置 |
| openpyxl | Excel 创建与排版 |

pyproject 中直接声明 BoTorch、pandas、openpyxl、PyYAML；Torch、GPyTorch 等随依赖关系提供。

### 13.2 `requirements.txt`

内容为 `-e .`，表示对当前项目执行可编辑安装。`python -m pip install -r requirements.txt` 与这里使用的 `python -m pip install -e .` 都会安装本项目。

可编辑安装后，Python 使用 src 中的代码；通常修改 .py 文件不必重新安装，但改变安装依赖或命令入口时应刷新安装信息。

### 13.3 `requirements-lock.txt`

记录本机曾解析的环境版本，注明 Python 3.11 / macOS arm64。它不是运行时配置，安装 -e . 不会自动读取它。

里面包含比 pyproject 直接依赖更多的库，不能因为存在于该列表就认定当前代码使用了它。当前通用模型没有导入 sklearn 的 ExtraTrees，也没有使用 GNDs 的 joblib 模型。

不同系统或硬件不能未经核对就保证这些历史版本全部可用；一般先按当前 pyproject 配置环境，只有需要复现实验环境时再核对版本快照。

### 13.4 `.gitignore`

忽略 Python 缓存、IDE 设置、环境目录、构建信息、日志与本地敏感配置、src/Test.py、outputs 和真实数据。

`data/*` 忽略 data 内容，随后 `!data/initial_design.csv` 单独保留公开的空目标示例。

Git 适合保存代码、设置、测试和文档。真实数据和模型仍需你另外备份；“被 Git 忽略”不等于自动获得备份。

### 13.5 文档与生成文件

README 和 docs/LEARNING_GUIDE.md 是说明，不参与数值计算。data/initial_design.csv 是输入示例，目标为空，不能直接训练。

outputs 中的旧通用 demo 是模拟产物，不是源代码或真实科研结论；__pycache__ 和 egg-info 为运行/安装生成，不需要手工逐行学习。

## 14. 完整操作：首次实验到后续重训

以下命令都在项目根目录执行。文件名为示例；有同名文件时换新版本名，不覆盖实测记录。没有真实实验值就停在建表步骤，不要编造数值运行后续训练。

### 14.1 准备环境和设置

已有 AI 环境时不必再次创建：

```bash
conda activate AI
cd /Users/william/Documents/ProjectPython/COATING_AI
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

PyCharm 解释器选择 `/opt/anaconda3/envs/AI/bin/python`。随后在 settings/project.yaml 定义真实输入、目标、单位和约束。单位由你保持一致，代码没有自动单位换算。

### 14.2 生成新初始实验表

已有 initial_design.csv，所以使用新的输出名：

```bash
python -m coating_ai init --out data/round_00_to_measure.csv
```

或生成 Excel：

```bash
python scripts/generate_doe_excel.py --output data/round_00_to_measure.xlsx
```

完成实验后另存为 `data/round_00_measured.csv`，保留 sample_id/data_kind，填写所有启用目标。

### 14.3 验证和训练

```bash
python -m coating_ai validate --data data/round_00_measured.csv
python -m coating_ai train --data data/round_00_measured.csv
```

这两条都默认做留一验证，会重复该项计算。可以直接 train 完成验证与最终拟合；单独 validate 更适合先审查质量、尚不保存最终模型的场景。

只在快速调试时使用 `train --skip-cv`。记录 train 命令实际打印出的完整输出目录。

### 14.4 告诉模型你想要的目标

在 settings/project.yaml 中把 mode 设为 target，修改目标变量的 goal。例如 X 希望接近 0.8、Y 不低于 0、Z 在 -0.5—0.5，当前示例已这样设置。

随后执行下面的命令，把尖括号部分替换成实际目录名；它不是可以照抄的真实路径：

```text
python -m coating_ai recommend --run outputs/<实际训练目录名>
```

只改 goal 的值/权重/容差通常不需要重训；改输入或目标名称、启用集合或顺序则需要重训。Pareto 模式改为 pareto 并设置各目标方向，使用同一个 recommend 命令。

当前通用项目不读取 Target.csv 请求表；指定目标通过 YAML 设置传入，也没有固定部分输入的独立接口。不要把归档 GNDs 流程与此处混用。

### 14.5 完成新实验并合并

把 candidates_to_measure.csv 另存为 `data/batch_01_measured.csv`，保留推荐样本编号，填真实目标。

```bash
python -m coating_ai merge \
  --data data/round_00_measured.csv \
  --new data/batch_01_measured.csv \
  --out data/round_01_measured.csv
python -m coating_ai train --data data/round_01_measured.csv
```

这是用完整新旧数据重新拟合，不是在旧模型上只增加一次神经网络训练步骤。推荐时改用新训练目录。

### 14.6 只学习软件流程时

```bash
python -m coating_ai demo --rounds 1
```

demo 会写入新的 outputs 目录；不改真实实验表。切换 recommendation.mode 后可以分别观察 target 与 pareto 演示，但不要把模拟模型交给正式 recommend。

## 15. 当前限制、调试练习和修改位置

### 15.1 需要记住的实际限制

1. 只支持连续和整数输入，不支持类别变量；不自动处理角度周期性。
2. CSV 必须带 sample_id/data_kind，目标完整；不做缺失值自动填补。
3. target 模式的 goal 在加载配置时就需要有效，即使只是训练。
4. 输入约束只支持线性不等式；目标值/区间是打分要求，不是硬性可行约束。
5. 推荐在有限候选池里搜索，不保证全局最优；反向解可能不存在或有多个。
6. target 批次距离在候选不足时会放宽；不是严格的最小间距保证。
7. 恢复模型时当前边界/约束仍会检查历史数据，缩小搜索域可能导致加载失败。
8. 单个 GP 目标相互独立，不显式建模跨目标相关性；没有单独的深度学习或增量训练接口。
9. 留一验证没有独立留出测试集；不能单靠自动测试或演示证明科研预测准确。
10. 推荐不输出自动的全目标达标判定，潜在 std 也不是全部测量误差。
11. 配置允许动态目标数量，但 Pareto 主要用于多目标；没有专门单目标优化分支。已存在的测试不是对任意维数/参数的完整保证。
12. 未实现自动执行实验、自动导入新测量、自动停止或无人值守训练；闭环中的真实实验与文件管理仍由用户完成。

以上是说明当前代码，不是本次新增修改或已经修复的问题。

### 15.2 在 PyCharm 中设置断点

| 断点位置 | 观察内容 |
|---|---|
| config.load_project_config 返回处 | config 的各字段、启用变量、目标和约束 |
| data.candidate_pool 返回处 | 候选形状、整数列、mask、去重结果 |
| data.load_data 返回处 | CSV 如何转换为 x/y，列顺序是否正确 |
| learning.build_model 返回处 | 输入/目标变换、核结构和 batch_shape |
| learning.predict 返回处 | mean/std 的形状和目标原单位 |
| optimization._target_scores 循环处 | 每个 goal 的误差、尺度、权重和风险贡献 |
| optimization._select_diverse 选择处 | 哪些点被跳过或延后，多样性怎样放宽 |
| artifacts.load_training_run 签名比较处 | 为什么配置变化会拒绝旧模型 |
| __main__.main 的 handlers 调用处 | 参数如何分发到不同命令 |

运行 Python Module `coating_ai`，用 init 或 demo 作入门调试；不要直接从包中某个源文件启动。

### 15.3 推荐练习

1. 在一个单独练习 YAML 中，把 A 改名为 temperature，同步修改约束引用，观察生成表列名随之改变。
2. 调整 C 的整数上下界，观察 initial_points 和 CSV 显示。
3. 关闭约束总开关，比较生成候选是否仍受 A+B+C 限制。
4. 改变一个目标的 weight/tolerance，观察 normalized_error 和 target_match_score。
5. 用 demo 对比 target 与 pareto；理解同一正向模型可以支持不同推荐规则。
6. 在测试里先明确一条期望行为，再修改对应代码，运行全部测试验证。

练习 YAML 可以通过 `--settings` 传入，避免直接改正式实验配置。不要把练习模拟结果并入 real 数据。

### 15.4 需求与修改位置速查

| 你想改变什么 | 优先看哪里 | 是否通常需要重训 |
|---|---|---|
| 目标值、区间、权重、容差 | settings/project.yaml 的 goal | 否，变量结构不变时 |
| 推荐方式和批次数量 | settings/project.yaml 的 recommendation | 否 |
| 输入名字、类型、开关；目标名字、开关、顺序 | settings + 对应数据列 | 是 |
| 边界和线性约束 | settings + config.resolve_search_bounds/feasible | 未必，但旧数据加载可能失败 |
| CSV 格式和检查规则 | data.frame/load_data/save_csv | 数据/结构变化时重新评估 |
| GP 核、变换或训练算法 | learning.build_model/fit | 是 |
| target 匹配准则 | optimization._target_scores | 通常不需要重拟合，但需重新推荐与验证 |
| target 多样性选择 | optimization._select_diverse | 通常否 |
| Pareto 采集函数 | optimization.recommend_pareto | 视模型结构改动而定 |
| 存档字段或恢复方式 | artifacts.save_training_run/load_training_run | 需考虑旧模型兼容性 |
| 新命令或流程 | __main__.build_parser/main + 对应函数 | 视功能而定 |
| Excel 排版 | scripts/generate_doe_excel.py | 否 |

学习时最重要的一条主线是：配置定义规则，数据构成 x/y，模型学习正向关系，推荐算法选择新输入，真实测量后再重新训练。
