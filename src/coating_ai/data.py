from pathlib import Path

import pandas as pd
import torch

from .config import BOUNDS, DTYPE, FEATURES, TARGETS, feasible


def initial_points(n=20, seed=42):
    """Sobol 序列本身具有优秀的空间填充性和低差异性。但经过 feasible(x) 约束筛选后，被剔除的点破坏了原始序列的结构，剩余的点不再保证完整的均匀性。"""
    # Argument：5：搜索空间的维度，对应输入变量；
    # Argument：scramble=True：启用 Owen 加扰。加扰后能显著改善高维均匀性；
    # Argument：seed 随机种子。仅在 scramble=True 时生效，用于控制加扰过程的确定性，保证实验可复现。
    # orch.quasirandom.SobolEngine 这种写法是 Python 的模块层级访问语法。其中的 torch.quasirandom. 表示一个嵌套的模块路径。
    engine = torch.quasirandom.SobolEngine(5, scramble=True, seed=seed) #初始化 Sobol 引擎，产生engine（对象）
    # 属性（Attribute）和 方法（Method）// 根本区别在于：属性是“数据”，方法是“行为”。
    ## 属性：访问数据，不加括号
    # print(engine.dimension)   # 5        → 记录维度是多少
    # print(engine.scramble)    # True     → 记录是否启用了 scramble
    # print(engine.num_generated) # 0      → 记录已生成了多少个点
    # 方法：执行动作，必须加括号
    # points = engine.draw(10)  # 生成10个拟随机点 → 执行"采样"这个动作
    # engine.reset()            # 重置内部计数器   → 执行"重置"这个动作

    blocks = []
    for _ in range(100):
        x = BOUNDS[0] + engine.draw(max(64, n), dtype=DTYPE) * (BOUNDS[1] - BOUNDS[0])
        blocks.append(x[feasible(x)])
        if sum(len(b) for b in blocks) >= n:
            return torch.cat(blocks)[:n]
    raise ValueError("可行点不足，请检查约束。")


def frame(x, y=None, start=1, kind="real"):
    df = pd.DataFrame(x.detach().numpy(), columns=FEATURES)
    df.insert(0, "sample_id", [f"{kind}_{i:04d}" for i in range(start, start + len(x))])
    df.insert(1, "data_kind", kind)
    for j, name in enumerate(TARGETS):
        df[name] = float("nan") if y is None else y[:, j].detach().numpy()
    return df


def save(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 避免重新运行时覆盖已填入的真实实验结果。
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        df.to_csv(f, index=False)


def load(path, allow_demo=False, min_rows=6):
    df = pd.read_csv(path)
    required = ["sample_id", "data_kind"] + FEATURES + TARGETS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"缺少列：{sorted(missing)}")
    if len(df) < min_rows or df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError(f"至少需要{min_rows}组完整数据，且sample_id必须非空、不重复。")
    allowed = {"demo"} if allow_demo else {"real"}
    if set(df["data_kind"]) != allowed:
        raise ValueError(f"本次只接受data_kind={allowed}；不可混合模拟与真实数据。")
    x = torch.tensor(df[FEATURES].to_numpy(dtype=float), dtype=DTYPE)
    y = torch.tensor(df[TARGETS].to_numpy(dtype=float), dtype=DTYPE)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("存在空白、无穷值或无效数字；请填完三项测量值后再训练。")
    if not feasible(x).all():
        raise ValueError("部分配方超出边界或违反总填料/比例约束。")
    if ((y[:, 1] < 0) | (y[:, 1] > 5)).any() or (y[:, 2] < 0).any():
        raise ValueError("adhesion_score应在0–5之间，healing_pct不能为负。")
    return df, x, y


def simulate(x):
    """纯数学教学函数，不是化学模型，不代表材料性能或实验结论。"""
    u = (x - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    a, b, c, d, e = u.unbind(-1)
    y = torch.stack([7 + 2*a + 0.7*e - 0.8*b - (d-0.4)**2,
                     3.5 - 2*a + d - 0.5*e,
                     25 + 45*b + 15*c - 15*d + 5*a], -1)
    return y + torch.randn_like(y) * torch.tensor([0.03, 0.03, 0.3], dtype=DTYPE)
