"""通用输入设计、候选池、CSV读写和数据质量检查。"""
from pathlib import Path
import hashlib

import pandas as pd
import torch

from .config import DTYPE, ProjectConfig, feasible, resolve_search_bounds


def _stable_unique(values: torch.Tensor) -> torch.Tensor:
    """按生成顺序去重，避免排序后只保留搜索空间的一侧。"""
    seen: set[tuple[float, ...]] = set()
    indices: list[int] = []
    for index, row in enumerate(values.tolist()):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return values[indices]


def candidate_pool(
    config: ProjectConfig,
    bounds: torch.Tensor,
    size: int,
    seed: int,
) -> torch.Tensor:
    """生成满足边界、整数要求和已启用约束的Sobol候选池。"""
    dimension = len(config.input_names)
    # SobolEngine中的dimension是搜索空间维度；scramble和seed保证可复现的加扰序列。
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    blocks: list[torch.Tensor] = []
    for _ in range(100):
        unit = engine.draw(max(256, size * 2), dtype=DTYPE)
        values = bounds[0] + unit * (bounds[1] - bounds[0])
        for index in config.integer_indices:
            values[:, index] = values[:, index].round()
        values = values[feasible(config, values, bounds)]
        if len(values):
            blocks.append(values)
        if blocks:
            unique = _stable_unique(torch.cat(blocks))
            if len(unique) >= size:
                return unique[:size]
    found = len(_stable_unique(torch.cat(blocks))) if blocks else 0
    raise ValueError(f"可行候选点不足：需要{size}个，只找到{found}个。请检查边界和约束。")


def initial_points(config: ProjectConfig, n: int | None = None, seed: int | None = None) -> torch.Tensor:
    """生成初始实验点；初始阶段没有数据，因此所有启用输入必须具有显式边界。"""
    count = n or config.initial_design_size
    bounds = resolve_search_bounds(config)
    return candidate_pool(config, bounds, count, config.recommendation.seed if seed is None else seed)


def frame(
    config: ProjectConfig,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    start: int = 1,
    kind: str = "real",
) -> pd.DataFrame:
    """把Tensor转换为带变量名称、样本编号和数据来源的实验表。"""
    table = pd.DataFrame(x.detach().cpu().numpy(), columns=config.input_names)
    for index in config.integer_indices:
        table[config.input_names[index]] = table[config.input_names[index]].round().astype(int)
    table.insert(0, "sample_id", [f"{kind}_{i:04d}" for i in range(start, start + len(x))])
    table.insert(1, "data_kind", kind)
    for column, name in enumerate(config.target_names):
        table[name] = float("nan") if y is None else y[:, column].detach().cpu().numpy()
    return table


def save_csv(table: pd.DataFrame, path: str | Path) -> None:
    """独占创建CSV，防止重新运行时覆盖已经填写的真实实验结果。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8-sig", newline="") as file:
        table.to_csv(file, index=False)


def load_data(
    config: ProjectConfig,
    path: str | Path,
    *,
    allow_demo: bool = False,
    min_rows: int = 6,
    check_search_space: bool = True,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """验证动态列、来源、有限数值、整数类型和当前可行性后返回训练张量。"""
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    if table.columns.duplicated().any():
        raise ValueError("CSV列名清理空格后存在重复。")
    required = [*config.input_names, *config.target_names]
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"缺少列：{sorted(missing)}")
    if len(table) < min_rows:
        raise ValueError(f"至少需要{min_rows}组完整数据。")
    # 允许直接读取用户的原始变量表；在内存中补齐追踪信息，不修改源文件。
    if "sample_id" not in table:
        numeric = table[required].astype(float)
        ids = ["import_" + hashlib.sha256(
            ",".join(float(value).hex() for value in row).encode("utf-8")
        ).hexdigest()[:20] for row in numeric.to_numpy()]
        table.insert(0, "sample_id", ids)
    if "data_kind" not in table:
        if allow_demo:
            raise ValueError("模拟数据必须显式声明data_kind=demo。")
        table.insert(1, "data_kind", "real")
    if table["sample_id"].isna().any() or table["sample_id"].duplicated().any():
        raise ValueError("sample_id必须非空且不能重复。")
    allowed = {"demo"} if allow_demo else {"real"}
    if set(table["data_kind"]) != allowed:
        raise ValueError(f"本次只接受data_kind={allowed}；不可混合模拟与真实数据。")
    x = torch.tensor(table[config.input_names].to_numpy(dtype=float), dtype=DTYPE)
    y = torch.tensor(table[config.target_names].to_numpy(dtype=float), dtype=DTYPE)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("存在空白、无穷值或无效数字；请填完所有启用目标变量后再训练。")
    if check_search_space:
        bounds = resolve_search_bounds(config, x)
        invalid = ~feasible(config, x, bounds)
    else:
        # 缩小下一批搜索范围不应使历史训练样本无法加载。
        invalid = torch.zeros(len(x), dtype=torch.bool)
        for index in config.integer_indices:
            invalid |= ~torch.isclose(x[:, index], x[:, index].round(), atol=1e-6, rtol=0)
    if invalid.any():
        rows = table.loc[invalid.cpu().numpy(), "sample_id"].tolist()
        raise ValueError(f"以下样本违反输入类型、边界或已启用约束：{rows}")
    return table, x, y


def simulate(
    config: ProjectConfig,
    x: torch.Tensor,
    bounds: torch.Tensor,
    noise: float = 0.02,
) -> torch.Tensor:
    """动态维度数学演示函数；不代表任何材料规律或真实实验结论。"""
    unit = (x - bounds[0]) / (bounds[1] - bounds[0])
    dimensions = torch.arange(1, unit.shape[1] + 1, dtype=DTYPE)
    outputs: list[torch.Tensor] = []
    for target_index in range(len(config.target_names)):
        phase = target_index + 1
        weights = torch.cos(dimensions * phase)
        value = (unit * weights).sum(-1)
        value += 0.25 * torch.sin(unit * torch.pi * phase).sum(-1)
        outputs.append(value)
    y = torch.stack(outputs, dim=-1)
    return y + noise * torch.randn_like(y)
