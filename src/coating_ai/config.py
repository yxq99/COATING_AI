"""读取并验证用户设置；代码中不再写死变量名称、维度、边界和约束。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # Path把路径从纯字符串变成有行为的对象。
import math
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 当前文件绝对路径向上回溯两级。
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "settings" / "project.yaml"
DTYPE = torch.double
RESERVED_COLUMNS = {"sample_id", "data_kind"}


@dataclass(frozen=True)
class BoundsSpec:
    enabled: bool
    lower: float | None
    upper: float | None
    observed_margin: float = 0.0


@dataclass(frozen=True)
class InputVariable:
    name: str
    kind: str
    enabled: bool
    bounds: BoundsSpec
    period: float | None = None


@dataclass(frozen=True)
class GoalSpec:
    enabled: bool
    mode: str
    weight: float
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    tolerance: float | None = None


@dataclass(frozen=True)
class TargetVariable:
    name: str
    enabled: bool
    pareto_direction: str
    reference_value: float | None
    goal: GoalSpec


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    enabled: bool
    coefficients: dict[str, float]
    operator: str
    rhs: float


@dataclass(frozen=True)
class RecommendationSpec:
    mode: str
    batch_size: int
    candidate_pool_size: int
    seed: int
    risk_aversion: float
    diversity_min_distance: float
    auto_reference_margin: float


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    initial_design_size: int
    input_variables: tuple[InputVariable, ...]
    target_variables: tuple[TargetVariable, ...]
    constraints_enabled: bool
    constraints: tuple[LinearConstraint, ...]
    recommendation: RecommendationSpec
    source_path: Path

    @property
    def input_names(self) -> list[str]:
        return [item.name for item in self.input_variables if item.enabled]

    @property
    def target_names(self) -> list[str]:
        return [item.name for item in self.target_variables if item.enabled]

    @property
    def active_inputs(self) -> tuple[InputVariable, ...]:
        return tuple(item for item in self.input_variables if item.enabled)

    @property
    def active_targets(self) -> tuple[TargetVariable, ...]:
        return tuple(item for item in self.target_variables if item.enabled)

    @property
    def integer_indices(self) -> list[int]:
        return [i for i, item in enumerate(self.active_inputs) if item.kind == "integer"]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是YAML映射。")
    return value


def _number(value: Any, label: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是数字。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数字。")
    return result


def _goal(name: str, raw: Any) -> GoalSpec:
    data = _mapping(raw or {}, f"目标变量{name}.goal")
    enabled = bool(data.get("enabled", True))
    mode = str(data.get("mode", "ignore")).lower()
    allowed = {"target", "range", "at_least", "at_most", "maximize", "minimize", "ignore"}
    if mode not in allowed:
        raise ValueError(f"目标变量{name}的goal.mode必须属于{sorted(allowed)}。")
    weight = _number(data.get("weight", 1.0), f"目标变量{name}.goal.weight")
    if weight <= 0:
        raise ValueError(f"目标变量{name}.goal.weight必须大于0。")
    value = _number(data.get("value"), f"目标变量{name}.goal.value", optional=True)
    lower = _number(data.get("lower"), f"目标变量{name}.goal.lower", optional=True)
    upper = _number(data.get("upper"), f"目标变量{name}.goal.upper", optional=True)
    tolerance = _number(data.get("tolerance"), f"目标变量{name}.goal.tolerance", optional=True)
    if enabled and mode in {"target", "at_least", "at_most"} and value is None:
        raise ValueError(f"目标变量{name}的{mode}模式需要value。")
    if enabled and mode == "range":
        if lower is None or upper is None or lower >= upper:
            raise ValueError(f"目标变量{name}的range模式需要lower < upper。")
    if tolerance is not None and tolerance <= 0:
        raise ValueError(f"目标变量{name}.goal.tolerance必须大于0。")
    return GoalSpec(enabled, mode, weight, value, lower, upper, tolerance)


def load_project_config(path: str | Path = DEFAULT_SETTINGS_PATH) -> ProjectConfig:
    """从YAML加载项目设置，并在训练或优化前尽早报告配置错误。"""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    root = _mapping(raw, "设置文件根节点")
    project = _mapping(root.get("project", {}), "project")

    inputs_raw = _mapping(root.get("input_variables"), "input_variables")
    inputs: list[InputVariable] = []
    for name, item_raw in inputs_raw.items():
        if name in RESERVED_COLUMNS:
            raise ValueError(f"输入变量名称{name}为系统保留列。")
        item = _mapping(item_raw, f"输入变量{name}")
        kind = str(item.get("type", "continuous")).lower()
        if kind not in {"continuous", "integer"}:
            raise ValueError(f"输入变量{name}.type只支持continuous或integer。")
        bounds_raw = _mapping(item.get("bounds", {}), f"输入变量{name}.bounds")
        bounds_enabled = bool(bounds_raw.get("enabled", True))
        lower = _number(bounds_raw.get("lower"), f"输入变量{name}.bounds.lower", optional=True)
        upper = _number(bounds_raw.get("upper"), f"输入变量{name}.bounds.upper", optional=True)
        margin = _number(bounds_raw.get("observed_margin", 0.0),
                         f"输入变量{name}.bounds.observed_margin")
        if margin < 0:
            raise ValueError(f"输入变量{name}.bounds.observed_margin不能为负。")
        if bounds_enabled:
            if lower is None or upper is None or lower >= upper:
                raise ValueError(f"输入变量{name}启用边界时需要lower < upper。")
            if kind == "integer" and (not lower.is_integer() or not upper.is_integer()):
                raise ValueError(f"整数变量{name}的显式上下界必须是整数。")
        period = _number(item.get("period"), f"输入变量{name}.period", optional=True)
        if period is not None and period <= 0:
            raise ValueError(f"输入变量{name}.period必须大于0。")
        inputs.append(InputVariable(str(name), kind, bool(item.get("enabled", True)),
                                    BoundsSpec(bounds_enabled, lower, upper, margin), period))

    targets_raw = _mapping(root.get("target_variables"), "target_variables")
    targets: list[TargetVariable] = []
    for name, item_raw in targets_raw.items():
        if name in RESERVED_COLUMNS or name in inputs_raw:
            raise ValueError(f"目标变量名称{name}重复或为系统保留列。")
        item = _mapping(item_raw, f"目标变量{name}")
        direction = str(item.get("pareto_direction", "maximize")).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"目标变量{name}.pareto_direction只支持maximize或minimize。")
        reference = _number(item.get("reference_value"),
                            f"目标变量{name}.reference_value", optional=True)
        targets.append(TargetVariable(str(name), bool(item.get("enabled", True)), direction,
                                      reference, _goal(str(name), item.get("goal"))))

    active_input_names = [item.name for item in inputs if item.enabled]
    active_target_names = [item.name for item in targets if item.enabled]
    if not active_input_names:
        raise ValueError("至少需要启用一个输入变量。")
    if not active_target_names:
        raise ValueError("至少需要启用一个目标变量。")

    constraints_raw = _mapping(root.get("constraints", {}), "constraints")
    constraints: list[LinearConstraint] = []
    rules = constraints_raw.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("constraints.rules必须是列表。")
    for index, rule_raw in enumerate(rules):
        rule = _mapping(rule_raw, f"constraints.rules[{index}]")
        coefficients_raw = _mapping(rule.get("coefficients", {}),
                                    f"constraints.rules[{index}].coefficients")
        coefficients = {str(key): _number(value, f"约束{index}系数{key}")
                        for key, value in coefficients_raw.items()}
        unknown = set(coefficients) - set(active_input_names)
        if unknown:
            raise ValueError(f"约束{index}引用了未启用或不存在的输入变量：{sorted(unknown)}")
        if not coefficients or all(value == 0 for value in coefficients.values()):
            raise ValueError(f"约束{index}至少需要一个非零系数。")
        operator = str(rule.get("operator", "<=")).strip()
        if operator not in {"<=", ">="}:
            raise ValueError(f"约束{index}.operator只支持<=或>=。")
        constraints.append(LinearConstraint(
            str(rule.get("name", f"constraint_{index + 1}")),
            bool(rule.get("enabled", True)), coefficients, operator,
            _number(rule.get("rhs"), f"约束{index}.rhs"),
        ))

    rec_raw = _mapping(root.get("recommendation", {}), "recommendation")
    mode = str(rec_raw.get("mode", "target")).lower()
    if mode not in {"target", "pareto"}:
        raise ValueError("recommendation.mode只支持target或pareto。")
    batch_size = int(rec_raw.get("batch_size", 5))
    pool_size = int(rec_raw.get("candidate_pool_size", 4096))
    if batch_size < 1 or pool_size < batch_size:
        raise ValueError("batch_size必须大于0，candidate_pool_size不能小于batch_size。")
    recommendation = RecommendationSpec(
        mode=mode,
        batch_size=batch_size,
        candidate_pool_size=pool_size,
        seed=int(rec_raw.get("seed", 42)),
        risk_aversion=_number(rec_raw.get("risk_aversion", 0.25), "risk_aversion"),
        diversity_min_distance=_number(rec_raw.get("diversity_min_distance", 0.05),
                                       "diversity_min_distance"),
        auto_reference_margin=_number(rec_raw.get("auto_reference_margin", 0.1),
                                      "auto_reference_margin"),
    )
    if recommendation.risk_aversion < 0 or recommendation.diversity_min_distance < 0:
        raise ValueError("risk_aversion和diversity_min_distance不能为负。")
    if recommendation.auto_reference_margin <= 0:
        raise ValueError("auto_reference_margin必须大于0。")
    # 训练时不需要先指定想要达到的目标；仅在执行推荐时检查goal。

    initial_size = int(project.get("initial_design_size", 20))
    if initial_size < 1:
        raise ValueError("project.initial_design_size必须大于0。")
    return ProjectConfig(
        name=str(project.get("name", source.stem)),
        initial_design_size=initial_size,
        input_variables=tuple(inputs),
        target_variables=tuple(targets),
        constraints_enabled=bool(constraints_raw.get("enabled", True)),
        constraints=tuple(constraints),
        recommendation=recommendation,
        source_path=source,
    )


def resolve_search_bounds(config: ProjectConfig, observed_x: torch.Tensor | None = None) -> torch.Tensor:
    """解析搜索边界；关闭显式边界时使用观测范围和用户设置的扩展比例。"""
    lowers: list[float] = []
    uppers: list[float] = []
    for column, variable in enumerate(config.active_inputs):
        spec = variable.bounds
        if spec.enabled:
            lower, upper = spec.lower, spec.upper
        else:
            if observed_x is None or len(observed_x) == 0:
                raise ValueError(
                    f"输入变量{variable.name}关闭了显式边界；生成初始设计前没有观测范围可用。"
                )
            minimum = float(observed_x[:, column].min())
            maximum = float(observed_x[:, column].max())
            span = maximum - minimum
            base = span if span > 0 else max(abs(minimum), 1.0)
            delta = spec.observed_margin * base
            lower, upper = minimum - delta, maximum + delta
            if variable.kind == "integer":
                lower, upper = math.floor(lower), math.ceil(upper)
        lowers.append(float(lower))
        uppers.append(float(upper))
    return torch.tensor([lowers, uppers], dtype=DTYPE)


def feasible(config: ProjectConfig, x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """判断一批候选输入是否满足当前边界、整数要求和已启用线性约束。"""
    values = x if x.ndim == 2 else x.unsqueeze(0)
    eps = 1e-6  # 数值容差，防止浮点误差把边界点误判为不可行。
    mask = ((values >= bounds[0] - eps).all(-1)
            & (values <= bounds[1] + eps).all(-1))
    for index in config.integer_indices:
        mask &= torch.isclose(values[:, index], values[:, index].round(), atol=eps, rtol=0)
    if config.constraints_enabled:
        name_to_index = {name: i for i, name in enumerate(config.input_names)}
        for rule in config.constraints:
            if not rule.enabled:
                continue
            total = sum(rule.coefficients[name] * values[:, name_to_index[name]]
                        for name in rule.coefficients)
            mask &= total <= rule.rhs + eps if rule.operator == "<=" else total >= rule.rhs - eps
    return mask


def target_signs(config: ProjectConfig) -> torch.Tensor:
    """BoTorch按最大化处理目标；最小化目标乘以-1后进入Pareto计算。"""
    return torch.tensor([
        1.0 if item.pareto_direction == "maximize" else -1.0
        for item in config.active_targets
    ], dtype=DTYPE)


def resolve_reference_point(config: ProjectConfig, y: torch.Tensor) -> torch.Tensor:
    """返回最大化空间中的参考点；未设置时由当前观测最差值及余量生成。"""
    signs = target_signs(config)
    transformed = y * signs
    result: list[float] = []
    for column, target in enumerate(config.active_targets):
        if target.reference_value is not None:
            result.append(target.reference_value * float(signs[column]))
            continue
        values = transformed[:, column]
        span = float(values.max() - values.min())
        base = span if span > 0 else max(abs(float(values.min())), 1.0)
        result.append(float(values.min()) - config.recommendation.auto_reference_margin * base)
    return torch.tensor(result, dtype=DTYPE)


def schema_signature(config: ProjectConfig) -> dict[str, Any]:
    """模型存档使用的变量结构；推荐时防止拿错项目或列顺序。"""
    signature = {
        "input_names": config.input_names,
        "input_types": [item.kind for item in config.active_inputs],
        "target_names": config.target_names,
    }
    if any(item.period is not None for item in config.active_inputs):
        signature["input_periods"] = [item.period for item in config.active_inputs]
    return signature
