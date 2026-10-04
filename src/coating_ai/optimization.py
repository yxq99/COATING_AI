"""Pareto多目标优化与指定目标反向推荐。"""
from __future__ import annotations

import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from .config import (
    DTYPE,
    ProjectConfig,
    resolve_reference_point,
    target_signs,
)
from .data import candidate_pool
from .learning import predict


def pareto_summary(
    config: ProjectConfig,
    y: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """按每个目标的最大化/最小化方向返回非支配点、HV和实际参考点。"""
    transformed = y * target_signs(config)
    reference = resolve_reference_point(config, y)
    mask = is_non_dominated(transformed)
    hv = Hypervolume(reference).compute(transformed[mask])
    return mask, hv, reference


def _prediction_report(
    config: ProjectConfig,
    candidates: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> pd.DataFrame:
    table = pd.DataFrame(candidates.cpu().numpy(), columns=config.input_names)
    for index in config.integer_indices:
        table[config.input_names[index]] = table[config.input_names[index]].round().astype(int)
    table.insert(0, "candidate_id", [f"candidate_{i + 1:02d}" for i in range(len(table))])
    for column, name in enumerate(config.target_names):
        table[f"{name}_predicted"] = mean[:, column].cpu().numpy()
        table[f"{name}_latent_std"] = std[:, column].cpu().numpy()
    return table


def recommend_pareto(
    config: ProjectConfig,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    search_bounds: torch.Tensor,
) -> tuple[torch.Tensor, pd.DataFrame, dict]:
    """在动态候选池中用qLogNEHVI顺序选择一批Pareto改进候选。"""
    choices = candidate_pool(
        config, search_bounds, config.recommendation.candidate_pool_size,
        config.recommendation.seed,
    )
    reference = resolve_reference_point(config, train_y)
    objective = WeightedMCMultiOutputObjective(weights=target_signs(config))
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference.tolist(),
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        objective=objective,
        prune_baseline=True,
    )
    candidates, acquisition_values = optimize_acqf_discrete(
        acq_function=acquisition,
        q=config.recommendation.batch_size,
        choices=choices,
        unique=True,
        X_avoid=train_x,
        max_batch_size=1024,
    )
    candidates = candidates.detach()
    mean, std = predict(model, candidates)
    report = _prediction_report(config, candidates, mean, std)
    report["recommendation_mode"] = "pareto"
    metadata = {
        "reference_point_in_maximization_space": reference.tolist(),
        "acquisition_value": torch.as_tensor(acquisition_values).detach().cpu().tolist(),
    }
    return candidates, report, metadata


def _target_scores(
    config: ProjectConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    train_y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算目标偏差与不确定性惩罚；分数越小越符合用户目标。"""
    observed_scale = train_y.std(dim=0, unbiased=False)
    fallback = train_y.abs().mean(0).clamp_min(1.0) * 0.1
    observed_scale = torch.where(observed_scale > 1e-9, observed_scale, fallback)
    total = torch.zeros(len(mean), dtype=DTYPE)
    components: dict[str, torch.Tensor] = {}
    total_weight = 0.0

    for column, target in enumerate(config.active_targets):
        goal = target.goal
        if not goal.enabled or goal.mode == "ignore":
            continue
        scale = torch.tensor(goal.tolerance or float(observed_scale[column]), dtype=DTYPE)
        prediction = mean[:, column]
        if goal.mode == "target":
            error = (prediction - goal.value).abs() / scale
        elif goal.mode == "range":
            range_scale = goal.tolerance or (goal.upper - goal.lower)
            error = (torch.relu(goal.lower - prediction) + torch.relu(prediction - goal.upper)) / range_scale
            scale = torch.tensor(range_scale, dtype=DTYPE)
        elif goal.mode == "at_least":
            error = torch.relu(goal.value - prediction) / scale
        elif goal.mode == "at_most":
            error = torch.relu(prediction - goal.value) / scale
        elif goal.mode == "maximize":
            span = (prediction.max() - prediction.min()).clamp_min(1e-9)
            error = (prediction.max() - prediction) / span
        elif goal.mode == "minimize":
            span = (prediction.max() - prediction.min()).clamp_min(1e-9)
            error = (prediction - prediction.min()) / span
        else:  # 配置加载阶段已验证，不应到达。
            raise RuntimeError(f"未知目标模式：{goal.mode}")
        uncertainty = std[:, column] / scale.clamp_min(1e-9)
        weighted = goal.weight * (error + config.recommendation.risk_aversion * uncertainty)
        total += weighted
        total_weight += goal.weight
        components[f"{target.name}_normalized_error"] = error
    return total / total_weight, components


def _select_diverse(
    scores: torch.Tensor,
    pool: torch.Tensor,
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    count: int,
    minimum_distance: float,
) -> torch.Tensor:
    """按目标分数从优到劣选择，并在归一化输入空间保持批次多样性。"""
    scale = (bounds[1] - bounds[0]).clamp_min(1e-12)
    normalized = (pool - bounds[0]) / scale
    train_normalized = (train_x - bounds[0]) / scale
    order = torch.argsort(scores)
    selected: list[int] = []
    deferred: list[int] = []
    for raw_index in order.tolist():
        distances_to_training = torch.linalg.vector_norm(
            train_normalized - normalized[raw_index], dim=1
        )
        if torch.any(distances_to_training < 1e-9):
            continue
        if selected:
            distances = torch.linalg.vector_norm(
                normalized[selected] - normalized[raw_index], dim=1
            )
            if torch.any(distances < minimum_distance):
                deferred.append(raw_index)
                continue
        selected.append(raw_index)
        if len(selected) == count:
            break
    if len(selected) < count:
        for raw_index in deferred:
            if raw_index not in selected:
                selected.append(raw_index)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError("去除已实验点并应用多样性要求后，候选数量不足。")
    return torch.tensor(selected, dtype=torch.long)


def recommend_target(
    config: ProjectConfig,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    search_bounds: torch.Tensor,
) -> tuple[torch.Tensor, pd.DataFrame, dict]:
    """根据target/range/阈值等设置，反向推荐最符合目标的输入变量。"""
    if not any(t.goal.enabled and t.goal.mode != "ignore" for t in config.active_targets):
        raise ValueError("请先在设置文件中启用至少一个goal，并填写你希望达到的目标值。")
    pool = candidate_pool(
        config, search_bounds, config.recommendation.candidate_pool_size,
        config.recommendation.seed,
    )
    mean, std = predict(model, pool)
    scores, components = _target_scores(config, mean, std, train_y)
    indices = _select_diverse(
        scores, pool, train_x, search_bounds,
        config.recommendation.batch_size,
        config.recommendation.diversity_min_distance,
    )
    candidates = pool[indices]
    report = _prediction_report(config, candidates, mean[indices], std[indices])
    report["target_match_score"] = scores[indices].cpu().numpy()
    for name, values in components.items():
        report[name] = values[indices].cpu().numpy()
    checks = []
    for column, target in enumerate(config.active_targets):
        goal = target.goal
        if not goal.enabled or goal.mode == "ignore":
            continue
        values = mean[indices, column]
        if goal.mode == "target" and goal.tolerance is not None:
            matched = (values - goal.value).abs() <= goal.tolerance
        elif goal.mode == "range":
            matched = (values >= goal.lower) & (values <= goal.upper)
        elif goal.mode == "at_least":
            matched = values >= goal.value
        elif goal.mode == "at_most":
            matched = values <= goal.value
        else:
            continue  # 最大/最小化或没有容差的点目标，没有“达标”的二元定义。
        report[f"{target.name}_predicted_condition_met"] = matched.cpu().numpy()
        checks.append(matched)
    if checks:
        report["all_evaluable_conditions_met"] = torch.stack(checks).all(0).cpu().numpy()
    outside = ((candidates < train_x.min(0).values - 1e-8)
               | (candidates > train_x.max(0).values + 1e-8)).any(-1)
    report["outside_observed_input_range"] = outside.cpu().numpy()
    report["recommendation_mode"] = "target"
    return candidates, report, {"score_definition": "lower_is_better"}


def recommend_candidates(
    config: ProjectConfig,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    search_bounds: torch.Tensor,
) -> tuple[torch.Tensor, pd.DataFrame, dict]:
    """由设置文件中的recommendation.mode选择推荐策略。"""
    if config.recommendation.mode == "pareto":
        return recommend_pareto(config, model, train_x, train_y, search_bounds)
    return recommend_target(config, model, train_x, train_y, search_bounds)
