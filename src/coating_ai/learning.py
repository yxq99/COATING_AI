"""动态维度高斯过程训练、预测与留一交叉验证。"""
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import ProjectConfig, resolve_search_bounds


def build_model(x: torch.Tensor, y: torch.Tensor, model_bounds: torch.Tensor) -> SingleTaskGP:
    """按当前输入维度和目标数量，为各目标创建相互独立的批量GP。"""
    input_count = x.shape[1]
    target_count = y.shape[1]
    # SingleTaskGP的单目标没有输出批次轴；强行使用[1]会使拟合维度不一致。
    batch_shape = torch.Size([target_count]) if target_count > 1 else torch.Size()
    kernel = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=input_count, batch_shape=batch_shape),
        batch_shape=batch_shape,
    )
    return SingleTaskGP(
        train_X=x,
        train_Y=y,
        covar_module=kernel,
        input_transform=Normalize(d=input_count, bounds=model_bounds),
        outcome_transform=Standardize(m=target_count),
    )


def fit(x: torch.Tensor, y: torch.Tensor, model_bounds: torch.Tensor) -> SingleTaskGP:
    """最大化边缘似然，学习长度尺度、信号强度和观测噪声。"""
    model = build_model(x, y, model_bounds)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def predict(model: SingleTaskGP, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回原始目标单位下的后验均值和潜在函数标准差。"""
    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean
        std = posterior.variance.clamp_min(0).sqrt()
    return mean, std


def cross_validate(
    config: ProjectConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    model_bounds: torch.Tensor,
    groups=None,
) -> pd.DataFrame:
    """留组验证；同源轨迹同折，观测范围及标准化仅从训练折拟合。"""
    predictions = torch.empty_like(y)
    labels = list(range(len(x))) if groups is None else list(groups)
    if len(labels) != len(x) or pd.isna(labels).any():
        raise ValueError("validation_group必须完整且与样本数量一致。")
    unique = list(dict.fromkeys(labels))
    if len(unique) < 2:
        raise ValueError("至少需要两个独立验证组。")
    for number, label in enumerate(unique, 1):
        print(f"留组验证 {number}/{len(unique)}", flush=True)
        held = torch.tensor([item == label for item in labels])
        mask = ~held
        if int(mask.sum()) < 2:
            raise ValueError("每一折至少需要两条训练记录。")
        fold_bounds = resolve_search_bounds(config, x[mask])
        model = fit(x[mask], y[mask], fold_bounds)
        predictions[held] = predict(model, x[held])[0]
    error = predictions - y
    total = ((y - y.mean(0)) ** 2).sum(0)
    r2 = torch.where(total > 0, 1 - error.square().sum(0) / total, torch.nan)
    return pd.DataFrame({
        "target_variable": config.target_names,
        "MAE": error.abs().mean(0).cpu().numpy(),
        "RMSE": error.square().mean(0).sqrt().cpu().numpy(),
        "R2": r2.cpu().numpy(),
        "validation_folds": len(unique),
    })
