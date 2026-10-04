"""动态维度高斯过程训练、预测与留一交叉验证。"""
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import ProjectConfig


def build_model(x: torch.Tensor, y: torch.Tensor, model_bounds: torch.Tensor) -> SingleTaskGP:
    """按当前输入维度和目标数量，为各目标创建相互独立的批量GP。"""
    input_count = x.shape[1]
    target_count = y.shape[1]
    batch_shape = torch.Size([target_count])
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
) -> pd.DataFrame:
    """逐样本留一验证；每一折重新拟合标准化和模型，避免目标信息泄漏。"""
    predictions = torch.empty_like(y)
    for index in range(len(x)):
        print(f"留一验证 {index + 1}/{len(x)}", flush=True)
        mask = torch.arange(len(x)) != index
        model = fit(x[mask], y[mask], model_bounds)
        predictions[index] = predict(model, x[index:index + 1])[0][0]
    error = predictions - y
    total = ((y - y.mean(0)) ** 2).sum(0)
    r2 = torch.where(total > 0, 1 - error.square().sum(0) / total, torch.nan)
    return pd.DataFrame({
        "target_variable": config.target_names,
        "MAE": error.abs().mean(0).cpu().numpy(),
        "RMSE": error.square().mean(0).sqrt().cpu().numpy(),
        "R2": r2.cpu().numpy(),
    })
