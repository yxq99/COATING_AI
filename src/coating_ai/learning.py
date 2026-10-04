"""动态维度高斯过程训练、预测与留一交叉验证。"""
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import ProjectConfig, resolve_search_bounds


class AngularInputTransform(InputTransform):
    """周期输入转为sin/cos，其余输入按训练边界缩放；外部仍使用度数。"""

    def __init__(self, bounds: torch.Tensor, periods: list[float | None]):
        super().__init__()
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.periods = periods
        self.register_buffer("bounds", bounds.clone())

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        columns = []
        for index, period in enumerate(self.periods):
            if period is None:
                scale = (self.bounds[1, index] - self.bounds[0, index]).clamp_min(1e-8)
                columns.append((X[..., index] - self.bounds[0, index]) / scale)
            else:
                radians = X[..., index] * (2 * torch.pi / period)
                columns.extend(((radians.sin() + 1) / 2, (radians.cos() + 1) / 2))
        return torch.stack(columns, dim=-1)


def build_model(x: torch.Tensor, y: torch.Tensor, model_bounds: torch.Tensor,
                config: ProjectConfig | None = None) -> SingleTaskGP:
    """按当前输入维度和目标数量，为各目标创建相互独立的批量GP。"""
    input_count = x.shape[1]
    periods = [item.period for item in config.active_inputs] if config else [None] * input_count
    if any(period is not None for period in periods):
        transform = AngularInputTransform(model_bounds, periods)
        input_count += sum(period is not None for period in periods)
    else:
        transform = Normalize(d=input_count, bounds=model_bounds)
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
        input_transform=transform,
        outcome_transform=Standardize(m=target_count),
    )


def fit(x: torch.Tensor, y: torch.Tensor, model_bounds: torch.Tensor,
        config: ProjectConfig | None = None) -> SingleTaskGP:
    """最大化边缘似然，学习长度尺度、信号强度和观测噪声。"""
    model = build_model(x, y, model_bounds, config)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def predict(model: SingleTaskGP, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回原始目标单位下的后验均值和潜在函数标准差。"""
    with torch.no_grad():
        means, deviations = [], []
        for chunk in x.split(256):
            posterior = model.posterior(chunk)
            means.append(posterior.mean)
            deviations.append(posterior.variance.clamp_min(0).sqrt())
    return torch.cat(means), torch.cat(deviations)


def cross_validate(
    config: ProjectConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    model_bounds: torch.Tensor,
) -> pd.DataFrame:
    """逐样本留一验证；每一折重新拟合标准化和模型，避免目标信息泄漏。"""
    predictions = torch.empty_like(y)
    baseline_predictions = torch.empty_like(y)
    for index in range(len(x)):
        print(f"留一验证 {index + 1}/{len(x)}", flush=True)
        mask = torch.arange(len(x)) != index
        baseline_predictions[index] = y[mask].mean(0)
        fold_bounds = resolve_search_bounds(config, x[mask])
        model = fit(x[mask], y[mask], fold_bounds, config)
        predictions[index] = predict(model, x[index:index + 1])[0][0]
    error = predictions - y
    total = ((y - y.mean(0)) ** 2).sum(0)
    r2 = torch.where(total > 0, 1 - error.square().sum(0) / total, torch.nan)
    return pd.DataFrame({
        "target_variable": config.target_names,
        "MAE": error.abs().mean(0).cpu().numpy(),
        "RMSE": error.square().mean(0).sqrt().cpu().numpy(),
        "R2": r2.cpu().numpy(),
        "mean_baseline_RMSE": (baseline_predictions - y).square().mean(0).sqrt().cpu().numpy(),
    })
