"""模型训练与验证：供主动学习闭环反复调用。"""
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import BOUNDS, TARGETS


def fit(x, y):
    # 显式指定三输出独立核，避免依赖库默认核函数。
    batch = torch.Size([3])
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=5, batch_shape=batch),
                         batch_shape=batch)
    model = SingleTaskGP(x, y, covar_module=kernel,
                         input_transform=Normalize(d=5, bounds=BOUNDS),
                         outcome_transform=Standardize(m=3))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def cross_validate(x, y):
    """逐配方留一验证；每一折重新拟合标准化和模型，避免数据泄漏。"""
    predictions = torch.empty_like(y)
    for i in range(len(x)):
        print(f"留一验证 {i+1}/{len(x)}", flush=True)
        mask = torch.arange(len(x)) != i
        model = fit(x[mask], y[mask])
        with torch.no_grad():
            predictions[i] = model.posterior(x[i:i+1]).mean[0]
    err = predictions - y
    total = ((y - y.mean(0))**2).sum(0)
    r2 = torch.where(total > 0, 1 - err.square().sum(0)/total, torch.nan)
    # 避免附着力为0时MAPE无定义；使用原始单位的MAE/RMSE。
    metrics = pd.DataFrame({"target": TARGETS, "MAE": err.abs().mean(0).numpy(),
                            "RMSE": err.square().mean(0).sqrt().numpy(), "R2": r2.numpy()})
    return metrics
