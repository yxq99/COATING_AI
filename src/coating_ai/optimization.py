"""贝叶斯优化：用模型推荐下一批实验。"""
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from .config import BOUNDS, REF_POINT, constraints, feasible


def suggest(model, x, batch_size=5):
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=REF_POINT.tolist(), X_baseline=x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        prune_baseline=True)
    candidates, _ = optimize_acqf(acq, bounds=BOUNDS, q=batch_size,
                                  num_restarts=10, raw_samples=128,
                                  inequality_constraints=constraints(),
                                  sequential=True, options={"maxiter": 150})
    candidates = candidates.detach()
    if not feasible(candidates).all():
        raise ValueError("优化器返回不可行配方，请检查约束与收敛提示。")
    return candidates


def pareto_hv(y):
    mask = is_non_dominated(y)
    return mask, Hypervolume(REF_POINT).compute(y[mask])
