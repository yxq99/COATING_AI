"""项目的唯一配置源：路径、变量、边界、目标和物理约束。"""
from pathlib import Path #Path 是一个面向对象的文件路径类。它把路径从“纯字符串”变成了“有行为的对象”

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]  #获取当前文件路径.解析为绝对路径.向上回溯父目录[2]级
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INITIAL_DESIGN_PATH = DATA_DIR / "initial_design.csv"

DTYPE = torch.double  #定义kind数据类型
FEATURES = ["mxene_wt", "bta_ldh_wt", "phen_wt", "pda_nm", "film_um"]
TARGETS = ["log_z", "adhesion_score", "healing_pct"]
#两个数组，第一个数组为下界，第二个数组为上界
BOUNDS = torch.tensor([[0.5, 1, 0.2, 5, 30], [5, 8, 2, 60, 120]], dtype=DTYPE)
# 所有目标均最大化；HV始终在原始目标单位中计算，参考点跨轮次固定。
REF_POINT = torch.tensor([5.5, -0.5, -5.0], dtype=DTYPE)


""""===================================边界条件======================================="""
def feasible(x):
    """可行性约束（Feasibility Constraints）————用于判断一个或一批候选设计点 x 是否满足所有物理和工程限制条件"""
    eps = 1e-6   # 数值容差，防止浮点误差导致边界点被误判为不可行
    return ((x >= BOUNDS[0] - eps).all(-1)       # ① 下界检查
            & (x <= BOUNDS[1] + eps).all(-1)     # ② 上界检查
            & (x[:, :3].sum(-1) <= 12 + eps)     # ③ 线性约束：三种活性组分总重量占比 ≤ 12%
            & (x[:, 1] <= 6 * x[:, 2] + eps))    # ④ 比例约束：bta_ldh_wt ≤ 6 × phen_wt


def constraints():
    """"BoTorch 框架中定义线性不等式约束的标准接口"""
    # 每条约束 = (indices, coefficients, rhs)
    # 语义: sum(coefficients[i] * x[indices[i]]) >= rhs
    return [(torch.tensor([0, 1, 2]), torch.tensor([-1., -1., -1.], dtype=DTYPE), -12.),
            (torch.tensor([1, 2]), torch.tensor([-1., 6.], dtype=DTYPE), 0.)]
