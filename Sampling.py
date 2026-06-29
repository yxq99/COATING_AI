import numpy as np
import pandas as pd
from scipy.stats import qmc

# =====================================================================
# [ USER_CONFIG ] 人类实验者全局配置区 (在今后的DOE迭代中，你只需要修改这里)
# =====================================================================

# 1. 实验规模与随机种子
N_TARGET = 20  # 初始采样目标点数 n0
RANDOM_SEED = 42  # 随机种子(固定此数字，每次跑出来的配方表100%完全一致，确保论文可复现)
OVERSAMPLE_FACTOR = 30  # 拟蒙特卡洛超额抽样倍率 (20 * 30 = 600个备选池)

# 2. 本地文件输出开关
SAVE_TO_EXCEL = True  # 是否在当前文件夹下物理生成 .xlsx 表格
EXCEL_FILENAME = "MXene_Initial_DOE_20.xlsx"

# 3. 输入空间 5 维特征连续边界 [下界 lb, 上界 ub] (单位对应配方体系)
# 特征顺序: [x1: MXene(wt%), x2: BTA@LDH(wt%), x3: Phen(wt%), x4: PDA厚度(nm), x5: 膜厚(um)]
PARAM_BOUNDS_LB = [0.5, 1.0, 0.2, 5.0, 30.0]
PARAM_BOUNDS_UB = [5.0, 8.0, 2.0, 60.0, 120.0]

FEATURE_NAMES = [
    "x1_MXene_wt%",
    "x2_BTA@LDH_wt%",
    "x3_Phen_wt%",
    "x4_PDA_thickness_nm",
    "x5_Film_thickness_um"
]

# 4. 实验室物理可行性约束红线 (见方案 定义1.1)
MAX_TOTAL_FILLER = 12.0  # 约束 g1: 体系总填料含量上限 (x1 + x2 + x3 <= 12 wt%)
MAX_BTA_PHEN_RATIO = 6.0  # 约束 g2: BTA@LDH 与 Phen 浓度比例上限 (x2 / x3 <= 6)


# =====================================================================
# [ CORE_ENGINE ] 底层数学抽样与过滤引擎 (请勿修改以下逻辑)
# =====================================================================

def run_sobol_sampling_pipeline():
    lb = np.array(PARAM_BOUNDS_LB, dtype=float)
    ub = np.array(PARAM_BOUNDS_UB, dtype=float)
    dim = len(FEATURE_NAMES)

    # 实例化拟随机低差异引擎
    sobol_engine = qmc.Sobol(d=dim, scramble=True, seed=RANDOM_SEED)
    raw_cube_points = sobol_engine.random_base2(m=10)
    physical_points = qmc.scale(raw_cube_points, lb, ub)

    qualified_pool = []

    # 物理红线严密过筛
    for pt in physical_points:
        x1, x2, x3, x4, x5 = pt
        g1_violation = (x1 + x2 + x3) - MAX_TOTAL_FILLER
        g2_violation = (x2 / x3) - MAX_BTA_PHEN_RATIO

        if g1_violation <= 1e-6 and g2_violation <= 1e-6:
            qualified_pool.append(pt)

        if len(qualified_pool) == N_TARGET:
            break

    if len(qualified_pool) < N_TARGET:
        raise RuntimeError(
            f"红线过严：备选池生成了 {len(raw_cube_points)} 个点，仅 {len(qualified_pool)} 个合格。请放宽约束或调大 OVERSAMPLE_FACTOR。")

    # 工程精度洗练
    df_result = pd.DataFrame(qualified_pool, columns=FEATURE_NAMES)
    df_result["x1_MXene_wt%"] = df_result["x1_MXene_wt%"].round(2)
    df_result["x2_BTA@LDH_wt%"] = df_result["x2_BTA@LDH_wt%"].round(2)
    df_result["x3_Phen_wt%"] = df_result["x3_Phen_wt%"].round(2)
    df_result["x4_PDA_thickness_nm"] = df_result["x4_PDA_thickness_nm"].round(1)
    df_result["x5_Film_thickness_um"] = df_result["x5_Film_thickness_um"].round(0).astype(int)

    df_result.index = [f"DOE_Sample_{i + 1:02d}" for i in range(N_TARGET)]
    return df_result


if __name__ == "__main__":
    print(f"[*] 正在启动 Sobol DOE 抽样引擎 (Target: {N_TARGET} points)...")
    doe_plan = run_sobol_sampling_pipeline()

    print("\n" + "=" * 55)
    print("      Phen/BTA-LDH-PDA MXene 初始配方表")
    print("=" * 55)
    print(doe_plan)

    if SAVE_TO_EXCEL:
        try:
            doe_plan.to_excel(EXCEL_FILENAME)
            print(f"\n[+] 物理本地表格已成功导出至当前目录: < {EXCEL_FILENAME} >")
        except Exception as err:
            print(f"\n[-] Excel导出失败 (可能缺失 openpyxl 库): {err}")
            print("    提示: 可在终端执行 `pip install openpyxl` 后重试。")