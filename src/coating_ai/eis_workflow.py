"""主工作簿到统一ML表；缺测保持空白，不生成实验结果。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from .data import save_csv

INPUTS = ["hybrid_wt_pct", "ldh_mxene_ratio"]
DAYS = [1, 3, 7, 14, 28, 50]
KEYS = ["sample_id", "preparation_id", "specimen_id", "day"]
MEASURE_COLUMNS = KEYS + ["z_001_ohm", "area_cm2", "phase_001_deg", "source_file", "notes"]


def write_measurement_templates(candidates: pd.DataFrame, folder: Path) -> None:
    """为推荐配方生成空白后续测量和单条全谱格式模板。"""
    save_csv(candidates, folder / "candidates_to_measure.csv")
    rows = []
    for sample_id in candidates.sample_id:
        for rep in range(1, 4):
            for day in DAYS:
                rows.append([sample_id, f"R{rep}", f"{sample_id}_R{rep}_S1", day,
                             None, None, None, None, "填写独立制备的实际测量；空白不是0"])
    save_csv(pd.DataFrame(rows, columns=MEASURE_COLUMNS), folder / "measurements.csv")
    spectra = [[None, None, None, None, float(f), None, None]
               for f in np.logspace(5, -2, 71)]
    save_csv(pd.DataFrame(spectra, columns=KEYS+["frequency_hz", "z_real_ohm", "z_imag_ohm"]),
             folder / "spectra.csv")


def _combine_workbook(workbook: Path, new_data):
    forms = pd.read_excel(workbook, sheet_name="配方")
    low = pd.read_excel(workbook, sheet_name="EIS低频")
    spectral = pd.read_excel(workbook, sheet_name="EIS全谱")
    extra = pd.read_excel(workbook, sheet_name="新谱模板")
    spectral = pd.concat([spectral, extra], ignore_index=True)
    if new_data:
        folder = Path(new_data)
        forms = pd.concat([forms, pd.read_csv(folder/"candidates_to_measure.csv")], ignore_index=True)
        low = pd.concat([low, pd.read_csv(folder/"measurements.csv")], ignore_index=True)
        if (folder/"spectra.csv").exists():
            spectral = pd.concat([spectral, pd.read_csv(folder/"spectra.csv")], ignore_index=True)
    if forms.sample_id.isna().any() or forms.sample_id.duplicated().any():
        raise ValueError("配方sample_id不能为空或重复；同一配方的重复放在EIS表，不另造配方行。")
    for name in INPUTS:
        forms[name] = pd.to_numeric(forms[name], errors="raise")
    partial = forms[INPUTS].notna().any(axis=1) & ~forms[INPUTS].notna().all(axis=1)
    if partial.any():
        raise ValueError(f"请填齐配方输入：{forms.loc[partial,'sample_id'].tolist()}")
    filled = forms[INPUTS].notna().all(axis=1)
    if not np.isfinite(forms.loc[filled, INPUTS].to_numpy(dtype=float)).all():
        raise ValueError("配方输入不能是无穷或无效数字。")
    forms["validation_group"] = forms.get("validation_group", forms.sample_id).fillna(forms.sample_id)
    forms.loc[forms.validation_group.astype(str).str.strip()=="", "validation_group"] = forms.sample_id

    # 可直接填低频表，也可填写新谱模板的实部/虚部；缺测不做外推。
    measured_spectra = spectral[["z_real_ohm", "z_imag_ohm"]].notna().any(axis=1)
    for _, block in spectral.loc[measured_spectra].groupby(KEYS, dropna=False):
        if block[KEYS+["frequency_hz", "z_real_ohm", "z_imag_ohm"]].isna().any().any():
            raise ValueError("已填写的全谱行必须有样品、制备、试片、天数、频率及完整实虚部。")
        at = block.loc[np.isclose(block.frequency_hz.astype(float), 0.01, rtol=1e-5)]
        if len(at) != 1:
            raise ValueError("每条待提取的全谱需恰好一个0.01Hz点；不将最近频率冒充0.01Hz。")
        row = at.iloc[0]
        matched = np.ones(len(low), dtype=bool)
        for key in KEYS:
            matched &= low[key].eq(row[key]).to_numpy()
        if matched.sum() != 1:
            raise ValueError("全谱必须在EIS低频表中有唯一对应记录，并填写实际面积。")
        idx = low.index[matched][0]
        z = float(np.hypot(row.z_real_ohm, row.z_imag_ohm))
        if pd.isna(low.at[idx, "z_001_ohm"]):
            low.at[idx, "z_001_ohm"] = z
            low.at[idx, "phase_001_deg"] = float(np.degrees(np.arctan2(row.z_imag_ohm,row.z_real_ohm)))
        elif not np.isclose(float(low.at[idx,"z_001_ohm"]), z, rtol=1e-5):
            raise ValueError("低频表与全谱的0.01Hz模值冲突，请同步更新或移除过期全谱。")
    return forms, low


def prepare_data(workbook: Path, data_dir: Path, new_data=None) -> dict:
    forms, low = _combine_workbook(workbook, new_data)
    params = pd.read_excel(workbook, sheet_name="参数").set_index("parameter")["value"].to_dict()
    target_day = float(params["target_day"])
    if target_day not in (14,28,50):
        raise ValueError("当前模板后期目标日只支持14、28或50，且不早于7天预测截点。")
    target_day = int(target_day)
    target = f"logZ_{target_day}d"
    threshold = float(params["minimum_logZ"])
    measured = low.z_001_ohm.notna()
    filled = forms[INPUTS].notna().all(axis=1)
    valid_ids = set(forms.loc[filled,"sample_id"])
    observed = low.loc[measured].copy()
    if observed[KEYS+["area_cm2"]].isna().any().any():
        raise ValueError("有阻抗的记录必须填齐样品、制备、试片、天数及面积；未测记录留空。")
    if not set(observed.sample_id).issubset(valid_ids):
        raise ValueError("已有EIS记录的样品必须先在配方表填齐两个连续输入。")
    if observed.duplicated(KEYS).any():
        raise ValueError("同一试片同一天有重复记录，请先确认正式版本，不自动挑最好的一次。")
    for name in ("day", "z_001_ohm", "area_cm2"):
        observed[name] = pd.to_numeric(observed[name], errors="raise")
        if not np.isfinite(observed[name]).all() or (observed[name] <= 0).any():
            raise ValueError(f"{name}必须是正的有限数，空白不能填成0。")
    observed["logZ"] = np.log10(observed.z_001_ohm*observed.area_cm2)
    # 先聚合同一次配制内的技术试片，再统计独立配制；n=1时SD保持空白。
    preparations = observed.groupby(["sample_id","preparation_id","day"], as_index=False).logZ.mean()
    stats = preparations.groupby(["sample_id","day"]).logZ.agg(
        mean_logZ="mean", sd_logZ="std", n_preparations="count").reset_index()
    stats["meets_target_by_mean"] = np.where(stats.day.eq(target_day),
                                            stats.mean_logZ.ge(threshold), None)
    wide = stats.pivot(index="sample_id", columns="day", values="mean_logZ")
    design, early, pending_early, pending_design = [], [], [], []
    def value(sample, day):
        return wide.at[sample,day] if sample in wide.index and day in wide.columns else np.nan
    for _, row in forms.loc[filled].iterrows():
        base = {"sample_id": row.sample_id, **{n:float(row[n]) for n in INPUTS},
                "validation_group":str(row.validation_group)}
        y = value(row.sample_id,target_day)
        early_values = {f"logZ_{d}d": value(row.sample_id,d) for d in (1,3,7)}
        if pd.notna(y):
            design.append({**base,target:float(y)})
        else:
            pending_design.append(base)
        if all(pd.notna(v) for v in early_values.values()):
            features = {**base,**early_values}
            if pd.notna(y):
                early.append({**features,target:float(y)})
            else:
                pending_early.append(features)
    data_dir.mkdir(parents=True,exist_ok=True)
    tables = {
        "design_training.csv": pd.DataFrame(design,columns=["sample_id",*INPUTS,"validation_group",target]),
        "early_training.csv": pd.DataFrame(early,columns=["sample_id",*INPUTS,"validation_group",
                                                       "logZ_1d","logZ_3d","logZ_7d",target]),
        "early_prediction_inputs.csv": pd.DataFrame(pending_early,columns=["sample_id",*INPUTS,
                                          "validation_group","logZ_1d","logZ_3d","logZ_7d"]),
        "design_prediction_inputs.csv": pd.DataFrame(pending_design,columns=["sample_id",*INPUTS,"validation_group"]),
        "repeat_statistics.csv":stats,
    }
    for filename,table in tables.items():
        table.to_csv(data_dir/filename,index=False,encoding="utf-8-sig")
    def input_spec(lower,upper):
        return {"type":"continuous","enabled":True,
                "bounds":{"enabled":True,"lower":float(lower),"upper":float(upper),"observed_margin":0.1}}
    variables={INPUTS[0]:input_spec(params["hybrid_lower"],params["hybrid_upper"]),
               INPUTS[1]:input_spec(params["ratio_lower"],params["ratio_upper"])}
    config={"project":{"name":"coating_durability_design","task":"design","initial_design_size":12},
            "input_variables":variables,
            "target_variables":{target:{"enabled":True,"pareto_direction":"maximize","reference_value":6.0,
                "goal":{"enabled":True,"mode":"at_least","value":threshold,"tolerance":0.3,"weight":1.0}}},
            "constraints":{"enabled":False,"rules":[]},
            "recommendation":{"mode":"target","batch_size":3,"candidate_pool_size":512,"seed":42,
                "risk_aversion":float(params["risk_aversion"]),"diversity_min_distance":0.05,"auto_reference_margin":0.1}}
    (data_dir/"project.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
    config["project"]={"name":"coating_durability_early","task":"prediction","initial_design_size":12}
    config["input_variables"]={**variables,**{f"logZ_{d}d":{"type":"continuous","enabled":True,
                     "bounds":{"enabled":False,"observed_margin":0.1}} for d in (1,3,7)}}
    (data_dir/"early_model.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
    summary={"design_rows":len(design),"early_rows":len(early),"pending_early_rows":len(pending_early),
             "measured_records":len(observed),"target_day":target_day,"threshold_logZ":threshold,
             "probability_model":"未实现；当前只报告均值、潜在标准差和实测重复统计，不提供可靠概率",
             "derived_files":"可由主工作簿重建，prepare-eis会覆盖这些派生CSV/YAML，不修改工作簿或原始文件"}
    (data_dir/"preparation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary
