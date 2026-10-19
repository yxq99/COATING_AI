"""统一命令行：EIS整理、建表、验证、训练、预测、推荐和合并。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd
import torch

from .artifacts import create_run, load_training_run, save_training_run, write_json
from .config import DATA_DIR, DEFAULT_SETTINGS_PATH, load_project_config, resolve_search_bounds
from .data import frame, initial_points, load_data, save_csv
from .learning import cross_validate, fit, predict
from .optimization import pareto_summary, recommend_candidates


def _add_settings_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings", default=str(DEFAULT_SETTINGS_PATH),
        help="项目YAML设置文件（默认：data/project.yaml）",
    )


def _save_pareto_report(config, table, y, folder: Path) -> dict:
    mask, hypervolume, reference = pareto_summary(config, y)
    save_csv(table.loc[mask.cpu().numpy()].copy(), folder / "observed_pareto.csv")
    return {
        "observed_pareto_rows": int(mask.sum()),
        "observed_hypervolume": float(hypervolume),
        "reference_point_in_maximization_space": reference.tolist(),
    }


def _candidate_measurement_table(config, candidates, folder: Path) -> pd.DataFrame:
    table = frame(config, candidates)
    table["sample_id"] = [f"{folder.name}_{index + 1:02d}" for index in range(len(table))]
    return table


def _command_init(args, config) -> None:
    count = args.n or config.initial_design_size
    output = Path(args.out) if args.out else DATA_DIR / "initial_design.csv"
    points = initial_points(config, count, args.seed)
    save_csv(frame(config, points), output)
    print(f"已生成{count}组初始实验：{output.resolve()}")
    print("请完成实验并填写所有启用的目标变量列，再运行validate或train。")


def _command_validate(args, config) -> None:
    table, x, y = load_data(config, args.data)
    model_bounds = resolve_search_bounds(config, x)
    folder = create_run("validate")
    save_csv(table, folder / "training_snapshot.csv")
    shutil.copy2(config.source_path, folder / "settings_snapshot.yaml")
    metrics = cross_validate(config, x, y, model_bounds, table.get("validation_group"))
    save_csv(metrics, folder / "cv_metrics.csv")
    metadata = _save_pareto_report(config, table, y, folder)
    write_json(metadata, folder / "validation_metadata.json")
    print(metrics.to_string(index=False))
    print(f"验证结果：{folder.resolve()}")


def _command_train(args, config) -> None:
    table, x, y = load_data(config, args.data)
    model_bounds = resolve_search_bounds(config, x)
    folder = create_run("train")
    metrics = None
    if not args.skip_cv:
        metrics = cross_validate(config, x, y, model_bounds, table.get("validation_group"))
        print(metrics.to_string(index=False))
    print("正在使用全部数据拟合最终模型……", flush=True)
    model = fit(x, y, model_bounds)
    save_training_run(config, folder, table, model, model_bounds, metrics)
    metadata = _save_pareto_report(config, table, y, folder)
    write_json(metadata, folder / "pareto_metadata.json")
    print(f"训练模型：{folder.resolve()}")


def _command_recommend(args, config) -> None:
    if config.task == "prediction":
        raise ValueError("早期EIS模型只用于predict；观测特征不能当作可控制配方搜索。")
    model, training_table, train_x, train_y, _ = load_training_run(config, args.run)
    search_bounds = resolve_search_bounds(config, train_x)
    candidates, prediction_report, metadata = recommend_candidates(
        config, model, train_x, train_y, search_bounds
    )
    folder = create_run(f"recommend_{config.recommendation.mode}")
    measurement_table = _candidate_measurement_table(config, candidates, folder)
    prediction_report.insert(0, "sample_id", measurement_table["sample_id"])
    save_csv(measurement_table, folder / "candidates_to_measure.csv")
    save_csv(prediction_report, folder / "predictions_not_measurements.csv")
    shutil.copy2(config.source_path, folder / "settings_snapshot.yaml")
    metadata.update(_save_pareto_report(config, training_table, train_y, folder))
    metadata["recommendation_mode"] = config.recommendation.mode
    metadata["source_training_run"] = str(Path(args.run).expanduser().resolve())
    write_json(metadata, folder / "recommendation_metadata.json")
    if args.out_data:
        from .eis_workflow import write_measurement_templates
        write_measurement_templates(measurement_table, Path(args.out_data))
        print(f"待填实验数据：{Path(args.out_data).resolve()}")
    print(prediction_report.to_string(index=False))
    print(f"推荐结果：{folder.resolve()}")


def _command_merge(args, config) -> None:
    old, _, _ = load_data(config, args.data)
    new, _, _ = load_data(config, args.new, min_rows=1)
    combined = pd.concat([old, new], ignore_index=True)
    if combined["sample_id"].duplicated().any():
        duplicates = combined.loc[combined["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"新旧数据存在重复sample_id：{duplicates}")
    combined_path = Path(args.out)
    save_csv(combined, combined_path)
    try:
        load_data(config, combined_path)
    except Exception:
        combined_path.unlink(missing_ok=True)
        raise
    print(f"已合并{len(old)}+{len(new)}={len(combined)}组完整数据：{combined_path.resolve()}")


def _command_predict(args, config) -> None:
    model, _, train_x, _, _ = load_training_run(config, args.run)
    table = pd.read_csv(args.data)
    needed = ["sample_id", *config.input_names]
    if set(needed) - set(table):
        raise ValueError(f"预测表缺少列：{sorted(set(needed)-set(table))}")
    if table.empty:
        raise ValueError("没有可预测行。请先填写新配方及1/3/7天EIS，再运行prepare-eis。")
    if table.sample_id.isna().any() or table.sample_id.duplicated().any():
        raise ValueError("预测sample_id不能为空或重复。")
    x = torch.tensor(table[config.input_names].to_numpy(dtype=float), dtype=torch.double)
    if not torch.isfinite(x).all():
        raise ValueError("预测输入包含空白或无效数字。")
    from .config import feasible
    bounds = resolve_search_bounds(config, train_x)
    for column, item in enumerate(config.active_inputs):
        if not item.bounds.enabled:
            bounds[:, column] = torch.tensor([-torch.inf, torch.inf])
    if not feasible(config, x, bounds).all():
        raise ValueError("预测输入违反显式边界、整数或线性约束。")
    mean, std = predict(model, x)
    report = table[needed].copy()
    for column, name in enumerate(config.target_names):
        report[f"{name}_predicted"] = mean[:, column].numpy()
        report[f"{name}_latent_std"] = std[:, column].numpy()
    output = Path(args.out) if args.out else create_run("predict") / "predictions.csv"
    save_csv(report, output)
    print(report.to_string(index=False))
    print(f"预测结果：{output.resolve()}；标准差不等于制备波动或可靠达标概率。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用实验AI：验证、训练和反向推荐")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-eis", help="从主工作簿整理时间特征、目标和重复统计")
    prepare.add_argument("--workbook", default=str(DATA_DIR / "ML_data.xlsx"))
    prepare.add_argument("--data-dir", default=str(DATA_DIR))
    prepare.add_argument("--new-data", help="recommend --out-data生成、填写后的新实验目录")

    init = commands.add_parser("init", help="按设置生成待填写的初始实验表")
    _add_settings_argument(init)
    init.add_argument("--out", help="CSV输出路径")
    init.add_argument("--n", type=int, help="覆盖设置中的初始实验数量")
    init.add_argument("--seed", type=int, help="覆盖设置中的随机种子")

    validate = commands.add_parser("validate", help="执行数据检查和留一交叉验证")
    _add_settings_argument(validate)
    validate.add_argument("--data", required=True, help="已填写目标变量的统一CSV")

    train = commands.add_parser("train", help="验证并保存可供推荐使用的最终模型")
    _add_settings_argument(train)
    train.add_argument("--data", required=True, help="已填写目标变量的统一CSV")
    train.add_argument("--skip-cv", action="store_true", help="跳过留一验证（仅用于快速调试）")

    recommend = commands.add_parser("recommend", help="按设置中的target或pareto模式推荐输入")
    _add_settings_argument(recommend)
    recommend.add_argument("--run", required=True, help="train命令生成的训练目录")
    recommend.add_argument("--out-data", help="额外在指定data子目录生成待填配方和EIS模板")

    prediction = commands.add_parser("predict", help="用已有模型预测新样本，不需要后期标签")
    _add_settings_argument(prediction)
    prediction.add_argument("--run", required=True)
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--out", help="不覆盖已有文件的预测CSV路径")

    merge = commands.add_parser("merge", help="检查并合并旧数据和新完成的实验")
    _add_settings_argument(merge)
    merge.add_argument("--data", required=True, help="原完整数据CSV")
    merge.add_argument("--new", required=True, help="新完成的完整数据CSV")
    merge.add_argument("--out", required=True, help="新的合并CSV；不会覆盖旧文件")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "n", None) is not None and args.n < 1:
        parser.error("n必须大于0。")
    torch.set_num_threads(2)
    if args.command == "prepare-eis":
        from .eis_workflow import prepare_data
        result = prepare_data(Path(args.workbook), Path(args.data_dir), args.new_data)
        print(result)
        return
    config = load_project_config(args.settings)
    handlers = {
        "init": _command_init,
        "validate": _command_validate,
        "train": _command_train,
        "recommend": _command_recommend,
        "merge": _command_merge,
        "predict": _command_predict,
    }
    handlers[args.command](args, config)


if __name__ == "__main__":
    main()
