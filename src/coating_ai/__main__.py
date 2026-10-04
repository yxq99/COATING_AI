"""项目命令行入口：建表、验证、训练、推荐、合并和模拟演示。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd
import torch

from .artifacts import create_run, load_training_run, save_training_run, write_json
from .config import DATA_DIR, DEFAULT_SETTINGS_PATH, load_project_config, resolve_search_bounds
from .data import frame, initial_points, load_data, save_csv, simulate
from .learning import cross_validate, fit
from .optimization import pareto_summary, recommend_candidates


def _add_settings_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings", default=str(DEFAULT_SETTINGS_PATH),
        help="项目YAML设置文件（默认：settings/project.yaml）",
    )


def _save_pareto_report(config, table, y, folder: Path) -> dict:
    if config.recommendation.mode != "pareto":
        return {}
    mask, hypervolume, reference = pareto_summary(config, y)
    save_csv(table.loc[mask.cpu().numpy()].copy(), folder / "observed_pareto.csv")
    return {
        "observed_pareto_rows": int(mask.sum()),
        "observed_hypervolume": float(hypervolume),
        "reference_point_in_maximization_space": reference.tolist(),
    }


def _candidate_measurement_table(config, candidates, folder: Path) -> pd.DataFrame:
    table = frame(config, candidates, kind="real")
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
    metrics = cross_validate(config, x, y, model_bounds)
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
        metrics = cross_validate(config, x, y, model_bounds)
        print(metrics.to_string(index=False))
    print("正在使用全部数据拟合最终模型……", flush=True)
    model = fit(x, y, model_bounds, config)
    save_training_run(config, folder, table, model, model_bounds, metrics)
    metadata = _save_pareto_report(config, table, y, folder)
    write_json(metadata, folder / "pareto_metadata.json")
    print(f"训练模型：{folder.resolve()}")


def _command_recommend(args, config) -> None:
    model, training_table, train_x, train_y, _ = load_training_run(config, args.run)
    if set(training_table["data_kind"]) != {"real"}:
        raise ValueError("recommend命令只接受真实实验训练模型；模拟模型只能用于demo流程。")
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
    print(f"已合并{len(old)}+{len(new)}={len(combined)}组真实数据：{combined_path.resolve()}")


def _command_demo(args, config) -> None:
    print("这是数学模拟闭环，所有目标值都不代表真实材料规律。", flush=True)
    torch.manual_seed(config.recommendation.seed)
    bounds = resolve_search_bounds(config)
    x = initial_points(config)
    y = simulate(config, x, bounds)
    folder = create_run(f"demo_{config.recommendation.mode}")
    history: list[dict] = []
    final_model = None

    for round_number in range(args.rounds + 1):
        table = frame(config, x, y, kind="demo")
        save_csv(table, folder / f"round_{round_number:02d}.csv")
        _, hypervolume, _ = pareto_summary(config, y)
        history.append({"round": round_number, "rows": len(x), "hypervolume": hypervolume})
        print(f"轮次{round_number}：数据量={len(x)}，观测HV={hypervolume:.4f}", flush=True)
        final_model = fit(x, y, bounds, config)
        if round_number == args.rounds:
            break
        new_x, report, _ = recommend_candidates(config, final_model, x, y, bounds)
        save_csv(report, folder / f"round_{round_number + 1:02d}_recommendations.csv")
        new_y = simulate(config, new_x, bounds)
        x = torch.cat([x, new_x])
        y = torch.cat([y, new_y])

    save_csv(pd.DataFrame(history), folder / "history.csv")
    save_training_run(config, folder, table, final_model, bounds)
    metadata = _save_pareto_report(config, table, y, folder)
    metadata["recommendation_mode"] = config.recommendation.mode
    metadata["simulation_only"] = True
    write_json(metadata, folder / "demo_metadata.json")
    print(f"模拟结果：{folder.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用实验AI：验证、训练和反向推荐")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="按设置生成待填写的初始实验表")
    _add_settings_argument(init)
    init.add_argument("--out", help="CSV输出路径")
    init.add_argument("--n", type=int, help="覆盖设置中的初始实验数量")
    init.add_argument("--seed", type=int, help="覆盖设置中的随机种子")

    validate = commands.add_parser("validate", help="执行数据检查和留一交叉验证")
    _add_settings_argument(validate)
    validate.add_argument("--data", required=True, help="已填写目标变量的真实实验CSV")

    train = commands.add_parser("train", help="验证并保存可供推荐使用的最终模型")
    _add_settings_argument(train)
    train.add_argument("--data", required=True, help="已填写目标变量的真实实验CSV")
    train.add_argument("--skip-cv", action="store_true", help="跳过留一验证（仅用于快速调试）")

    recommend = commands.add_parser("recommend", help="按设置中的target或pareto模式推荐输入")
    _add_settings_argument(recommend)
    recommend.add_argument("--run", required=True, help="train命令生成的训练目录")

    merge = commands.add_parser("merge", help="检查并合并旧数据和新完成的实验")
    _add_settings_argument(merge)
    merge.add_argument("--data", required=True, help="原真实实验CSV")
    merge.add_argument("--new", required=True, help="新完成的真实实验CSV")
    merge.add_argument("--out", required=True, help="新的合并CSV；不会覆盖旧文件")

    demo = commands.add_parser("demo", help="用模拟函数演示完整闭环")
    _add_settings_argument(demo)
    demo.add_argument("--rounds", type=int, default=1, help="推荐并模拟测量的轮数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "n", None) is not None and args.n < 1:
        parser.error("n必须大于0。")
    if getattr(args, "rounds", 0) < 0 or getattr(args, "rounds", 0) > 20:
        parser.error("rounds必须在0到20之间。")
    torch.set_num_threads(2)
    config = load_project_config(args.settings)
    handlers = {
        "init": _command_init,
        "validate": _command_validate,
        "train": _command_train,
        "recommend": _command_recommend,
        "merge": _command_merge,
        "demo": _command_demo,
    }
    handlers[args.command](args, config)


if __name__ == "__main__":
    main()
