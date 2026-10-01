import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .config import INITIAL_DESIGN_PATH, OUTPUT_DIR, TARGETS
from .data import frame, initial_points, load, save, simulate
from .learning import cross_validate, fit
from .optimization import pareto_hv, suggest


def new_run(label):
    path = OUTPUT_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def report(df, y, folder):
    mask, hv = pareto_hv(y)
    save(df.loc[mask.numpy()], folder / "pareto.csv")
    print(f"数据量={len(df)}，非支配配方数={int(mask.sum())}，HV={hv:.4f}", flush=True)
    return hv


def main():
    parser = argparse.ArgumentParser(description="涂层AI：模拟演示、初始设计、验证、推荐")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="纯模拟闭环，不使用真实实验数据")
    demo.add_argument("--rounds", type=int, default=1)
    demo.add_argument("--batch-size", type=int, default=5)
    init = sub.add_parser("init", help="生成待填写的20组真实实验模板")
    init.add_argument("--out", default=str(INITIAL_DESIGN_PATH))
    merge = sub.add_parser("merge", help="检查并合并已完成的新实验，保留旧数据文件")
    merge.add_argument("--data", required=True)
    merge.add_argument("--new", required=True)
    merge.add_argument("--out", required=True)
    for name in ["validate", "suggest"]:
        command = sub.add_parser(name)
        command.add_argument("--data", required=True)
        if name == "suggest":
            command.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    if hasattr(args, "batch_size") and not 1 <= args.batch_size <= 5:
        parser.error("batch-size必须在1–5之间。")
    if args.command == "demo" and not 1 <= args.rounds <= 6:
        parser.error("rounds必须在1–6之间。")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.command == "init":
        save(frame(initial_points()), args.out)
        print(f"已生成 {args.out}；三项目标留空，请实验后填写。")
        return
    if args.command == "merge":
        old, _, _ = load(args.data)
        new, _, _ = load(args.new, min_rows=1)
        combined = pd.concat([old, new], ignore_index=True)
        if combined["sample_id"].duplicated().any():
            raise ValueError("新旧数据存在重复sample_id，可能重复合并了同一批实验。")
        save(combined, args.out)
        print(f"已合并{len(old)}+{len(new)}={len(combined)}组真实数据：{args.out}")
        return
    if args.command == "demo":
        print("模拟数据演示：输出不具有真实材料含义。", flush=True)
        folder = new_run("demo")
        x = initial_points()
        y = simulate(x)
        history = []
        for t in range(args.rounds + 1):
            df = frame(x, y, kind="demo")
            mask, hv = pareto_hv(y)
            history.append({"round": t, "n": len(x), "hv": hv})
            save(df, folder / f"round_{t:02d}.csv")
            print(f"轮次 {t}：n={len(x)}, HV={hv:.4f}", flush=True)
            if t == args.rounds:
                break
            model = fit(x, y)
            new_x = suggest(model, x, args.batch_size)
            x, y = torch.cat([x, new_x]), torch.cat([y, simulate(new_x)])
        save(pd.DataFrame(history), folder / "hv_history.csv")
        report(df, y, folder)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot([h["n"] for h in history], [h["hv"] for h in history], "o-")
        plt.xlabel("Number of simulated formulations")
        plt.ylabel("Observed hypervolume (fixed reference)")
        plt.title("SIMULATION ONLY")
        plt.tight_layout()
        plt.savefig(folder / "hv.png", dpi=180)
        plt.close()
    else:
        df, x, y = load(args.data)
        folder = new_run(args.command)
        save(df, folder / "training_snapshot.csv")
        if args.command == "validate":
            metrics = cross_validate(x, y)
            save(metrics, folder / "cv_metrics.csv")
            print(metrics.to_string(index=False))
        else:
            print("正在拟合模型并推荐；正式实验前请检查验证结果及实际可制备性。", flush=True)
            model = fit(x, y)
            new_x = suggest(model, x, args.batch_size)
            candidates = frame(new_x, start=len(df)+1)
            prefix = folder.name
            candidates["sample_id"] = [f"{prefix}_{i+1:02d}" for i in range(len(new_x))]
            save(candidates, folder / "candidates_to_measure.csv")
            with torch.no_grad():
                posterior = model.posterior(new_x)
                mean, std = posterior.mean, posterior.variance.clamp_min(0).sqrt()
            predictions = candidates[["sample_id"]].copy()
            for j, name in enumerate(TARGETS):
                predictions[f"{name}_predicted"] = mean[:, j].numpy()
                predictions[f"{name}_latent_std"] = std[:, j].numpy()
            save(predictions, folder / "predictions_not_measurements.csv")
            report(df, y, folder)
    print(f"输出目录：{folder.resolve()}")


if __name__ == "__main__":
    main()
