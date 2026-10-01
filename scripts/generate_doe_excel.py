"""从主项目配置生成 Sobol 初始实验表，避免维护第二套变量和约束。"""
import argparse
from pathlib import Path

from coating_ai.config import FEATURES, PROJECT_ROOT
from coating_ai.data import frame, initial_points


def main():
    parser = argparse.ArgumentParser(description="生成满足物理约束的初始 DOE 表")
    parser.add_argument("--n", type=int, default=20, help="配方数量")
    parser.add_argument("--seed", type=int, default=42, help="Sobol 随机种子")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "initial_design.xlsx",
        help="Excel 输出路径",
    )
    parser.add_argument("--preview", action="store_true", help="只在终端预览，不写文件")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n 必须大于 0")

    table = frame(initial_points(args.n, args.seed))[["sample_id", *FEATURES]]
    print(table.to_string(index=False))
    if args.preview:
        return
    if args.output.exists():
        raise FileExistsError(f"文件已存在，为避免覆盖请更换名称：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(args.output, index=False)
    print(f"已生成：{args.output.resolve()}")


if __name__ == "__main__":
    main()

