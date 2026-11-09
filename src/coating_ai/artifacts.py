"""保存训练运行，并在之后的推荐步骤中安全地恢复模型。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd
import torch

from .config import OUTPUT_DIR, ProjectConfig, schema_signature
from .data import load_data, save_csv
from .learning import build_model


def create_run(label: str) -> Path:
    """在固定输出目录下创建一个明确命名、不会覆盖旧结果的子目录。"""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", label):
        raise ValueError("结果名称必须为1–64个字母、数字、下划线或连字符，且以字母或数字开头。")
    path = OUTPUT_DIR / label
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"结果目录已存在：{path}。请用 --name 选择新名称，不覆盖已有结果。") from error
    return path


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """以便于人工检查的格式写入运行元数据。"""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_training_run(
    config: ProjectConfig,
    folder: Path,
    table: pd.DataFrame,
    model,
    model_bounds: torch.Tensor,
    metrics: pd.DataFrame | None = None,
) -> None:
    """保存模型、训练数据快照、设置快照和可选验证指标。"""
    save_csv(table, folder / "training_snapshot.csv")
    shutil.copy2(config.source_path, folder / "settings_snapshot.yaml")
    if metrics is not None:
        save_csv(metrics, folder / "cv_metrics.csv")
    torch.save(
        {
            "schema": schema_signature(config),
            "model_bounds": model_bounds.detach().cpu(),
            "model_state_dict": model.state_dict(),
        },
        folder / "model.pt",
    )
    write_json(
        {
            "project_name": config.name,
            "training_rows": len(table),
            "input_variables": config.input_names,
            "target_variables": config.target_names,
            "model_file": "model.pt",
        },
        folder / "training_metadata.json",
    )


def load_training_run(
    config: ProjectConfig,
    run: str | Path,
) -> tuple[object, pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    """恢复训练模型，并核对当前设置与训练时的变量结构是否一致。"""
    folder = Path(run).expanduser().resolve()
    snapshot_path = folder / "training_snapshot.csv"
    checkpoint_path = folder / "model.pt"
    if not snapshot_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("训练目录必须同时包含training_snapshot.csv和model.pt。")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # 兼容较旧的PyTorch，但项目锁定版本通常不会走到这里。
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema") != schema_signature(config):
        raise ValueError("当前设置中的变量名称、类型或顺序与该训练模型不一致，请重新训练。")
    table, train_x, train_y = load_data(config, snapshot_path)
    model_bounds = checkpoint["model_bounds"].to(dtype=train_x.dtype)
    model = build_model(train_x, train_y, model_bounds)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, table, train_x, train_y, model_bounds
