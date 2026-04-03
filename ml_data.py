#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


FEATURE_KEYS = [
    "x",
    "y",
    "side_is_offense",
    "dist_to_los",
    "dist_to_sideline",
    "dist_to_nearest_teammate",
    "dist_to_nearest_opponent",
    "dist_to_team_center",
    "formation_center_dx",
    "formation_center_dy",
]


def load_positions(path: str) -> List[str]:
    payload = json.loads(Path(path).read_text())
    offense = payload.get("offense", [])
    defense = payload.get("defense", [])
    labels = [str(label) for label in offense + defense]
    if not labels:
        raise ValueError("No labels found in config positions file.")
    return labels


def load_split_ids(path: str) -> List[str]:
    payload = json.loads(Path(path).read_text())
    return [str(play_id) for play_id in payload.get("play_ids", [])]


def player_to_feature_row(player: Dict) -> List[float]:
    features = player.get("features", {})
    return [
        float(player.get("x", 0.0)),
        float(player.get("y", 0.0)),
        1.0 if player.get("side") == "offense" else 0.0,
        float(features.get("dist_to_los", 0.0)),
        float(features.get("dist_to_sideline", 0.0)),
        float(features.get("dist_to_nearest_teammate", 0.0)),
        float(features.get("dist_to_nearest_opponent", 0.0)),
        float(features.get("dist_to_team_center", 0.0)),
        float(features.get("formation_center_dx", 0.0)),
        float(features.get("formation_center_dy", 0.0)),
    ]


def load_examples_from_split(
    data_dir: str,
    split_path: str,
    labels: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    play_ids = load_split_ids(split_path)

    x_rows: List[List[float]] = []
    y_rows: List[int] = []

    for play_id in play_ids:
        play_path = Path(data_dir) / f"{play_id}.json"
        if not play_path.exists():
            continue

        play_payload = json.loads(play_path.read_text())
        for player in play_payload.get("players", []):
            label = player.get("label")
            if label not in label_to_index:
                continue
            x_rows.append(player_to_feature_row(player))
            y_rows.append(label_to_index[label])

    if not x_rows:
        raise ValueError(
            f"No training examples found. Check split '{split_path}' and data dir '{data_dir}'."
        )

    return torch.tensor(x_rows, dtype=torch.float32), torch.tensor(y_rows, dtype=torch.long)


def build_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for truth, pred in zip(y_true, y_pred):
        cm[int(truth), int(pred)] += 1
    return cm
