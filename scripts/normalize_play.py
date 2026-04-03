#!/usr/bin/env python3
"""
Normalize a single play JSON into a canonical coordinate system.

Canonical output guarantees:
- Offense direction is left_to_right.
- LOS is fixed at target_los_x (default 0.5).
- Player x/y coordinates are clipped to [0, 1].
- Optional geometric features are generated per player.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def pairwise_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def normalize_value(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def infer_bounds(players: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs = [float(p["x"]) for p in players]
    ys = [float(p["y"]) for p in players]
    return min(xs), max(xs), min(ys), max(ys)


def compute_features(players: List[Dict[str, Any]], los_x: float) -> None:
    if not players:
        return

    center_x = sum(float(p["x"]) for p in players) / len(players)
    center_y = sum(float(p["y"]) for p in players) / len(players)

    points = [(float(p["x"]), float(p["y"])) for p in players]

    for i, player in enumerate(players):
        side = player.get("side", "defense")
        my_point = points[i]

        teammate_distances = []
        opponent_distances = []
        team_sum_x = 0.0
        team_sum_y = 0.0
        team_count = 0
        for j, other in enumerate(players):
            if i == j:
                continue
            distance = pairwise_distance(my_point, points[j])
            if other.get("side", "defense") == side:
                teammate_distances.append(distance)
                team_sum_x += points[j][0]
                team_sum_y += points[j][1]
                team_count += 1
            else:
                opponent_distances.append(distance)

        nearest_teammate = min(teammate_distances) if teammate_distances else 0.0
        nearest_opponent = min(opponent_distances) if opponent_distances else 0.0
        if team_count > 0:
            team_center = (team_sum_x / team_count, team_sum_y / team_count)
            dist_to_team_center = pairwise_distance(my_point, team_center)
        else:
            dist_to_team_center = 0.0

        player["features"] = {
            "dist_to_los": abs(float(player["x"]) - los_x),
            "dist_to_sideline": min(float(player["y"]), 1.0 - float(player["y"])),
            "dist_to_nearest_teammate": nearest_teammate,
            "dist_to_nearest_opponent": nearest_opponent,
            "dist_to_team_center": dist_to_team_center,
            "formation_center_dx": float(player["x"]) - center_x,
            "formation_center_dy": float(player["y"]) - center_y,
        }


def normalize_play(payload: Dict[str, Any], target_los_x: float) -> Dict[str, Any]:
    players = payload.get("players", [])
    if not players:
        raise ValueError("Play payload must include a non-empty 'players' list.")

    normalization = payload.get("normalization", {})
    x_min, x_max, y_min, y_max = infer_bounds(players)

    # If provided, trust explicit raw field bounds.
    raw_x_min = float(normalization.get("field_x_min", x_min))
    raw_x_max = float(normalization.get("field_x_max", x_max))
    raw_y_min = float(normalization.get("field_y_min", y_min))
    raw_y_max = float(normalization.get("field_y_max", y_max))

    raw_los = float(normalization.get("los_x", (raw_x_min + raw_x_max) / 2.0))
    offense_direction = normalization.get("offense_direction", "left_to_right")

    normalized_players = []
    for player in players:
        x = normalize_value(float(player["x"]), raw_x_min, raw_x_max)
        y = normalize_value(float(player["y"]), raw_y_min, raw_y_max)

        normalized_player = {
            "id": str(player["id"]),
            "side": player.get("side", "defense"),
            "x": x,
            "y": y,
            "label": str(player["label"]),
        }
        normalized_players.append(normalized_player)

    los_norm = normalize_value(raw_los, raw_x_min, raw_x_max)

    if offense_direction == "right_to_left":
        for player in normalized_players:
            player["x"] = 1.0 - float(player["x"])
        los_norm = 1.0 - los_norm

    shift = target_los_x - los_norm
    for player in normalized_players:
        player["x"] = clamp01(float(player["x"]) + shift)
        player["y"] = clamp01(float(player["y"]))

    # Deterministic ordering makes model inputs reproducible.
    side_rank = {"offense": 0, "defense": 1}
    normalized_players.sort(
        key=lambda p: (side_rank.get(p.get("side", "defense"), 2), float(p["x"]), float(p["y"]))
    )

    compute_features(normalized_players, target_los_x)

    return {
        "play_id": payload["play_id"],
        "source_image": payload.get("source_image", ""),
        "normalization": {
            "offense_direction": "left_to_right",
            "los_x": target_los_x,
            "field_x_min": 0.0,
            "field_x_max": 1.0,
            "field_y_min": 0.0,
            "field_y_max": 1.0,
        },
        "players": normalized_players,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a play JSON into canonical format.")
    parser.add_argument("--input", required=True, help="Path to input play JSON.")
    parser.add_argument("--output", required=True, help="Path to write normalized play JSON.")
    parser.add_argument(
        "--target-los-x",
        type=float,
        default=0.5,
        help="Target normalized x position for LOS (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    payload = json.loads(in_path.read_text())
    normalized = normalize_play(payload, target_los_x=args.target_los_x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(normalized, indent=2))
    print(f"Wrote normalized play to {out_path}")


if __name__ == "__main__":
    main()
