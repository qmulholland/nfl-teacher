#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from decoder import decode_with_constraints


LABEL_ANCHORS: Dict[str, Tuple[float, float]] = {
    # Offense
    "QB": (0.38, 0.50),
    "RB": (0.32, 0.50),
    "WR": (0.36, 0.50),
    "TE": (0.44, 0.72),
    "LT": (0.47, 0.34),
    "LG": (0.47, 0.43),
    "C": (0.47, 0.50),
    "RG": (0.47, 0.57),
    "RT": (0.47, 0.66),
    # Defense
    "DE": (0.53, 0.50),
    "DT": (0.53, 0.44),
    "NT": (0.53, 0.50),
    "LB": (0.60, 0.50),
    "CB": (0.69, 0.14),
    "S": (0.76, 0.52),
    "EDGE": (0.58, 0.80),
}

OFFENSE_ANCHOR_CENTER = (0.47, 0.50)
DEFENSE_ANCHOR_CENTER = (0.60, 0.50)

# Normalized x spans the full 100-yard field between end zones in the UI.
FIELD_LENGTH_YARDS = 100.0

# Practical depth bands (from LOS) used to separate on-line DL vs off-ball second level.
EDGE_OFF_BALL_MIN_YARDS = 3.0
EDGE_OFF_BALL_MAX_YARDS = 5.0
DE_MAX_DEPTH_YARDS = 5.5
MAX_EDGE_BACK_FROM_LINE_YARDS = 2.5
MAX_INTERIOR_BACK_FROM_LINE_YARDS = 1.75
MIN_SAFETY_DEPTH_YARDS = 5.0
DEEP_SAFETY_PROMOTION_YARDS = 12.0
SINGLE_HIGH_DEPTH_GAP_YARDS = 3.0


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalized_distance_to_yards(dist: float) -> float:
    return abs(dist) * FIELD_LENGTH_YARDS


def softmax(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def expected_team_center_distance(label: str, side: str) -> float:
    ax, ay = LABEL_ANCHORS.get(label, (0.5, 0.5))
    if side == "offense":
        return distance((ax, ay), OFFENSE_ANCHOR_CENTER)
    return distance((ax, ay), DEFENSE_ANCHOR_CENTER)


def enrich_features(players: Sequence[dict], los_x: float) -> List[dict]:
    if not players:
        return []
    center_x = sum(float(p.get("x", 0.5)) for p in players) / len(players)
    center_y = sum(float(p.get("y", 0.5)) for p in players) / len(players)

    points = [(float(p.get("x", 0.5)), float(p.get("y", 0.5))) for p in players]
    out: List[dict] = []
    for i, player in enumerate(players):
        p_side = player.get("side", "defense")
        me = points[i]
        nearest_team = float("inf")
        nearest_opp = float("inf")
        team_sum_x = 0.0
        team_sum_y = 0.0
        team_count = 0
        for j, other in enumerate(players):
            if i == j:
                continue
            d = distance(me, points[j])
            if other.get("side", "defense") == p_side:
                nearest_team = min(nearest_team, d)
                team_sum_x += points[j][0]
                team_sum_y += points[j][1]
                team_count += 1
            else:
                nearest_opp = min(nearest_opp, d)
        if nearest_team == float("inf"):
            nearest_team = 0.0
        if nearest_opp == float("inf"):
            nearest_opp = 0.0
        if team_count > 0:
            team_center = (team_sum_x / team_count, team_sum_y / team_count)
            dist_to_team_center = distance(me, team_center)
        else:
            dist_to_team_center = 0.0
        clone = dict(player)
        clone["x"] = clamp01(float(player.get("x", 0.5)))
        clone["y"] = clamp01(float(player.get("y", 0.5)))
        clone["features"] = {
            "dist_to_los": abs(clone["x"] - los_x),
            "dist_to_sideline": min(clone["y"], 1.0 - clone["y"]),
            "dist_to_nearest_teammate": nearest_team,
            "dist_to_nearest_opponent": nearest_opp,
            "dist_to_team_center": dist_to_team_center,
            "formation_center_dx": clone["x"] - center_x,
            "formation_center_dy": clone["y"] - center_y,
        }
        out.append(clone)
    return out


def score_label(player: Mapping[str, object], label: str, los_x: float) -> float:
    x = float(player.get("x", 0.5))
    y = float(player.get("y", 0.5))
    features = player.get("features", {}) or {}
    dist_to_los = float(features.get("dist_to_los", abs(x - los_x)))
    dist_to_sideline = float(features.get("dist_to_sideline", min(y, 1.0 - y)))
    nearest_teammate = float(features.get("dist_to_nearest_teammate", 0.0))
    nearest_opp = float(features.get("dist_to_nearest_opponent", 0.0))
    dist_to_team_center = float(features.get("dist_to_team_center", 0.0))
    center_dx = float(features.get("formation_center_dx", 0.0))
    center_dy = float(features.get("formation_center_dy", 0.0))
    depth_yards = normalized_distance_to_yards(dist_to_los)

    ax, ay = LABEL_ANCHORS.get(label, (0.5, 0.5))
    anchor_dist_sq = (x - ax) ** 2 + (y - ay) ** 2
    expected_los_dist = abs(ax - los_x)
    expected_center_dist = expected_team_center_distance(label, str(player.get("side", "defense")))
    los_mismatch = abs(dist_to_los - expected_los_dist)

    # Broad football geometry priors.
    score = -9.0 * anchor_dist_sq
    score -= 2.3 * los_mismatch
    score -= 1.3 * abs(dist_to_team_center - expected_center_dist)
    score -= 0.7 * abs(center_dx - (ax - 0.5))
    score -= 0.4 * abs(center_dy - (ay - 0.5))
    score += 0.08 * min(dist_to_sideline, 0.25)
    score += 0.5 * min(nearest_opp, 0.4)
    score -= 0.3 * min(nearest_teammate, 0.4)

    # Keep edges near line depth and prefer off-ball depth for LBs.
    if label == "DE":
        if depth_yards > EDGE_OFF_BALL_MAX_YARDS:
            score -= 1.6 * (depth_yards - EDGE_OFF_BALL_MAX_YARDS)
        if EDGE_OFF_BALL_MIN_YARDS <= depth_yards <= EDGE_OFF_BALL_MAX_YARDS:
            score -= 0.6
    elif label == "LB":
        if EDGE_OFF_BALL_MIN_YARDS <= depth_yards <= EDGE_OFF_BALL_MAX_YARDS:
            score += 0.6

    return score


def offense_width_context(players: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    ol_labels = {"LT", "LG", "C", "RG", "RT"}
    ol_players = [
        p
        for p in players
        if str(p.get("side", "defense")) == "offense"
        and str(p.get("locked_label") or p.get("predicted_label") or "") in ol_labels
    ]
    if len(ol_players) < 3:
        ol_players = sorted(
            [p for p in players if str(p.get("side", "defense")) == "offense"],
            key=lambda p: abs(float(p.get("x", 0.5)) - 0.47),
        )[:5]

    if not ol_players:
        return {"has_ol": 0.0, "ol_min_y": 0.35, "ol_max_y": 0.65, "ol_center_y": 0.50, "ol_mean_x": 0.47}

    ys = [float(p.get("y", 0.5)) for p in ol_players]
    xs = [float(p.get("x", 0.5)) for p in ol_players]
    return {
        "has_ol": 1.0,
        "ol_min_y": min(ys),
        "ol_max_y": max(ys),
        "ol_center_y": sum(ys) / len(ys),
        "ol_mean_x": sum(xs) / len(xs),
    }


def offense_label_adjustment(
    player: Mapping[str, object],
    label: str,
    ctx: Mapping[str, float],
) -> float:
    if str(player.get("side", "defense")) != "offense":
        return 0.0

    y = float(player.get("y", 0.5))
    x = float(player.get("x", 0.5))
    sideline_gap = min(y, 1.0 - y)
    ol_min_y = float(ctx.get("ol_min_y", 0.35))
    ol_max_y = float(ctx.get("ol_max_y", 0.65))
    ol_center_y = float(ctx.get("ol_center_y", 0.50))
    ol_mean_x = float(ctx.get("ol_mean_x", 0.47))
    edge_gap = min(abs(y - ol_min_y), abs(y - ol_max_y))

    adjust = 0.0
    if label == "TE":
        # Tight ends should be very near the outside OL and not split wide.
        if edge_gap > 0.10:
            adjust -= 18.0 * (edge_gap - 0.10)
        if sideline_gap < 0.13:
            adjust -= 16.0 * (0.13 - sideline_gap)
        if edge_gap < 0.07:
            adjust += 1.8 * (0.07 - edge_gap)

    if label == "WR":
        # Outside WRs should be farther from OL edge and closer to sidelines.
        if edge_gap < 0.12:
            adjust -= 8.0 * (0.12 - edge_gap)
        if sideline_gap > 0.18:
            adjust -= 4.0 * (sideline_gap - 0.18)
        if sideline_gap < 0.14:
            adjust += 2.0 * (0.14 - sideline_gap)

    if label == "RB":
        # RBs should live near the formation core/backfield, not split wide.
        lateral_from_core = abs(y - ol_center_y)
        if sideline_gap < 0.20:
            adjust -= 36.0 * (0.20 - sideline_gap)
        if edge_gap < 0.12:
            adjust -= 18.0 * (0.12 - edge_gap)
        if lateral_from_core > 0.18:
            adjust -= 16.0 * (lateral_from_core - 0.18)
        # Mild penalty if RB aligns at/over LOS relative to OL body line.
        if x > ol_mean_x + 0.01:
            adjust -= 8.0 * (x - (ol_mean_x + 0.01))

    return adjust


def allowed_labels_for_side(side: str, config: Mapping[str, object]) -> List[str]:
    if side == "offense":
        return [str(v) for v in config.get("offense", [])]
    return [str(v) for v in config.get("defense", [])]


def _dl_labels(count: int) -> List[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["DT"]
    if count == 2:
        return ["DE", "DE"]
    if count == 3:
        return ["DE", "DT", "DE"]
    if count == 4:
        return ["DE", "DT", "DT", "DE"]
    if count == 5:
        return ["DE", "DT", "DT", "DT", "DE"]
    return ["DE"] + ["DT"] * (count - 2) + ["DE"]


def _lb_labels(count: int) -> List[str]:
    if count <= 0:
        return []
    return ["LB"] * count


def _secondary_labels(players: List[dict]) -> Dict[str, str]:
    if not players:
        return {}

    by_depth = sorted(players, key=lambda p: float(p.get("x", 0.5)), reverse=True)
    out: Dict[str, str] = {}

    deep = by_depth[:2] if len(by_depth) >= 2 else by_depth
    if len(deep) == 2:
        deep_sorted = sorted(deep, key=lambda p: float(p.get("y", 0.5)))
        out[str(deep_sorted[0]["id"])] = "S"
        out[str(deep_sorted[1]["id"])] = "S"
    elif len(deep) == 1:
        out[str(deep[0]["id"])] = "S"

    remaining = [p for p in players if str(p["id"]) not in out]
    if not remaining:
        return out

    remaining = sorted(remaining, key=lambda p: float(p.get("y", 0.5)))
    if len(remaining) == 1:
        out[str(remaining[0]["id"])] = "CB"
        return out

    # Corners on the edges; interior DBs are also reported as CB in this simplified label set.
    out[str(remaining[0]["id"])] = "CB"
    out[str(remaining[-1]["id"])] = "CB"
    for player in remaining[1:-1]:
        out[str(player["id"])] = "CB"
    return out


def _dl_slot_targets(count: int) -> List[Tuple[str, float]]:
    labels = _dl_labels(count)
    dt_count = sum(1 for label in labels if label == "DT")
    if dt_count <= 0:
        dt_targets: List[float] = []
    elif dt_count == 1:
        dt_targets = [0.50]
    elif dt_count == 2:
        dt_targets = [0.44, 0.56]
    elif dt_count == 3:
        dt_targets = [0.40, 0.50, 0.60]
    else:
        step = 0.20 / max(1, dt_count - 1)
        dt_targets = [0.40 + i * step for i in range(dt_count)]

    dt_idx = 0
    de_idx = 0
    de_targets = [0.34, 0.66]
    slots: List[Tuple[str, float]] = []
    for label in labels:
        if label == "DT":
            target_y = dt_targets[min(dt_idx, len(dt_targets) - 1)] if dt_targets else 0.50
            dt_idx += 1
        elif label == "DE":
            target_y = de_targets[min(de_idx, len(de_targets) - 1)]
            de_idx += 1
        elif label == "NT":
            target_y = 0.50
        else:
            target_y = float(LABEL_ANCHORS.get(label, (0.53, 0.50))[1])
        slots.append((label, target_y))
    return slots


def _assign_dl_labels(dl_group: List[dict]) -> Dict[str, str]:
    if not dl_group:
        return {}

    slots = _dl_slot_targets(len(dl_group))
    remaining = list(dl_group)
    out: Dict[str, str] = {}

    # Assign each DL slot against a target lane.
    for label, target_y in slots:
        if not remaining:
            break

        best = min(
            remaining,
            key=lambda p: (
                (1.7 if label in {"DT", "NT"} else 1.0) * abs(float(p.get("y", 0.5)) - target_y)
                + (4.8 if label in {"DT", "NT"} else 2.8)
                * max(0.0, 0.22 - min(float(p.get("y", 0.5)), 1.0 - float(p.get("y", 0.5))))
            ),
        )
        out[str(best["id"])] = label
        remaining.remove(best)

    # Safety fallback for any mismatch.
    assigned_ids = set(out.keys())
    unassigned_players = [p for p in remaining if str(p["id"]) not in assigned_ids]
    unassigned_labels = [label for label in _dl_labels(len(dl_group)) if label not in out.values()]
    for player, label in zip(unassigned_players, unassigned_labels):
        out[str(player["id"])] = label

    return out


def _assign_defense_from_front(
    defense_players: List[dict],
    front_counts: Tuple[int, int, int] | None,
    los_x: float,
    offense_players: Sequence[Mapping[str, object]] | None = None,
    offense_assignment: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    if not defense_players:
        return {}

    n = len(defense_players)
    if front_counts is None:
        # Fallback split when front is unknown.
        dl = max(3, min(4, n // 3))
        lb = max(2, min(4, n // 3))
        db = max(0, n - dl - lb)
    else:
        dl = max(0, min(front_counts[0], n))
        lb = max(0, min(front_counts[1], n - dl))
        db = max(0, n - dl - lb)

    dl_targets = [target for _, target in _dl_slot_targets(dl)]
    ordered_for_dl = sorted(
        defense_players,
        key=lambda p: (
            abs(float(p.get("x", 0.5)) - los_x)
            + 1.25 * (min(abs(float(p.get("y", 0.5)) - t) for t in dl_targets) if dl_targets else 0.0)
            + 4.5 * max(0.0, 0.22 - min(float(p.get("y", 0.5)), 1.0 - float(p.get("y", 0.5))))
        ),
    )
    dl_group = ordered_for_dl[:dl]
    dl_ids = {str(player["id"]) for player in dl_group}

    remaining = [player for player in defense_players if str(player["id"]) not in dl_ids]

    def lb_selection_cost(player: Mapping[str, object]) -> float:
        x = float(player.get("x", 0.5))
        y = float(player.get("y", 0.5))
        sideline_gap = min(y, 1.0 - y)
        lateral = abs(y - 0.5)
        return (
            abs(x - los_x)
            + 2.6 * lateral
            + 7.5 * max(0.0, 0.20 - sideline_gap)
            + 2.0 * max(0.0, lateral - 0.22)
        )

    ordered_for_lb = sorted(
        remaining,
        key=lb_selection_cost,
    )
    lb_group = sorted(ordered_for_lb[:lb], key=lambda p: float(p.get("y", 0.5)))
    lb_ids = {str(player["id"]) for player in lb_group}
    db_group = [player for player in remaining if str(player["id"]) not in lb_ids][:db]

    player_by_id = {str(player["id"]): player for player in defense_players}
    assignment: Dict[str, str] = {}
    assignment.update(_assign_dl_labels(dl_group))
    for player, label in zip(lb_group, _lb_labels(len(lb_group))):
        assignment[str(player["id"])] = label
    assignment.update(_secondary_labels(db_group))

    # Sanity pass: if a DL is too deep off the LOS compared to the line,
    # treat that defender as a stand-up LB.
    dl_depths = [
        abs(float(player_by_id[pid].get("x", 0.5)) - los_x)
        for pid, label in assignment.items()
        if label in {"DE", "DT", "NT"} and pid in player_by_id
    ]
    if dl_depths:
        dl_depths = sorted(dl_depths)
        # Approximate LOS line depth from the shallowest DL cluster.
        line_depth = dl_depths[min(len(dl_depths) - 1, max(0, len(dl_depths) // 4))]
    else:
        line_depth = 0.06

    line_depth_yards = normalized_distance_to_yards(line_depth)
    for pid, label in list(assignment.items()):
        if label not in {"DE", "DT", "NT"}:
            continue
        player = player_by_id.get(pid)
        if not player:
            continue
        dl_depth = abs(float(player.get("x", 0.5)) - los_x)
        dl_depth_yards = normalized_distance_to_yards(dl_depth)
        back_from_line_yards = max(0.0, dl_depth_yards - line_depth_yards)
        if label == "DE":
            # DEs should be on/near the line; 3-5 yards is typically off-ball OLB depth.
            off_ball_band = EDGE_OFF_BALL_MIN_YARDS <= dl_depth_yards <= EDGE_OFF_BALL_MAX_YARDS
            if dl_depth_yards > DE_MAX_DEPTH_YARDS or back_from_line_yards > MAX_EDGE_BACK_FROM_LINE_YARDS:
                assignment[pid] = "LB"
                continue
            if off_ball_band and back_from_line_yards >= 1.5:
                assignment[pid] = "LB"
                continue
        if label in {"DT", "NT"} and back_from_line_yards > MAX_INTERIOR_BACK_FROM_LINE_YARDS:
            assignment[pid] = "LB"

    # Tackle-relative front lanes:
    # - 0.5 person inside to 1.5 person outside LT/RT => DE lane.
    # - Between DE lanes => DT lane.
    # - Outside these lanes => LB/CB path (CB conversion can still occur later).
    if offense_players and offense_assignment:
        ol_label_set = {"LT", "LG", "C", "RG", "RT"}

        def offense_label(player: Mapping[str, object]) -> str:
            pid = str(player.get("id", ""))
            if pid in offense_assignment:
                return str(offense_assignment[pid])
            return str(player.get("locked_label") or player.get("predicted_label") or "")

        labeled_ol = [p for p in offense_players if offense_label(p) in ol_label_set]
        ol_ys = sorted(float(p.get("y", 0.5)) for p in labeled_ol)
        if len(ol_ys) >= 2:
            gaps = [ol_ys[i + 1] - ol_ys[i] for i in range(len(ol_ys) - 1)]
            positive_gaps = sorted(g for g in gaps if g > 0.0)
            if positive_gaps:
                mid = len(positive_gaps) // 2
                median_gap = (
                    positive_gaps[mid]
                    if len(positive_gaps) % 2 == 1
                    else (positive_gaps[mid - 1] + positive_gaps[mid]) / 2.0
                )
            else:
                median_gap = 0.07

            person_width = max(0.045, min(0.08, 0.9 * median_gap))
            inside_band = 0.5 * person_width
            outside_band = 1.5 * person_width

            lt_player = next((p for p in offense_players if offense_label(p) == "LT"), None)
            rt_player = next((p for p in offense_players if offense_label(p) == "RT"), None)
            lt_y = float(lt_player.get("y", ol_ys[0])) if lt_player else ol_ys[0]
            rt_y = float(rt_player.get("y", ol_ys[-1])) if rt_player else ol_ys[-1]

            left_de_min = lt_y - outside_band
            left_de_max = lt_y + inside_band
            right_de_min = rt_y - inside_band
            right_de_max = rt_y + outside_band
            dt_min = left_de_max
            dt_max = right_de_min

            for pid, label in list(assignment.items()):
                if label not in {"LB", "DE", "DT", "NT"}:
                    continue
                player = player_by_id.get(pid)
                if not player:
                    continue
                x = float(player.get("x", 0.5))
                y = float(player.get("y", 0.5))
                depth = abs(x - los_x)
                depth_yards = normalized_distance_to_yards(depth)
                back_from_line_yards = max(0.0, depth_yards - line_depth_yards)

                # Only apply tackle-lane remap to near-line defenders.
                if back_from_line_yards > MAX_EDGE_BACK_FROM_LINE_YARDS:
                    if label in {"DE", "DT", "NT"}:
                        assignment[pid] = "LB"
                    continue

                in_left_de = left_de_min <= y <= left_de_max
                in_right_de = right_de_min <= y <= right_de_max
                in_dt_lane = dt_min < dt_max and dt_min < y < dt_max

                if in_left_de or in_right_de:
                    assignment[pid] = "DE"
                elif in_dt_lane:
                    assignment[pid] = "DT"
                else:
                    assignment[pid] = "LB"

    # Edge-lane sanity: DEs should be outside interior DT lane, not stacked in it.
    dt_ys = sorted(
        float(player_by_id[pid].get("y", 0.5))
        for pid, label in assignment.items()
        if label in {"DT", "NT"} and pid in player_by_id
    )
    if dt_ys:
        dt_min_y = dt_ys[0]
        dt_max_y = dt_ys[-1]
        min_outside_gap = 0.05
        for pid, label in list(assignment.items()):
            if label != "DE":
                continue
            player = player_by_id.get(pid)
            if not player:
                continue
            y = float(player.get("y", 0.5))
            if dt_min_y - min_outside_gap <= y <= dt_max_y + min_outside_gap:
                assignment[pid] = "LB"

    # Apex pass: a "corner" aligned shallow and interior (over slot/inside WR)
    # should be treated as a linebacker, not an outside CB.
    apex_candidates: List[Tuple[float, str]] = []
    for pid, label in assignment.items():
        if label != "CB":
            continue
        player = player_by_id.get(pid)
        if not player:
            continue
        y = float(player.get("y", 0.5))
        x = float(player.get("x", 0.5))
        sideline_gap = min(y, 1.0 - y)
        depth = abs(x - los_x)
        if depth <= 0.14 and sideline_gap >= 0.18:
            # Prefer the most shallow interior "CB".
            apex_candidates.append((depth, pid))

    if apex_candidates:
        _, apex_pid = min(apex_candidates, key=lambda t: t[0])
        assignment[apex_pid] = "LB"

    # Receiver-alignment pass: corners over inside receivers (slot/apex) should
    # usually be LB/overhang players; boundary-over-boundary tends to stay CB.
    if offense_players and offense_assignment:
        receiver_ys: List[float] = []
        interior_ol_ys: List[float] = []
        for offense_player in offense_players:
            pid = str(offense_player.get("id", ""))
            label = str(
                offense_assignment.get(
                    pid,
                    offense_player.get("locked_label") or offense_player.get("predicted_label") or "",
                )
            )
            if label in {"WR", "TE"}:
                receiver_ys.append(float(offense_player.get("y", 0.5)))
            if label in {"LG", "C", "RG"}:
                interior_ol_ys.append(float(offense_player.get("y", 0.5)))

        if len(receiver_ys) >= 2:
            receiver_ys = sorted(receiver_ys)
            outer_ys = [receiver_ys[0], receiver_ys[-1]]
            inner_ys = receiver_ys[1:-1]
            if not inner_ys:
                inner_ys = [(outer_ys[0] + outer_ys[1]) / 2.0]

            for pid, label in list(assignment.items()):
                if label not in {"CB", "S", "FS", "SS", "LB"}:
                    continue
                player = player_by_id.get(pid)
                if not player:
                    continue
                y = float(player.get("y", 0.5))
                x = float(player.get("x", 0.5))
                depth = abs(x - los_x)
                sideline_gap = min(y, 1.0 - y)
                near_inner = any(abs(y - ry) <= 0.08 for ry in inner_ys)
                near_outer = any(abs(y - ry) <= 0.08 for ry in outer_ys)

                if label in {"CB", "S", "FS", "SS"} and near_inner and depth <= 0.16 and sideline_gap >= 0.16:
                    assignment[pid] = "LB"
                elif label == "LB" and near_outer and depth >= 0.14 and sideline_gap <= 0.18:
                    assignment[pid] = "CB"

        # Interior relation pass: DTs should align to interior OL, not over slot/TE space.
        if interior_ol_ys:
            for pid, label in list(assignment.items()):
                if label not in {"DT", "NT"}:
                    continue
                player = player_by_id.get(pid)
                if not player:
                    continue
                y = float(player.get("y", 0.5))
                x = float(player.get("x", 0.5))
                depth = abs(x - los_x)
                interior_gap = min(abs(y - iy) for iy in interior_ol_ys)
                receiver_gap = min((abs(y - ry) for ry in receiver_ys), default=1.0)
                # If this "DT" is detached from interior OL and sits in receiver/alley space,
                # treat it as LB (second-level overhang).
                if interior_gap >= 0.075 and receiver_gap <= 0.09 and depth >= line_depth + 0.025:
                    assignment[pid] = "LB"

    # Depth sanity: LBs should not align as deep as the secondary.
    lb_depth_limit = max(0.18, line_depth + 0.10)
    for pid, label in list(assignment.items()):
        if label != "LB":
            continue
        player = player_by_id.get(pid)
        if not player:
            continue
        x = float(player.get("x", 0.5))
        y = float(player.get("y", 0.5))
        depth = abs(x - los_x)
        if depth <= lb_depth_limit:
            continue

        sideline_gap = min(y, 1.0 - y)
        # Deep second-level defenders roll into CB in the simplified label set.
        assignment[pid] = "CB"

    # Final DB role alignment:
    # - Safeties are deep and generally inside the width of the outside LBs.
    # - CBs align horizontally with outside WRs (y-lane match).
    if offense_players and offense_assignment:
        db_ids = [
            pid
            for pid, label in assignment.items()
            if label in {"CB", "S", "FS", "SS"} and pid in player_by_id
        ]
        if len(db_ids) >= 2:
            wr_ys: List[float] = []
            for offense_player in offense_players:
                pid = str(offense_player.get("id", ""))
                olabel = str(
                    offense_assignment.get(
                        pid,
                        offense_player.get("locked_label") or offense_player.get("predicted_label") or "",
                    )
                )
                if olabel == "WR":
                    wr_ys.append(float(offense_player.get("y", 0.5)))

            if len(wr_ys) >= 2:
                wr_ys = sorted(wr_ys)
                outside_wr_ys = [wr_ys[0], wr_ys[-1]]
                hash_targets = [1.0 / 3.0, 2.0 / 3.0]

                # Default to two safeties in standard 4+ DB structures.
                safety_count = 2 if len(db_ids) >= 4 else 1
                cb_count = max(0, len(db_ids) - safety_count)
                safety_min_depth = MIN_SAFETY_DEPTH_YARDS / FIELD_LENGTH_YARDS

                lb_ys = sorted(
                    float(player_by_id[pid].get("y", 0.5))
                    for pid, label in assignment.items()
                    if label == "LB" and pid in player_by_id
                )
                if len(lb_ys) >= 2:
                    safety_window_min = lb_ys[0] - 0.01
                    safety_window_max = lb_ys[-1] + 0.01
                elif len(lb_ys) == 1:
                    safety_window_min = lb_ys[0] - 0.10
                    safety_window_max = lb_ys[0] + 0.10
                else:
                    # Fallback interior band when LB lanes are unavailable.
                    safety_window_min = 0.30
                    safety_window_max = 0.70

                def in_safety_window(player_id: str) -> bool:
                    y = float(player_by_id[player_id].get("y", 0.5))
                    return safety_window_min <= y <= safety_window_max

                remaining = list(db_ids)
                s_ids: List[str] = []
                deep_candidates = [
                    dpid
                    for dpid in remaining
                    if abs(float(player_by_id[dpid].get("x", 0.5)) - los_x) >= safety_min_depth
                    and in_safety_window(dpid)
                ]

                for target_y in hash_targets:
                    if len(s_ids) >= safety_count or not deep_candidates:
                        break
                    best_pid = min(
                        deep_candidates,
                        key=lambda dpid: (
                            abs(float(player_by_id[dpid].get("y", 0.5)) - target_y),
                            -abs(float(player_by_id[dpid].get("x", 0.5)) - los_x),
                        ),
                    )
                    s_ids.append(best_pid)
                    deep_candidates.remove(best_pid)
                    remaining.remove(best_pid)

                while deep_candidates and len(s_ids) < safety_count:
                    best_pid = min(
                        deep_candidates,
                        key=lambda dpid: (
                            min(
                                abs(float(player_by_id[dpid].get("y", 0.5)) - hash_targets[0]),
                                abs(float(player_by_id[dpid].get("y", 0.5)) - hash_targets[1]),
                            ),
                            -abs(float(player_by_id[dpid].get("x", 0.5)) - los_x),
                        ),
                    )
                    s_ids.append(best_pid)
                    deep_candidates.remove(best_pid)
                    remaining.remove(best_pid)

                cb_ids: List[str] = []
                for target_y in outside_wr_ys:
                    if not remaining or len(cb_ids) >= cb_count:
                        break
                    best_pid = min(
                        remaining,
                        key=lambda dpid: abs(float(player_by_id[dpid].get("y", 0.5)) - target_y),
                    )
                    cb_ids.append(best_pid)
                    remaining.remove(best_pid)

                # Fill remaining corners by outside-WR lane proximity.
                while remaining and len(cb_ids) < cb_count:
                    best_pid = min(
                        remaining,
                        key=lambda dpid: min(
                            abs(float(player_by_id[dpid].get("y", 0.5)) - outside_wr_ys[0]),
                            abs(float(player_by_id[dpid].get("y", 0.5)) - outside_wr_ys[1]),
                        ),
                    )
                    cb_ids.append(best_pid)
                    remaining.remove(best_pid)

                for dpid in db_ids:
                    assignment[dpid] = "S" if dpid in s_ids else "CB"

    # Boundary sanity: corners should not sit in the interior DE corridor.
    # Keep LB logic authoritative here: interior CBs become LBs.
    de_ys = sorted(
        float(player_by_id[pid].get("y", 0.5))
        for pid, label in assignment.items()
        if label == "DE" and pid in player_by_id
    )
    if len(de_ys) >= 2:
        de_min_y = de_ys[0]
        de_max_y = de_ys[-1]
        interior_margin = 0.01
        for pid, label in list(assignment.items()):
            if label != "CB":
                continue
            player = player_by_id.get(pid)
            if not player:
                continue
            y = float(player.get("y", 0.5))
            if de_min_y + interior_margin <= y <= de_max_y - interior_margin:
                assignment[pid] = "LB"

    # Hard safety depth floor: safeties are never within 5 yards of the LOS.
    safety_min_depth = MIN_SAFETY_DEPTH_YARDS / FIELD_LENGTH_YARDS
    for pid, label in list(assignment.items()):
        if label not in {"S", "FS", "SS"}:
            continue
        player = player_by_id.get(pid)
        if not player:
            continue
        depth = abs(float(player.get("x", 0.5)) - los_x)
        if depth < safety_min_depth:
            assignment[pid] = "LB"

    # Deep fallback: if we're short on safeties, promote very deep DB/LB candidates.
    # Use expected DB count from the front template, not current labels, so a deep defender
    # mislabeled as LB can still be promoted back to safety when appropriate.
    db_ids = [pid for pid, label in assignment.items() if label in {"CB", "S", "FS", "SS"}]
    expected_db_count = max(0, db)
    if expected_db_count > 0:
        target_safeties = 2 if expected_db_count >= 4 else 1
    else:
        target_safeties = 2 if len(db_ids) >= 4 else 1
    current_safeties = [
        pid for pid, label in assignment.items() if label in {"S", "FS", "SS"} and pid in player_by_id
    ]
    if len(current_safeties) < target_safeties:
        promote_candidates: List[Tuple[float, float, str]] = []
        for pid, label in assignment.items():
            if label not in {"CB", "LB"}:
                continue
            player = player_by_id.get(pid)
            if not player:
                continue
            y = float(player.get("y", 0.5))
            sideline_gap = min(y, 1.0 - y)
            depth_yards = normalized_distance_to_yards(abs(float(player.get("x", 0.5)) - los_x))
            if depth_yards < DEEP_SAFETY_PROMOTION_YARDS:
                continue
            # LB->S promotion should stay interior; CB can be either boundary or interior.
            if label == "LB" and sideline_gap < 0.18:
                continue
            promote_candidates.append((depth_yards, sideline_gap, pid))

        for _, _, pid in sorted(
            promote_candidates,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        ):
            if len(current_safeties) >= target_safeties:
                break
            assignment[pid] = "S"
            current_safeties.append(pid)

    # Single-high variation: when one safety is clearly much deeper than the rest,
    # keep only that deepest defender as S.
    safety_ids = [
        pid
        for pid, label in assignment.items()
        if label in {"S", "FS", "SS"} and pid in player_by_id
    ]
    if len(safety_ids) >= 2:
        safety_depths: List[Tuple[float, str]] = [
            (
                normalized_distance_to_yards(abs(float(player_by_id[pid].get("x", 0.5)) - los_x)),
                pid,
            )
            for pid in safety_ids
        ]
        safety_depths.sort(key=lambda item: item[0], reverse=True)
        deepest_depth, deepest_pid = safety_depths[0]
        second_depth = safety_depths[1][0]
        if (
            deepest_depth >= DEEP_SAFETY_PROMOTION_YARDS
            and (deepest_depth - second_depth) >= SINGLE_HIGH_DEPTH_GAP_YARDS
        ):
            for _, pid in safety_depths[1:]:
                y = float(player_by_id[pid].get("y", 0.5))
                sideline_gap = min(y, 1.0 - y)
                assignment[pid] = "LB" if sideline_gap >= 0.18 else "CB"
            assignment[deepest_pid] = "S"

    # Interior DBs in the shallow/second-level band should be treated as LBs.
    for pid, label in list(assignment.items()):
        if label not in {"CB", "S", "FS", "SS"}:
            continue
        player = player_by_id.get(pid)
        if not player:
            continue
        x = float(player.get("x", 0.5))
        y = float(player.get("y", 0.5))
        depth_yards = normalized_distance_to_yards(abs(x - los_x))
        sideline_gap = min(y, 1.0 - y)
        if sideline_gap >= 0.18 and depth_yards <= 8.5:
            assignment[pid] = "LB"

    # LB line completion: if we already have two LBs on a common depth line,
    # promote one interior DB on that same line to LB (typical 4-3 shape).
    lb_ids = [pid for pid, label in assignment.items() if label == "LB" and pid in player_by_id]
    if 2 <= len(lb_ids) < 3:
        lb_depths = sorted(abs(float(player_by_id[pid].get("x", 0.5)) - los_x) for pid in lb_ids)
        lb_depth_ref = lb_depths[len(lb_depths) // 2]
        lb_center_y = sum(float(player_by_id[pid].get("y", 0.5)) for pid in lb_ids) / len(lb_ids)

        promote_candidates: List[str] = []
        for pid, label in assignment.items():
            if label not in {"CB", "S", "FS", "SS"}:
                continue
            player = player_by_id.get(pid)
            if not player:
                continue
            x = float(player.get("x", 0.5))
            y = float(player.get("y", 0.5))
            sideline_gap = min(y, 1.0 - y)
            depth = abs(x - los_x)
            if sideline_gap < 0.18:
                continue
            if abs(depth - lb_depth_ref) <= 0.03:
                promote_candidates.append(pid)

        if promote_candidates:
            promote_pid = min(
                promote_candidates,
                key=lambda pid: (
                    abs(abs(float(player_by_id[pid].get("x", 0.5)) - los_x) - lb_depth_ref),
                    abs(float(player_by_id[pid].get("y", 0.5)) - lb_center_y),
                ),
            )
            assignment[promote_pid] = "LB"

    return assignment


def _best_non_label(probs: Mapping[str, float], allowed: Sequence[str], banned: set[str]) -> str:
    candidates = [label for label in allowed if label not in banned]
    if not candidates:
        return allowed[0] if allowed else "RB"
    return max(candidates, key=lambda label: float(probs.get(label, 0.0)))


def _assign_offense_from_geometry(
    offense_players: List[dict],
    decoded_labels_by_id: Mapping[str, str],
    probs_by_id: Mapping[str, Mapping[str, float]],
    offense_labels: Sequence[str],
) -> Dict[str, str]:
    if not offense_players:
        return {}

    ol_labels = {"LT", "LG", "C", "RG", "RT"}
    player_by_id = {str(player["id"]): player for player in offense_players}
    labels_by_id = {str(player["id"]): str(decoded_labels_by_id.get(str(player["id"]), "RB")) for player in offense_players}
    for pid, label in list(labels_by_id.items()):
        if label == "FB":
            labels_by_id[pid] = "RB"
        if label == "SLOT":
            labels_by_id[pid] = "WR"

    ol_players = [p for p in offense_players if labels_by_id.get(str(p["id"])) in ol_labels]
    center_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "C"), None)

    # 1) QB should be closest player directly behind center (depth can be under center).
    if center_player is not None:
        center_x = float(center_player.get("x", 0.5))
        center_y = float(center_player.get("y", 0.5))
        ball_x = 0.5

        non_ol = [p for p in offense_players if labels_by_id.get(str(p["id"])) not in ol_labels]
        qb_pool = [
            p
            for p in non_ol
            if abs(float(p.get("y", 0.5)) - center_y) <= 0.18
            or float(p.get("x", 0.5)) <= center_x - 0.03
        ]
        if not qb_pool:
            qb_pool = non_ol
        if qb_pool:
            behind_ball = [p for p in qb_pool if float(p.get("x", 0.5)) <= ball_x + 0.02]
            if not behind_ball:
                behind_ball = qb_pool

            # "Directly behind the ball" lane first.
            lane_half_width = 0.02
            lane_players = [
                p
                for p in behind_ball
                if abs(float(p.get("y", 0.5)) - center_y) <= lane_half_width
            ]
            candidate_pool = lane_players if lane_players else behind_ball

            # Immediate player behind the ball from that pool.
            immediate = sorted(
                candidate_pool,
                key=lambda p: (
                    -float(p.get("x", 0.5)),
                    abs(float(p.get("y", 0.5)) - center_y),
                ),
            )[0]

            def qb_score(player: Mapping[str, object]) -> float:
                x = float(player.get("x", 0.5))
                y = float(player.get("y", 0.5))
                lateral = abs(y - center_y)
                ahead_penalty = max(0.0, x - (center_x + 0.01)) * 8.0
                return lateral + ahead_penalty

            # QB is always the immediate player directly behind the ball.
            qb_player = immediate
            qb_id = str(qb_player["id"])

            # Demote any previous QB assignment first.
            for player in qb_pool:
                pid = str(player["id"])
                if pid == qb_id:
                    continue
                if labels_by_id.get(pid) == "QB":
                    probs = probs_by_id.get(pid, {})
                    labels_by_id[pid] = _best_non_label(probs, offense_labels, {"QB"} | ol_labels)

            labels_by_id[qb_id] = "QB"

    # 2) Tight ends: near outside OL edge + not too close to sideline + near OL x-width.
    if len(ol_players) >= 3:
        ol_ys = [float(p.get("y", 0.5)) for p in ol_players]
        ol_min_y = min(ol_ys)
        ol_max_y = max(ol_ys)
        ol_x_mean = sum(float(p.get("x", 0.5)) for p in ol_players) / len(ol_players)

        for player in offense_players:
            pid = str(player["id"])
            current = labels_by_id.get(pid, "RB")
            if current in ol_labels or current in {"QB", "RB"}:
                continue
            y = float(player.get("y", 0.5))
            x = float(player.get("x", 0.5))
            edge_gap = min(abs(y - ol_min_y), abs(y - ol_max_y))
            sideline_gap = min(y, 1.0 - y)
            inline_gap = abs(x - ol_x_mean)

            if edge_gap <= 0.10 and sideline_gap >= 0.13 and inline_gap <= 0.08:
                labels_by_id[pid] = "TE"

        # Hard cap: TE must stay within 1.5 "person widths" of a tackle.
        lt_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "LT"), None)
        rt_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "RT"), None)
        lt_y = float(lt_player.get("y", ol_min_y)) if lt_player else ol_min_y
        rt_y = float(rt_player.get("y", ol_max_y)) if rt_player else ol_max_y

        sorted_ol_ys = sorted(ol_ys)
        gaps = [sorted_ol_ys[i + 1] - sorted_ol_ys[i] for i in range(len(sorted_ol_ys) - 1)]
        positive_gaps = sorted([g for g in gaps if g > 0.0])
        if positive_gaps:
            mid = len(positive_gaps) // 2
            median_gap = (
                positive_gaps[mid]
                if len(positive_gaps) % 2 == 1
                else (positive_gaps[mid - 1] + positive_gaps[mid]) / 2.0
            )
        else:
            median_gap = 0.07

        person_width = max(0.045, min(0.08, 0.9 * median_gap))
        max_te_tackle_gap = 1.3 * person_width

        for player in offense_players:
            pid = str(player["id"])
            if labels_by_id.get(pid) != "TE":
                continue
            y = float(player.get("y", 0.5))
            tackle_gap = min(abs(y - lt_y), abs(y - rt_y))
            if tackle_gap > max_te_tackle_gap:
                labels_by_id[pid] = "WR"

    # 3) Formation-aware backfield pass:
    # - Inline edge players can be TE even if base decode picked RB.
    # - RB is only forced when a true backfield candidate exists.
    center_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "C"), None)
    center_x = float(center_player.get("x", 0.47)) if center_player else 0.47
    center_y = float(center_player.get("y", 0.50)) if center_player else 0.50

    ol_ys_for_rel = sorted(float(p.get("y", 0.5)) for p in ol_players) if ol_players else []
    if ol_ys_for_rel:
        ol_min_y = ol_ys_for_rel[0]
        ol_max_y = ol_ys_for_rel[-1]
        ol_x_mean = sum(float(p.get("x", 0.5)) for p in ol_players) / len(ol_players)
    else:
        ol_min_y = 0.35
        ol_max_y = 0.65
        ol_x_mean = 0.47

    qb_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "QB"), None)
    qb_id = str(qb_player["id"]) if qb_player else ""

    for player in offense_players:
        pid = str(player["id"])
        if labels_by_id.get(pid) != "RB":
            continue
        if pid == qb_id:
            continue

        y = float(player.get("y", 0.5))
        x = float(player.get("x", 0.5))
        edge_gap = min(abs(y - ol_min_y), abs(y - ol_max_y))
        sideline_gap = min(y, 1.0 - y)
        inline_gap = abs(x - ol_x_mean)
        # Near tackle edge + close to OL x-depth + not split to sideline => TE.
        if edge_gap <= 0.13 and inline_gap <= 0.09 and sideline_gap >= 0.13:
            labels_by_id[pid] = "TE"

    skill_candidates = [
        p
        for p in offense_players
        if labels_by_id.get(str(p["id"])) not in ol_labels and labels_by_id.get(str(p["id"])) != "QB"
    ]
    if skill_candidates:
        def rb_cost(player: Mapping[str, object]) -> float:
            x = float(player.get("x", 0.5))
            y = float(player.get("y", 0.5))
            lateral = abs(y - center_y)
            sideline_gap = min(y, 1.0 - y)
            ahead_penalty = max(0.0, x - (center_x + 0.01)) * 5.0
            too_deep_penalty = max(0.0, (center_x - x) - 0.28) * 2.0
            wide_penalty = max(0.0, 0.20 - sideline_gap) * 6.0
            return (2.0 * lateral) + ahead_penalty + too_deep_penalty + wide_penalty

        backfield_candidates = [
            p
            for p in skill_candidates
            if float(p.get("x", 0.5)) <= center_x - 0.02
            and abs(float(p.get("y", 0.5)) - center_y) <= 0.24
        ]
        if backfield_candidates:
            primary_rb = min(backfield_candidates, key=rb_cost)
            primary_rb_id = str(primary_rb["id"])
            labels_by_id[primary_rb_id] = "RB"
        else:
            primary_rb_id = ""

        for player in skill_candidates:
            pid = str(player["id"])
            if pid == primary_rb_id:
                continue
            if labels_by_id.get(pid) != "RB":
                continue
            y = float(player.get("y", 0.5))
            sideline_gap = min(y, 1.0 - y)
            if abs(y - center_y) > 0.18 or sideline_gap < 0.20:
                probs = probs_by_id.get(pid, {})
                labels_by_id[pid] = _best_non_label(probs, offense_labels, {"QB", "RB"} | ol_labels)

    # 3b) QB/OL depth split for RB vs TE:
    # - Between the QB and OL, RB only owns the first 10% from the QB side.
    # - The remaining 90% toward the OL is treated as TE.
    if qb_player is not None and ol_players:
        half_body_depth = 0.01
        qb_front_x = float(qb_player.get("x", 0.5)) + half_body_depth
        ol_back_x = min(float(p.get("x", 0.5)) for p in ol_players) - 0.01
        span = ol_back_x - qb_front_x
        rb_max_x = qb_front_x + (0.10 * span) if span > 1e-6 else qb_front_x
        for player in offense_players:
            pid = str(player["id"])
            if labels_by_id.get(pid) != "RB" or pid == qb_id:
                continue
            x = float(player.get("x", 0.5))
            rb_back_x = x - half_body_depth
            # If the back of the RB is already in front of the QB's front,
            # the back is effectively aligned as a TE.
            if rb_back_x > qb_front_x:
                labels_by_id[pid] = "TE"
                continue
            if span > 1e-6 and qb_front_x <= x <= ol_back_x and x > rb_max_x:
                labels_by_id[pid] = "TE"

    # 3c) Any non-WR skill player in front of the QB is treated as TE,
    # while keeping the same vertical TE restrictions.
    if qb_player is not None and ol_players:
        qb_front_x = float(qb_player.get("x", 0.5)) + 0.01
        lt_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "LT"), None)
        rt_player = next((p for p in offense_players if labels_by_id.get(str(p["id"])) == "RT"), None)
        lt_y = float(lt_player.get("y", ol_min_y)) if lt_player else ol_min_y
        rt_y = float(rt_player.get("y", ol_max_y)) if rt_player else ol_max_y

        if len(ol_ys_for_rel) >= 2:
            gaps = [ol_ys_for_rel[i + 1] - ol_ys_for_rel[i] for i in range(len(ol_ys_for_rel) - 1)]
            positive_gaps = sorted([g for g in gaps if g > 0.0])
            if positive_gaps:
                mid = len(positive_gaps) // 2
                median_gap = (
                    positive_gaps[mid]
                    if len(positive_gaps) % 2 == 1
                    else (positive_gaps[mid - 1] + positive_gaps[mid]) / 2.0
                )
            else:
                median_gap = 0.07
        else:
            median_gap = 0.07

        person_width = max(0.045, min(0.08, 0.9 * median_gap))
        max_te_tackle_gap = 1.3 * person_width

        for player in offense_players:
            pid = str(player["id"])
            label = labels_by_id.get(pid)
            if label in ol_labels or label in {"QB", "WR"}:
                continue

            x = float(player.get("x", 0.5))
            y = float(player.get("y", 0.5))
            if x <= qb_front_x:
                continue

            edge_gap = min(abs(y - ol_min_y), abs(y - ol_max_y))
            sideline_gap = min(y, 1.0 - y)
            tackle_gap = min(abs(y - lt_y), abs(y - rt_y))
            if edge_gap <= 0.13 and sideline_gap >= 0.13 and tackle_gap <= max_te_tackle_gap:
                labels_by_id[pid] = "TE"

    # 3d) If a TE is even with the QB in depth, treat it as a RB.
    if qb_player is not None:
        qb_x = float(qb_player.get("x", 0.5))
        even_depth_tolerance = 0.012
        for player in offense_players:
            pid = str(player["id"])
            if labels_by_id.get(pid) != "TE":
                continue
            te_x = float(player.get("x", 0.5))
            if abs(te_x - qb_x) <= even_depth_tolerance:
                labels_by_id[pid] = "RB"

    # 3e) Final safeguard: if an RB's back is ahead of the QB's front,
    # it must be a TE.
    if qb_player is not None:
        half_body_depth = 0.01
        qb_front_x = float(qb_player.get("x", 0.5)) + half_body_depth
        for player in offense_players:
            pid = str(player["id"])
            if labels_by_id.get(pid) != "RB" or pid == qb_id:
                continue
            rb_back_x = float(player.get("x", 0.5)) - half_body_depth
            if rb_back_x > qb_front_x:
                labels_by_id[pid] = "TE"

    # 4) Safety pass: never allow duplicate OL labels (LT/LG/C/RG/RT).
    for ol_label in ["LT", "LG", "C", "RG", "RT"]:
        holders = [pid for pid, label in labels_by_id.items() if label == ol_label]
        if len(holders) <= 1:
            continue

        def keep_score(pid: str) -> tuple[float, float]:
            player = player_by_id.get(pid, {})
            locked_bonus = 1.0 if str(player.get("locked_label") or "") == ol_label else 0.0
            label_prob = float(probs_by_id.get(pid, {}).get(ol_label, 0.0))
            return (locked_bonus, label_prob)

        keep_pid = max(holders, key=keep_score)
        for pid in holders:
            if pid == keep_pid:
                continue
            probs = probs_by_id.get(pid, {})
            labels_by_id[pid] = _best_non_label(probs, offense_labels, {"QB"} | ol_labels)

    # 5) Beginner mode: all slot-like alignments are treated as WR for now.
    for pid, label in list(labels_by_id.items()):
        if label == "SLOT":
            labels_by_id[pid] = "WR"

    return labels_by_id


def infer_play(
    players: Sequence[dict],
    config: Mapping[str, object],
    los_x: float = 0.5,
    defense_front_counts: Tuple[int, int, int] | None = None,
) -> List[dict]:
    enriched = enrich_features(players, los_x=los_x)
    off_ctx = offense_width_context(enriched)

    probs_per_player: List[Dict[str, float]] = []
    for player in enriched:
        labels = allowed_labels_for_side(str(player.get("side", "defense")), config)
        raw_scores = {}
        for label in labels:
            base = score_label(player, label, los_x=los_x)
            base += offense_label_adjustment(player, label, off_ctx)
            raw_scores[label] = base
        probs_per_player.append(softmax(raw_scores))

    decoded = decode_with_constraints(enriched, probs_per_player, config)
    decoded_by_id = {
        str(player.get("id", f"p{idx}")): str(decoded[idx]) for idx, player in enumerate(enriched)
    }
    probs_by_id = {
        str(player.get("id", f"p{idx}")): probs_per_player[idx] for idx, player in enumerate(enriched)
    }

    offense_players = [p for p in enriched if str(p.get("side", "defense")) == "offense"]
    offense_labels = [str(v) for v in config.get("offense", [])]
    offense_assignment = _assign_offense_from_geometry(
        offense_players=offense_players,
        decoded_labels_by_id=decoded_by_id,
        probs_by_id=probs_by_id,
        offense_labels=offense_labels,
    )
    defense_players = [p for p in enriched if str(p.get("side", "defense")) == "defense"]
    defense_assignment = _assign_defense_from_front(
        defense_players,
        front_counts=defense_front_counts,
        los_x=los_x,
        offense_players=offense_players,
        offense_assignment=offense_assignment,
    )

    out: List[dict] = []
    for idx, player in enumerate(enriched):
        probs = probs_per_player[idx]
        player_id = str(player.get("id", f"p{idx}"))
        decoded_label = decoded[idx]
        if str(player.get("side", "defense")) == "defense":
            decoded_label = defense_assignment.get(player_id, decoded_label)
            if decoded_label not in probs:
                probs = dict(probs)
                probs[decoded_label] = 0.55
        else:
            decoded_label = offense_assignment.get(player_id, decoded_label)
            if decoded_label not in probs:
                probs = dict(probs)
                probs[decoded_label] = 0.55

        out.append(
            {
                "id": player_id,
                "side": str(player.get("side", "defense")),
                "x": float(player.get("x", 0.5)),
                "y": float(player.get("y", 0.5)),
                "locked_label": player.get("locked_label"),
                "features": player.get("features", {}),
                "predicted_label": decoded_label,
                "predicted_confidence": float(probs.get(decoded_label, 0.0)),
                "label_probabilities": probs,
            }
        )
    return out


def starter_players(config: Mapping[str, object]) -> List[dict]:
    del config  # starter layout is fixed for a beginner-friendly first screen.
    los_x = 0.5
    yard_to_x = 0.02  # One hash tick = one yard in the UI starter layout.
    offense_back_shift_yards = 0.5

    def off_x(yards_from_los: float) -> float:
        return los_x - ((yards_from_los + offense_back_shift_yards) * yard_to_x)

    def def_x(yards_from_los: float) -> float:
        return los_x + (yards_from_los * yard_to_x)

    offense_starter = [
        (off_x(1.75), 0.13, None),  # WR (outside top): moved back +0.75 yd
        (off_x(2.5), 0.24, None),  # WR (inside top): moved back +0.5 yd
        (off_x(0.8), 0.372, "LT"),  # LT: OL spacing tightened by 20%
        (off_x(0.8), 0.436, "LG"),  # LG: OL spacing tightened by 20%
        (off_x(0.8), 0.50, "C"),  # C: 0.8 yd
        (off_x(0.8), 0.564, "RG"),  # RG: OL spacing tightened by 20%
        (off_x(0.8), 0.628, "RT"),  # RT: OL spacing tightened by 20%
        (off_x(3.0), 0.596, None),  # TE: centered between RG and RT lanes
        (off_x(1.5), 0.89, None),  # WR (outside bottom): moved back +0.5 yd
        (off_x(5.5), 0.50, None),  # QB: moved back +0.5 yd
        (off_x(7.5), 0.50, None),  # RB: moved back +0.5 yd
    ]

    defense_starter = [
        (def_x(0.6), 0.35),  # DE: 0.6 yds off LOS, DL group centered on ball
        (def_x(0.5), 0.45),  # DT: 0.5 yds off LOS, same spacing
        (def_x(0.5), 0.55),  # DT: 0.5 yds off LOS, same spacing
        (def_x(0.6), 0.65),  # DE: 0.6 yds off LOS, DL group centered on ball
        (def_x(6.75), 0.38),  # LB: aligned to next white line behind prior position
        (def_x(6.75), 0.50),  # LB: aligned to next white line behind prior position
        (def_x(6.75), 0.62),  # LB: aligned to next white line behind prior position
        (def_x(9.25), 0.13),  # CB: moved back +0.75 yds
        (def_x(9.25), 0.89),  # CB: moved back +0.75 yds
        (def_x(16.0), 0.333),  # S: moved back +2.0 yds (top hash lane)
        (def_x(16.0), 0.667),  # S: moved back +2.0 yds (bottom hash lane)
    ]

    players: List[dict] = []
    for idx, (x, y, locked_label) in enumerate(offense_starter):
        players.append(
            {
                "id": f"o{idx + 1}",
                "side": "offense",
                "x": x,
                "y": y,
                "locked_label": locked_label,
            }
        )
    for idx, (x, y) in enumerate(defense_starter):
        players.append(
            {
                "id": f"d{idx + 1}",
                "side": "defense",
                "x": x,
                "y": y,
                "locked_label": None,
            }
        )
    return players
