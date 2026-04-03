#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def _best_available_label(
    probs: Mapping[str, float],
    labels: Sequence[str],
    counts: Dict[str, int],
    caps: Mapping[str, int],
) -> str | None:
    best_label = None
    best_prob = float("-inf")
    for label in labels:
        cap = int(caps.get(label, 10_000))
        if counts.get(label, 0) >= cap:
            continue
        prob = float(probs.get(label, 0.0))
        if prob > best_prob:
            best_prob = prob
            best_label = label
    return best_label


def decode_with_constraints(
    players: Sequence[dict],
    label_probabilities: Sequence[Mapping[str, float]],
    config: Mapping[str, object],
) -> List[str]:
    """
    Assign one final label per player while enforcing lineup constraints.

    This is intentionally deterministic and simple:
    1) honor locked labels first (if present),
    2) enforce exactly_one labels,
    3) greedily assign the rest under max caps.
    """
    offense_labels = [str(v) for v in config.get("offense", [])]
    defense_labels = [str(v) for v in config.get("defense", [])]
    constraints = config.get("constraints", {}) or {}
    exactly_one = [str(v) for v in constraints.get("exactly_one", [])]
    max_per_play = {str(k): int(v) for k, v in (constraints.get("max_per_play", {}) or {}).items()}

    # exactly_one implies a max cap of 1.
    caps = dict(max_per_play)
    for label in exactly_one:
        caps[label] = min(1, caps.get(label, 1))

    assignments: List[str | None] = [None for _ in players]
    counts: Dict[str, int] = {}
    unassigned = set(range(len(players)))

    def side_labels(side: str) -> List[str]:
        return offense_labels if side == "offense" else defense_labels

    def assign(idx: int, label: str) -> None:
        assignments[idx] = label
        counts[label] = counts.get(label, 0) + 1
        if idx in unassigned:
            unassigned.remove(idx)

    # 1) Locked labels first.
    for idx, player in enumerate(players):
        locked_label = player.get("locked_label")
        if not locked_label:
            continue
        locked_label = str(locked_label)
        allowed = side_labels(player.get("side", "defense"))
        if locked_label not in allowed:
            continue
        cap = int(caps.get(locked_label, 10_000))
        if counts.get(locked_label, 0) >= cap:
            continue
        assign(idx, locked_label)

    # 2) Exactly-one labels.
    for label in exactly_one:
        cap = int(caps.get(label, 1))
        if counts.get(label, 0) >= cap:
            continue

        best_idx = None
        best_prob = float("-inf")
        for idx in list(unassigned):
            allowed = side_labels(players[idx].get("side", "defense"))
            if label not in allowed:
                continue
            prob = float(label_probabilities[idx].get(label, 0.0))
            if prob > best_prob:
                best_prob = prob
                best_idx = idx
        if best_idx is not None:
            assign(best_idx, label)

    # 3) Greedy fill under caps.
    while unassigned:
        best_idx = None
        best_label = None
        best_prob = float("-inf")

        for idx in list(unassigned):
            allowed = side_labels(players[idx].get("side", "defense"))
            label = _best_available_label(label_probabilities[idx], allowed, counts, caps)
            if label is None:
                continue
            prob = float(label_probabilities[idx].get(label, 0.0))
            if prob > best_prob:
                best_prob = prob
                best_idx = idx
                best_label = label

        if best_idx is None or best_label is None:
            # All caps saturated; fall back to each player's highest-prob side label.
            idx = next(iter(unassigned))
            allowed = side_labels(players[idx].get("side", "defense"))
            fallback = max(allowed, key=lambda l: float(label_probabilities[idx].get(l, 0.0)))
            assign(idx, fallback)
        else:
            assign(best_idx, best_label)

    return [str(label) for label in assignments]
