#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - runtime fallback
    cv2 = None
    np = None


FRONT_RE = re.compile(r"^\d-\d(?:-\d)?$")
TEMPLATE_VERSION = 2


def simplify_formation(folder_name: str) -> str:
    tokens = folder_name.split("_")
    if tokens and FRONT_RE.match(tokens[0]):
        return tokens[0]

    if tokens and tokens[0] in {"Nickel", "Dime"} and len(tokens) > 1 and FRONT_RE.match(tokens[1]):
        return f"{tokens[0]} {tokens[1]}"

    if tokens and tokens[0] == "Goal" and len(tokens) > 2 and FRONT_RE.match(tokens[2]):
        return tokens[2]

    if tokens:
        return tokens[0].replace("-", " ")
    return folder_name.replace("_", " ")


def simplify_coverage(play_name: str) -> str:
    name = play_name.upper()
    name = name.replace(".PNG", "")
    name = name.replace("-", "_")

    if "COVER_1" in name or "COV_1" in name or "1_HOLE" in name or "1_ROBBER" in name:
        return "Cover 1"
    if "TAMPA_2" in name or "COVER_2" in name or "2_INVERT" in name:
        return "Cover 2"
    if "COVER_3" in name or "3_MATCH" in name or "3_CLOUD" in name or "3_SKY" in name or "3_BUZZ" in name:
        return "Cover 3"
    if "COVER_4" in name or "4_PALMS" in name or "4_DROP" in name or "4_QUARTERS" in name:
        return "Cover 4"
    if "COVER_6" in name:
        return "Cover 6"
    if "COVER_9" in name:
        return "Cover 9"
    if "ZERO" in name:
        return "Cover 0"
    if "MAN" in name:
        return "Man"
    if "BLITZ" in name:
        return "Blitz"
    return "Zone"


def parse_front_counts(front: str) -> Tuple[int, int, int] | None:
    if FRONT_RE.match(front):
        parts = [int(v) for v in front.split("-")]
        if len(parts) == 2:
            # e.g., 4-3 -> assume remaining DB count to 11.
            dl, lb = parts
            return dl, lb, max(0, 11 - dl - lb)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return None


def fallback_front_counts(player_count: int) -> Tuple[int, int, int]:
    """
    Always produce a numeric front tuple (DL, LB, DB), even when geometry
    inference is uncertain.
    """
    n = max(0, int(player_count))
    if n == 0:
        return 0, 0, 0

    if n < 7:
        dl = max(1, min(3, n // 2))
        lb = max(0, n - dl - 2)
        db = max(0, n - dl - lb)
        return dl, lb, db

    # Default to a practical base family and adjust to keep counts valid.
    db = 4 if n >= 10 else 3
    dl = 4 if n >= 10 else 3
    lb = n - dl - db

    if lb < 2:
        take_from_dl = min(max(0, dl - 3), 2 - lb)
        dl -= take_from_dl
        lb = n - dl - db

    if lb < 1:
        take_from_db = min(max(0, db - 2), 1 - lb)
        db -= take_from_db
        lb = n - dl - db

    if lb < 0:
        lb = 0
        db = max(0, n - dl - lb)

    return dl, lb, db


def infer_front_counts(
    defense_players: Sequence[Mapping[str, object]],
    los_x: float = 0.5,
) -> Tuple[int, int, int]:
    estimated = estimate_front_counts_from_geometry(defense_players, los_x=los_x)
    if estimated is not None:
        return estimated
    return fallback_front_counts(len(defense_players))


def format_front(front_counts: Tuple[int, int, int]) -> str:
    dl, lb, db = front_counts
    # Traditional base fronts stay 2-part (e.g., 4-3, 3-4).
    if db == 4:
        return f"{dl}-{lb}"
    return f"{dl}-{lb}-{db}"


def select_shell_points(
    points: Sequence[Tuple[float, float]],
    *,
    los_x: float = 0.5,
    db_count: int | None = None,
) -> List[Tuple[float, float]]:
    if not points:
        return []

    if db_count is None:
        db_count = min(5, len(points))
    db_count = max(0, min(int(db_count), len(points)))
    if db_count == 0:
        return []

    ranked = sorted(points, key=lambda p: abs(p[0] - los_x), reverse=True)
    return ranked[:db_count]


def estimate_front_counts_from_geometry(
    defense_players: Sequence[Mapping[str, object]],
    los_x: float = 0.5,
) -> Tuple[int, int, int] | None:
    """
    Estimate front counts from defender x-depth relative to LOS.

    We use an elbow on LOS-distance to estimate DL count (3-5), then pick LB count
    from common families using a simple deep-secondary heuristic.
    """
    n = len(defense_players)
    if n < 7:
        return None

    dists = sorted(abs(float(p.get("x", 0.5)) - los_x) for p in defense_players)
    if len(dists) < 2:
        return None

    # Find biggest distance jump among first few defenders near the LOS.
    search_end = min(6, len(dists) - 1)
    best_i = 2
    best_gap = -1.0
    for i in range(1, search_end):
        gap = dists[i + 1] - dists[i]
        if gap > best_gap:
            best_gap = gap
            best_i = i
    dl = max(3, min(5, best_i + 1))

    deep_threshold = 0.18
    deep_count = sum(1 for d in dists if d >= deep_threshold)

    if dl == 3:
        lb = 2 if deep_count >= 3 else 3
    elif dl == 4:
        lb = 2 if deep_count >= 3 else 3
    else:
        lb = 2

    db = max(0, n - dl - lb)
    if db < 4:
        lb = max(1, lb - (4 - db))
        db = n - dl - lb
    if db > 6:
        lb += db - 6
        db = n - dl - lb

    return dl, lb, db


def _signature_from_points(points: Sequence[Tuple[float, float]], max_points: int = 11) -> List[float]:
    if not points:
        return [0.0] * (2 * max_points)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    centered = [(x - cx, y - cy) for x, y in points]
    scale = max(math.sqrt(dx * dx + dy * dy) for dx, dy in centered)
    if scale < 1e-6:
        scale = 1.0

    normalized = [(dx / scale, dy / scale) for dx, dy in centered]
    normalized = sorted(normalized, key=lambda p: (p[1], p[0]))
    normalized = list(normalized[:max_points])
    while len(normalized) < max_points:
        normalized.append((0.0, 0.0))

    vector: List[float] = []
    for x, y in normalized:
        vector.extend([round(float(x), 6), round(float(y), 6)])
    return vector


def _extract_white_markers(image_path: Path) -> List[Tuple[float, float]]:
    if cv2 is None or np is None:
        return []

    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Trim header/footer UI areas and side artifacts.
    x0 = int(0.07 * w)
    x1 = int(0.93 * w)
    y0 = int(0.20 * h)
    y1 = int(0.88 * h)
    if x1 <= x0 or y1 <= y0:
        return []

    roi = rgb[y0:y1, x0:x1]
    mask = (
        (roi[:, :, 0] > 200)
        & (roi[:, :, 1] > 200)
        & (roi[:, :, 2] > 200)
    ).astype("uint8")

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[float, float, int]] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 45:
            continue
        if width > 10 or height > 10:
            continue
        cx, cy = centroids[idx]
        abs_x = (x0 + float(cx)) / float(w)
        abs_y = (y0 + float(cy)) / float(h)
        candidates.append((abs_x, abs_y, area))

    # Defensive X markers are usually among the strongest small components.
    candidates.sort(key=lambda v: v[2], reverse=True)
    points = [(x, y) for x, y, _ in candidates[:11]]
    return points


def _build_templates(defense_dir: Path) -> List[Dict]:
    templates: List[Dict] = []
    for image_path in defense_dir.rglob("*.png"):
        points = _extract_white_markers(image_path)
        if len(points) < 8:
            continue

        folder = image_path.parent.name
        front = simplify_formation(folder)
        coverage = simplify_coverage(image_path.stem)
        call = f"{front} {coverage}".strip()
        signature = _signature_from_points(points)
        point_players = [{"x": x, "y": y} for x, y in points]
        front_counts = infer_front_counts(point_players, los_x=0.5)
        db_count = max(0, len(points) - front_counts[0] - front_counts[1])
        shell_points = select_shell_points(points, los_x=0.5, db_count=db_count)
        coverage_signature = _signature_from_points(shell_points, max_points=6)

        templates.append(
            {
                "image": str(image_path),
                "front": front,
                "coverage": coverage,
                "call": call,
                "num_points": len(points),
                "signature": signature,
                "coverage_signature": coverage_signature,
            }
        )
    return templates


def load_or_build_templates(
    defense_dir: Path,
    cache_path: Path,
) -> Dict[str, object]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        templates = payload.get("templates", []) or []
        if (
            payload.get("version") == TEMPLATE_VERSION
            and templates
            and "coverage_signature" in templates[0]
        ):
            return payload

    templates = _build_templates(defense_dir)
    payload: Dict[str, object] = {
        "version": TEMPLATE_VERSION,
        "source": str(defense_dir),
        "template_count": len(templates),
        "templates": templates,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))
    return payload


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return math.sqrt(total)


def match_defense_call(
    defense_players: Sequence[Mapping[str, object]],
    template_payload: Mapping[str, object],
    los_x: float = 0.5,
) -> Dict[str, object]:
    templates = template_payload.get("templates", []) or []
    points = [(float(p.get("x", 0.5)), float(p.get("y", 0.5))) for p in defense_players]
    observed_counts = infer_front_counts(defense_players, los_x=los_x)
    front = format_front(observed_counts)
    if not templates:
        return {
            "call": "Unknown Coverage",
            "front": front,
            "coverage": "Unknown",
            "score": None,
            "front_counts": observed_counts,
        }

    db_count = max(0, len(points) - observed_counts[0] - observed_counts[1])
    live_shell_points = select_shell_points(points, los_x=los_x, db_count=db_count)
    live_coverage_signature = _signature_from_points(live_shell_points, max_points=6)

    best = None
    best_score = float("inf")
    for template in templates:
        template_coverage_signature = template.get("coverage_signature", template.get("signature", []))
        score = _distance(live_coverage_signature, template_coverage_signature)
        if score < best_score:
            best = template
            best_score = score

    if not best:
        return {
            "call": "Unknown Coverage",
            "front": front,
            "coverage": "Unknown",
            "score": None,
            "front_counts": observed_counts,
        }

    template_front = str(best.get("front", "Unknown"))
    front_counts = observed_counts

    return {
        "call": f"{front} {best.get('coverage', 'Unknown')}".strip(),
        "front": front,
        "coverage": str(best.get("coverage", "Unknown")),
        "score": float(best_score),
        "front_counts": front_counts,
        "template_front": template_front,
    }
