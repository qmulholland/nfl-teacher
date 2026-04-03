#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from defense_catalog import load_or_build_templates, match_defense_call
from inference import enrich_features, infer_play, starter_players


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
POSITIONS_PATH = ROOT / "config" / "positions.json"
PLAYS_DIR = ROOT / "data" / "plays"
SPLITS_DIR = ROOT / "data" / "splits"
DEFENSE_DIR = ROOT / "Defense"
TEMPLATE_CACHE_PATH = ROOT / "data" / "templates" / "defense_templates.json"
MAX_PLAYERS_PER_SIDE = 11

TEMPLATE_PAYLOAD = load_or_build_templates(DEFENSE_DIR, TEMPLATE_CACHE_PATH)


def load_config() -> Dict:
    return json.loads(POSITIONS_PATH.read_text())


def safe_ui_path(request_path: str) -> Path:
    if request_path in ("/", ""):
        return UI_DIR / "index.html"
    rel = request_path.lstrip("/")
    path = (UI_DIR / rel).resolve()
    ui_root = UI_DIR.resolve()
    if not str(path).startswith(str(ui_root)):
        return UI_DIR / "index.html"
    if path.is_dir():
        return path / "index.html"
    return path


def append_play_to_split(play_id: str, split_name: str) -> None:
    split_path = SPLITS_DIR / f"{split_name}.json"
    if not split_path.exists():
        payload = {"split": split_name, "play_ids": []}
    else:
        payload = json.loads(split_path.read_text())
    play_ids = [str(v) for v in payload.get("play_ids", [])]
    if play_id not in play_ids:
        play_ids.append(play_id)
    payload["play_ids"] = play_ids
    split_path.write_text(json.dumps(payload, indent=2))


def count_players_by_side(players: List[Dict]) -> Dict[str, int]:
    counts = {"offense": 0, "defense": 0}
    for player in players:
        side = str(player.get("side", "defense"))
        if side in counts:
            counts[side] += 1
    return counts


def enforce_player_caps(players: List[Dict]) -> None:
    counts = count_players_by_side(players)
    if counts["offense"] > MAX_PLAYERS_PER_SIDE or counts["defense"] > MAX_PLAYERS_PER_SIDE:
        raise ValueError(
            f"Max {MAX_PLAYERS_PER_SIDE} players allowed per side "
            f"(offense={counts['offense']}, defense={counts['defense']})."
        )


def infer_response(players: List[Dict], config: Dict, los_x: float) -> Dict:
    enforce_player_caps(players)
    defense_players = [p for p in players if str(p.get("side", "defense")) == "defense"]
    defense_match = match_defense_call(defense_players, TEMPLATE_PAYLOAD, los_x=los_x)
    inferred_players = infer_play(
        players,
        config=config,
        los_x=los_x,
        defense_front_counts=defense_match.get("front_counts"),
    )
    return {
        "players": inferred_players,
        "los_x": los_x,
        "defense_call": defense_match.get("call"),
        "defense_front": defense_match.get("front"),
        "defense_coverage": defense_match.get("coverage"),
        "defense_match_score": defense_match.get("score"),
        "defense_template_front": defense_match.get("template_front"),
    }


def save_play(payload: Dict, config: Dict) -> Dict:
    players = payload.get("players", [])
    los_x = float(payload.get("los_x", 0.5))
    inference = infer_response(players, config=config, los_x=los_x)
    inferred_players = inference["players"]

    play_id = str(payload.get("play_id") or f"ui_play_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    source_image = str(payload.get("source_image", "ui/interactive"))
    split = str(payload.get("split", "train"))

    normalized_players: List[Dict] = []
    # Persist decoded label as ground truth from current UI confirmation.
    for player in enrich_features(inferred_players, los_x=los_x):
        normalized_players.append(
            {
                "id": str(player["id"]),
                "side": str(player["side"]),
                "x": float(player["x"]),
                "y": float(player["y"]),
                "label": str(player.get("predicted_label") or player.get("label") or "UNKNOWN"),
                "features": player.get("features", {}),
            }
        )

    play_doc = {
        "play_id": play_id,
        "source_image": source_image,
        "defense_call": inference.get("defense_call"),
        "defense_front": inference.get("defense_front"),
        "defense_coverage": inference.get("defense_coverage"),
        "normalization": {
            "offense_direction": "left_to_right",
            "los_x": los_x,
            "field_x_min": 0.0,
            "field_x_max": 1.0,
            "field_y_min": 0.0,
            "field_y_max": 1.0,
        },
        "players": normalized_players,
    }

    PLAYS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLAYS_DIR / f"{play_id}.json"
    out_path.write_text(json.dumps(play_doc, indent=2))
    append_play_to_split(play_id, split_name=split)

    return {
        "play_id": play_id,
        "path": str(out_path.relative_to(ROOT)),
        "split": split,
        "defense_call": inference.get("defense_call"),
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "FormationScoutHTTP/0.1"

    def _send_json(self, payload: Dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            config = load_config()
            self._send_json(
                {
                    "config": config,
                    "los_x": 0.5,
                    "starter_players": starter_players(config),
                    "max_players_per_side": MAX_PLAYERS_PER_SIDE,
                    "template_count": TEMPLATE_PAYLOAD.get("template_count", 0),
                }
            )
            return

        path = safe_ui_path(parsed.path)
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        config = load_config()

        if parsed.path == "/api/infer":
            try:
                payload = self._read_json()
                players = payload.get("players", [])
                los_x = float(payload.get("los_x", 0.5))
                result = infer_response(players, config=config, los_x=los_x)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/save":
            try:
                payload = self._read_json()
                result = save_play(payload, config=config)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.CREATED)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Keep terminal logs concise for local development.
        super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve interactive Formation Scout UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    UI_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Serving Formation Scout at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
