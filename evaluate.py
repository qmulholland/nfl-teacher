#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml_data import build_confusion_matrix, load_examples_from_split, load_positions
from model_mlp import PositionMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/mlp_checkpoint.pt")
    parser.add_argument("--positions", default="config/positions.json")
    parser.add_argument("--data-dir", default="data/plays")
    parser.add_argument("--split", default="data/splits/test.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    labels = load_positions(args.positions)
    x_eval, y_eval = load_examples_from_split(args.data_dir, args.split, labels)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = PositionMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=len(checkpoint["labels"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x_eval)
        preds = torch.argmax(logits, dim=1)

    accuracy = float((preds == y_eval).float().mean().item())
    cm = build_confusion_matrix(y_eval, preds, num_classes=len(labels))

    print(f"Split file: {args.split}")
    print(f"Examples:   {len(y_eval)}")
    print(f"Accuracy:   {accuracy:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm.tolist())


if __name__ == "__main__":
    main()
