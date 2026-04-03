#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_data import FEATURE_KEYS, load_examples_from_split, load_positions
from model_mlp import PositionMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline MLP player-position classifier.")
    parser.add_argument("--positions", default="config/positions.json")
    parser.add_argument("--data-dir", default="data/plays")
    parser.add_argument("--train-split", default="data/splits/train.json")
    parser.add_argument("--val-split", default="data/splits/val.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="artifacts/mlp_checkpoint.pt")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * len(x_batch)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y_batch).sum().item())
            total_examples += len(x_batch)
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    labels = load_positions(args.positions)
    x_train, y_train = load_examples_from_split(args.data_dir, args.train_split, labels)

    try:
        x_val, y_val = load_examples_from_split(args.data_dir, args.val_split, labels)
        has_val = True
    except ValueError:
        x_val, y_val = x_train, y_train
        has_val = False

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = PositionMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(labels),
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "feature_keys": FEATURE_KEYS,
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)

    meta_path = ckpt_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "num_labels": len(labels),
                "labels": labels,
                "input_dim": int(x_train.shape[1]),
                "used_validation_split": has_val,
                "best_val_acc": best_val_acc,
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metadata:   {meta_path}")


if __name__ == "__main__":
    main()
