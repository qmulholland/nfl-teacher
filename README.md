# Formation Scout: Alignment Classification System
An interactive-first platform that classifies football alignments for both offense and defense using normalized player coordinates, engineered features, and neural networks.

## What Is Done
- Unified player label vocabulary for offense and defense.
- Standard JSON format for every play.
- Starter train/val/test split files.
- Constraint-based decoding so assignments depend on all players in the formation.
- Interactive drag-and-drop UI with live relabeling and dataset save support.
- Defense template matcher built from real images in `Defense/`:
  - `3-2-6_Mug` + `3-2-6_Odd` are simplified to `3-2-6`.
  - `COVER_1_ROBBER_PRESS` is simplified to `Cover 1`.
  - Output call format is `Front + Coverage` (example: `3-2-6 Cover 1`).
  - Front is geometry-aware in live UI (so a true 4-2-5 alignment is reported as `4-2-5` even if the closest template folder variant differs).
- Field direction helpers in UI: LOS line, first-down line, and end-zone direction labels.
- Hard cap of 11 players per side in UI and API.

## Project Structure
- `config/positions.json`: master list of allowed labels and simple lineup constraints.
- `data/schemas/play.schema.json`: JSON Schema for one normalized play.
- `data/plays/`: normalized play JSON files used by training/evaluation.
- `data/splits/train.json`: list of play IDs in train split.
- `data/splits/val.json`: list of play IDs in validation split.
- `data/splits/test.json`: list of play IDs in test split.
- `scripts/normalize_play.py`: canonical normalization script.
- `model_mlp.py`: baseline PyTorch MLP classifier.
- `train.py`: baseline training script.
- `evaluate.py`: accuracy + confusion matrix evaluation script.
- `decoder.py`: lineup constraint solver (one QB, cap-limited labels, etc.).
- `inference.py`: relational feature scoring + probabilities + decoding.
- `defense_catalog.py`: extracts white defender markers from `Defense/*.png`, simplifies names, and matches formation/coverage templates.
- `app.py`: local API + static web server for the interactive UI.
- `ui/index.html`, `ui/app.js`, `ui/styles.css`: drag-and-drop interface.
- `docs/dataset_format.md`: plain-English explanation of the dataset.

## Run The Interactive UI
1. Start the local app server:
   - `python3 app.py --host 127.0.0.1 --port 8000`
2. Open:
   - `http://127.0.0.1:8000`
3. Drag players on the field.
4. Watch all predicted labels update live after every move.
5. See the simplified defensive call update live (example: `3-2-6 Cover 1`).
6. Click **Save Current Play** to write a normalized play JSON into `data/plays/` and append its ID to the chosen split file.

## Model + Data Workflow
1. Add labeled plays from your real workflow (UI save or your own source JSON).
2. Normalize each raw play into canonical format:
   - `python3 scripts/normalize_play.py --input path/to/raw_play.json --output data/plays/<play_id>.json`
3. Keep all labels inside `config/positions.json` so the model always trains on a fixed class list.
4. Put each play ID into one split file (`train.json`, `val.json`, or `test.json`).
5. Train the MLP baseline:
   - `python3 train.py --data-dir data/plays --train-split data/splits/train.json --val-split data/splits/val.json`
6. Evaluate the trained checkpoint:
   - `python3 evaluate.py --checkpoint artifacts/mlp_checkpoint.pt --data-dir data/plays --split data/splits/train.json`

## Interactive Behavior
- A player drag triggers full-formation inference, not isolated single-player inference.
- Relational features are recalculated (formation center, nearest teammate/opponent, LOS distance, sideline distance, and distance-to-team-center) for every player after each drag.
- The decoder enforces lineup constraints, so one player’s movement can change other players’ assigned positions.
- Defensive call matching uses the real white-marker geometry from your `Defense/` image templates.
