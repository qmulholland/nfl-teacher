# Dataset Format

This project uses one JSON object per play.

## Required top-level fields
- `play_id`: unique ID for the play.
- `source_image`: path to the source diagram image.
- `normalization`: metadata proving the play was normalized.
- `players`: list of labeled players.

## Player fields
- `id`: unique ID inside the play.
- `side`: `offense` or `defense`.
- `x`, `y`: normalized coordinates in `[0, 1]`.
- `label`: position name from `config/positions.json`.
- `features` (optional but recommended): engineered geometry values, including:
  - `dist_to_los`
  - `dist_to_sideline` (minimum distance to either sideline)
  - `dist_to_nearest_teammate`
  - `dist_to_nearest_opponent`
  - `dist_to_team_center` (distance to average location of all *other* teammates)
  - `formation_center_dx`, `formation_center_dy`

## Normalization assumptions
- Offense always faces left-to-right.
- LOS is stored as `normalization.los_x`.
- Field bounds are normalized to `[0, 1]`.

## Split files
Split files live in `data/splits/`.
- `train.json`
- `val.json`
- `test.json`

Each split file stores `play_ids` so the same play never appears in multiple splits.
