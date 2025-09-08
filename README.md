# sensor_calibration

Minimal pipeline for extracting longest stable "shelves" per degree and producing calibration points with errors.

Quick start (CLI):

- Ensure `combined_temperatures.csv` exists (produced by the notebook or your pipeline).
- Compute calibration points (one per shelf and per sensor) and plots:
  - `python recompute_stability.py`

Environment variables (optional):

- `REF_IDX` (default `8`): reference sensor index (uses column `T{REF_IDX}`).
- `N_FOLLOW` (default `3`): number of follower sensors (e.g., `T9..T11` when `REF_IDX=8`).
- `WINDOW_N` (default `20`): rolling window size in samples.
- `STD_THR` (default `0.05`): rolling std threshold for stability.
- `DIFF_THR` (default `0.002`): rolling mean absolute first-diff threshold (slope gate).
- `MIN_LEN` (default `20`): minimal interval length in samples.
- `GROUP_BY_FILE` (default `1`): treat each `source_file` independently.
- `SPLIT_BY_DEG` (default `1`): split long segments by reference degree buckets.
- `DEG_TOL` (default `1.0`): bucket size in °C for splitting/selection.
- `MAX_REF_DRIFT` (default `0.3`): maximum allowed drift on reference within an interval (°C). Use `None` to disable.
- `SELECT_LONGEST_PER_DEGREE` (default `0`): optionally keep only the longest per bucket.
- `CLUSTER_STRATEGY` (default `sliding`): `sliding` or `bucket` when selecting longest.

Outputs:

- `calibration_points_by_sensor.csv` — одна X–Y точка на «полочку» и датчик (Y — эталон, X — калибруемый, ошибки — std по X и Y, а также размеры окон и даты).
- `calibration_T*.png` — по одному PNG на датчик: верхний график — X–Y с error bars; нижний — столбчатые std(X) и std(Y) по центрам полочек.

Notes:

- Детекция: rolling std + ограничение по среднему |ΔT| (наклон), разрезка по «градусным полочкам», фильтр по дрейфу опоры.
- Отбор: автоматически оставляем один (самый длинный) интервал на каждую полочку в пределах файла и датчика.
