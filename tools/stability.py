"""Utilities for detecting stability intervals and selecting calibration segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class StabilityConfig:
    window: int = 20
    std_thr: float = 0.05
    diff_thr: float = 0.002
    min_len: int = 20
    group_by_file: bool = True
    split_by_ref_buckets: bool = True
    deg_tol: float = 1.0
    max_ref_drift: Optional[float] = 0.3


def rolling_std_mask(series: pd.Series, window: int, threshold: float) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std() <= threshold


def rolling_mean_abs_diff_mask(series: pd.Series, window: int, diff_threshold: float) -> pd.Series:
    diffs = series.diff().abs()
    return diffs.rolling(window=window, min_periods=window).mean() <= diff_threshold


def segments_from_mask(mask: Sequence[bool], window: int) -> List[Tuple[int, int]]:
    arr = mask.to_numpy() if hasattr(mask, 'to_numpy') else np.asarray(mask)
    segments: List[Tuple[int, int]] = []
    current: Optional[Tuple[int, int]] = None
    for idx, ok in enumerate(arr):
        if ok:
            start = max(0, idx - window + 1)
            end = idx
            if current is None:
                current = (start, end)
            else:
                cur_start, cur_end = current
                if start <= cur_end + 1:
                    current = (cur_start, max(cur_end, end))
                else:
                    segments.append(current)
                    current = (start, end)
        elif current is not None:
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def summarize_interval(df: pd.DataFrame, cols: Sequence[str], s: int, e: int) -> Dict[str, float]:
    row: Dict[str, float] = {
        'start_idx': int(s),
        'end_idx': int(e),
        'length': int(e - s + 1),
        'start_date': pd.to_datetime(df.loc[s, 'date']) if s < len(df) else pd.NaT,
        'end_date': pd.to_datetime(df.loc[e, 'date']) if e < len(df) else pd.NaT,
    }
    for col in cols:
        vals = df[col].to_numpy()[s : e + 1]
        good = ~np.isnan(vals)
        row[f'mean_{col}'] = float(np.nanmean(vals)) if good.any() else np.nan
        row[f'std_{col}'] = float(np.nanstd(vals, ddof=1)) if good.sum() > 1 else np.nan
        row[f'min_{col}'] = float(np.nanmin(vals)) if good.any() else np.nan
        row[f'max_{col}'] = float(np.nanmax(vals)) if good.any() else np.nan
        row[f'drift_{col}'] = row[f'max_{col}'] - row[f'min_{col}'] if good.any() else np.nan
    return row


def median_level(series: pd.Series, s: int, e: int) -> float:
    vals = series.to_numpy()[int(s) : int(e) + 1]
    return float(np.nanmedian(vals))


def split_segment_by_ref_buckets(ref_series: pd.Series, s: int, e: int, deg_tol: float) -> List[Tuple[int, int]]:
    vals = ref_series.to_numpy()[s : e + 1]
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return [(s, e)]
    buckets = np.floor(vals / float(deg_tol)).astype('float64')
    mask_good = ~np.isnan(buckets)
    if mask_good.any():
        last = None
        for idx in range(len(buckets)):
            if np.isnan(buckets[idx]):
                buckets[idx] = last if last is not None else buckets[mask_good][0]
            last = buckets[idx]
    segments: List[Tuple[int, int]] = []
    start = s
    base = buckets[0]
    for idx in range(1, len(buckets)):
        if buckets[idx] != base:
            segments.append((start, s + idx - 1))
            start = s + idx
            base = buckets[idx]
    segments.append((start, e))
    return segments


def detect_stability_improved(
    data: pd.DataFrame,
    ref_idx: int,
    follow_idxs: Sequence[int],
    *,
    window: int,
    std_thr: float,
    diff_thr: float,
    min_len: int,
    group_by_file: bool,
    split_by_ref_buckets: bool,
    deg_tol: float,
    max_ref_drift: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_col = f'T{ref_idx}'
    follow_cols = [f'T{i}' for i in follow_idxs]
    for col in [ref_col] + follow_cols:
        if col not in data.columns:
            raise ValueError(f'Missing column: {col}')

    if group_by_file and 'source_file' in data.columns:
        first_idx_by_source = (
            data.index.to_series().groupby(data['source_file']).min().astype(int).to_dict()
        )
        groups = list(data.groupby('source_file', sort=False))
    else:
        first_idx_by_source = None
        groups = [('ALL', data)]

    joint_rows: List[Dict[str, float]] = []
    by_sensor_rows: List[Dict[str, float]] = []

    for src, group in groups:
        g = group.reset_index(drop=True)
        mask_ref_std = rolling_std_mask(g[ref_col], window, std_thr)
        mask_ref_slope = rolling_mean_abs_diff_mask(g[ref_col], window, diff_thr)
        base_ref_mask = (mask_ref_std & mask_ref_slope).to_numpy()
        masks_others = {col: rolling_std_mask(g[col], window, std_thr).to_numpy() for col in follow_cols}
        joint_mask = base_ref_mask.copy()
        for col in follow_cols:
            joint_mask &= masks_others[col]

        def emit_segment(start: int, end: int, cols: Sequence[str], extra: Dict[str, str], container: List[Dict[str, float]]):
            segments = split_segment_by_ref_buckets(g[ref_col], start, end, deg_tol) if split_by_ref_buckets else [(start, end)]
            for s_idx, e_idx in segments:
                if (e_idx - s_idx + 1) < int(min_len):
                    continue
                row = {'source_file': src}
                row.update(extra)
                row.update(summarize_interval(g, cols, s_idx, e_idx))
                if src != 'ALL' and first_idx_by_source is not None:
                    first_idx = int(first_idx_by_source.get(src, 0))
                    row['start_idx_abs'] = row['start_idx'] + first_idx
                    row['end_idx_abs'] = row['end_idx'] + first_idx
                else:
                    row['start_idx_abs'] = row['start_idx']
                    row['end_idx_abs'] = row['end_idx']
                ref_med = median_level(g[ref_col], s_idx, e_idx)
                row['ref_level'] = ref_med
                row['drift_ref'] = row.get(f'drift_{ref_col}', np.nan)
                if max_ref_drift is not None and not np.isnan(row['drift_ref']) and row['drift_ref'] > max_ref_drift:
                    continue
                container.append(row)

        for seg_start, seg_end in [seg for seg in segments_from_mask(joint_mask, window) if (seg[1] - seg[0] + 1) >= int(min_len)]:
            emit_segment(seg_start, seg_end, [ref_col] + follow_cols, {'ref': ref_col, 'followers': ','.join(follow_cols)}, joint_rows)

        for col in follow_cols:
            gated = masks_others[col] & base_ref_mask
            for seg_start, seg_end in [seg for seg in segments_from_mask(gated, window) if (seg[1] - seg[0] + 1) >= int(min_len)]:
                emit_segment(seg_start, seg_end, [col, ref_col], {'sensor': col, 'ref': ref_col}, by_sensor_rows)

    joint = pd.DataFrame(joint_rows).sort_values(['source_file', 'start_idx']) if joint_rows else pd.DataFrame()
    by_sensor = pd.DataFrame(by_sensor_rows).sort_values(['source_file', 'sensor', 'start_idx']) if by_sensor_rows else pd.DataFrame()
    return joint, by_sensor


def select_longest_per_degree(
    table: Optional[pd.DataFrame],
    data: pd.DataFrame,
    ref_idx: int,
    mode: str,
    *,
    deg_tol: float,
    strategy: str = 'bucket_centered',
    centered: bool = True,
) -> Optional[pd.DataFrame]:
    if table is None or table.empty:
        return table

    ref_col = f'T{ref_idx}'
    tbl = table.copy()

    if 'ref_level' not in tbl.columns:
        ref_levels = []
        for _, row in tbl.iterrows():
            src = row['source_file']
            start = int(row['start_idx'])
            end = int(row['end_idx'])
            subset = data[data['source_file'] == src].reset_index(drop=True) if src != 'ALL' else data
            ref_levels.append(median_level(subset[ref_col], start, end))
        tbl['ref_level'] = ref_levels

    group_keys = ['source_file', 'sensor'] if (mode == 'by_sensor' and 'sensor' in tbl.columns) else ['source_file']
    results: List[pd.DataFrame] = []

    for _, grp in tbl.sort_values('ref_level').groupby(group_keys, as_index=False):
        g = grp.sort_values('ref_level').reset_index(drop=True)
        if strategy in ('bucket', 'bucket_centered') or centered:
            bins = (
                np.floor((g['ref_level'] + 0.5 * deg_tol) / deg_tol).astype(int)
                if (strategy == 'bucket_centered' or centered)
                else np.floor(g['ref_level'] / deg_tol).astype(int)
            )
            g = g.assign(_bin=bins)
            keep = g.sort_values('length', ascending=False).groupby('_bin', as_index=False).head(1)
            results.append(keep.drop(columns=['_bin']))
        else:
            i = 0
            n = len(g)
            while i < n:
                start_val = g.loc[i, 'ref_level']
                j = i
                while j + 1 < n and (g.loc[j + 1, 'ref_level'] - start_val) <= deg_tol:
                    j += 1
                cluster = g.loc[i:j].copy()
                keep = cluster.sort_values(['length', 'end_idx'], ascending=[False, False]).iloc[0:1]
                results.append(keep)
                i = j + 1

    return pd.concat(results, ignore_index=True) if results else tbl
