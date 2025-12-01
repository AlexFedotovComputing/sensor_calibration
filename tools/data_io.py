"""Helpers for loading calibration measurement tables."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

FileLike = Union[str, Tuple[str, bytes]]


def sniff_sep(sample: bytes) -> str:
    """Detect the delimiter in a CSV sample; fallback to comma."""
    try:
        dialect = csv.Sniffer().sniff(
            sample.decode("utf-8", errors="ignore"),
            delimiters=[",", ";", "\t", "|"],
        )
        return dialect.delimiter
    except Exception:
        line = sample.decode("utf-8", errors="ignore").splitlines()[0] if sample else ""
        for cand in (",", ";", "\t", "|"):
            if line.count(cand) >= 1:
                return cand
        return ","


def read_one_table(name: str, stream: io.BytesIO, date_format: Optional[str] = None) -> pd.DataFrame:
    """Read one measurement table into a normalized DataFrame."""
    head = stream.read(8192)
    stream.seek(0)
    sep = sniff_sep(head)
    df = pd.read_csv(stream, sep=sep, engine="python")
    if df.shape[1] < 17:
        raise ValueError(
            f"{name}: found {df.shape[1]} columns, expected >= 17 (1 date + 16 temperatures)."
        )
    df = df.iloc[:, :17].copy()
    df.columns = ["date"] + [f"T{i}" for i in range(16)]
    if date_format and date_format.strip():
        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    else:
        df["date"] = pd.to_datetime(
            df["date"], infer_datetime_format=True, errors="coerce", dayfirst=True
        )
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        print(f"[warn] {name}: dropped {bad} rows with unparsed dates.")
        df = df.dropna(subset=["date"])
    for c in [f"T{i}" for i in range(16)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source_file"] = name
    return df


def _filter_temperature_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Keep date/T* columns and optional source_file, preserving order."""
    t_cols = [c for c in data.columns if isinstance(c, str) and c.startswith("T")]
    cols_keep = ["date"] + t_cols + (["source_file"] if "source_file" in data.columns else [])
    return data[[c for c in cols_keep if c in data.columns]]


def load_measurements(
    selected_files: Optional[Sequence[FileLike]],
    date_format: Optional[str],
    combined_path: Union[str, Path] = "combined_temperatures.csv",
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Load measurements either from a cached CSV or provided files."""
    combined_path = Path(combined_path)
    errors: List[Tuple[str, str]] = []
    selected_list = list(selected_files) if selected_files else []

    if selected_list:
        frames: List[pd.DataFrame] = []
        for item in selected_list:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    name, content = item
                    stream = io.BytesIO(content)
                else:
                    path_obj = Path(item)
                    name = path_obj.name
                    stream = io.BytesIO(path_obj.read_bytes())
                df = read_one_table(name, stream, date_format=date_format)
                frames.append(df)
            except Exception as exc:
                errors.append((str(item), str(exc)))

        if not frames:
            raise RuntimeError("No files could be read successfully.")

        data = (
            pd.concat(frames, ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
        data = _filter_temperature_columns(data)
        return data, errors

    if combined_path.exists():
        data = pd.read_csv(combined_path)
        data = _filter_temperature_columns(data)
        return data.copy(), errors

    raise RuntimeError(
        "No input data: select files in the notebook or provide combined_temperatures.csv next to it."
    )

