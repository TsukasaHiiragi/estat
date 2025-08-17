
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, List

# Engine selection
try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    _HAS_POLARS = False
    import pandas as pd  # type: ignore
    import pyarrow.parquet as pq  # type: ignore


# =========================
# Exceptions
# =========================
class NotFound(FileNotFoundError):
    """Raised when the expected data file is not found."""
    pass


# =========================
# Helpers (engine-agnostic)
# =========================
def _base_path(base_dir: Union[str, Path]) -> Path:
    return Path(base_dir)


def _values_path(base_dir: Union[str, Path], item_id: str) -> Path:
    return _base_path(base_dir) / "muni_by_item" / f"item={item_id}" / "values.parquet"


def _catalog_items_path(base_dir: Union[str, Path]) -> Path:
    return _base_path(base_dir) / "catalog" / "items.parquet"


def _catalog_areas_path(base_dir: Union[str, Path]) -> Path:
    return _base_path(base_dir) / "catalog" / "areas.parquet"


def _ensure_exists(path: Path):
    if not path.exists():
        raise NotFound(str(path))


def _years_to_list(years: Union[Iterable[int], slice]) -> List[int]:
    if isinstance(years, slice):
        if years.start is None or years.stop is None:
            raise ValueError("slice must have both start and stop (closed interval).")
        step = years.step or 1
        if step <= 0:
            raise ValueError("slice step must be positive.")
        # Closed interval
        return list(range(int(years.start), int(years.stop) + 1, int(step)))
    # iterable
    ys = sorted({int(y) for y in years})
    return ys


def _warn_dupes_pl(df: "pl.DataFrame", key_cols: List[str]) -> "pl.DataFrame":
    if df.is_empty():
        return df
    dupes = (
        df.select([pl.all()])
          .with_columns(pl.len().over(key_cols).alias("_cnt"))
          .filter(pl.col("_cnt") > 1)
    )
    if dupes.height > 0:
        eg = dupes.select(key_cols).unique().head(1).to_dicts()[0]
        warnings.warn(
            f"Duplicate keys detected (taking first occurrence). example={eg}, n_dupe_rows={dupes.height}",
            RuntimeWarning,
        )
    return df.unique(subset=key_cols, keep="first", maintain_order=True)


def _warn_dupes_pd(df: "pd.DataFrame", key_cols: List[str]) -> "pd.DataFrame":
    if df.empty:
        return df
    n_before = len(df)
    n_unique = df.drop_duplicates(subset=key_cols, keep="first").shape[0]
    if n_before != n_unique:
        # find one example
        dup = df[df.duplicated(subset=key_cols, keep=False)].iloc[0][key_cols].to_dict()
        warnings.warn(
            f"Duplicate keys detected (taking first occurrence). example={dup}, n_dupe_rows={n_before - n_unique}",
            RuntimeWarning,
        )
    return df.drop_duplicates(subset=key_cols, keep="first")


# =========================
# Engine-specific Readers
# =========================
def _read_values_for_item_pl(
    base_dir: Union[str, Path],
    item_id: str,
    years: Optional[List[int]] = None,
    columns: Optional[List[str]] = None,
) -> "pl.DataFrame":
    path = _values_path(base_dir, item_id)
    _ensure_exists(path)
    lazy = pl.scan_parquet(str(path))
    sel_cols = columns or ["area_code", "year", "item_id", "value", "quality_flag"]
    # Select only existing columns (quality_flag/item_id may be missing)
    existing = [c for c in sel_cols if c in lazy.columns]
    lazy = lazy.select(existing)
    if years is not None:
        lazy = lazy.filter(pl.col("year").is_in(years))
    df = lazy.collect()
    # Ensure required columns present
    if "quality_flag" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("quality_flag"))
    if "item_id" not in df.columns:
        df = df.with_columns(pl.lit(item_id).alias("item_id"))
    # Deduplicate by (area_code, year)
    df = _warn_dupes_pl(df, ["area_code", "year"])
    # Sort
    df = df.sort(["year", "area_code"])
    return df


def _read_values_for_item_pd(
    base_dir: Union[str, Path],
    item_id: str,
    years: Optional[List[int]] = None,
    columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    path = _values_path(base_dir, item_id)
    _ensure_exists(path)
    sel_cols = columns or ["area_code", "year", "item_id", "value", "quality_flag"]
    # Pandas cannot select non-existent columns; read minimal then add missing.
    df_all = pd.read_parquet(path, columns=[c for c in sel_cols if c not in ("quality_flag",)])
    if "quality_flag" in sel_cols:
        try:
            q = pd.read_parquet(path, columns=["quality_flag"])
            df_all = df_all.join(q)
        except Exception:
            df_all["quality_flag"] = None
    if years is not None:
        df_all = df_all[df_all["year"].isin(years)]
    if "item_id" not in df_all.columns:
        df_all["item_id"] = item_id
    # Deduplicate by (area_code, year) keep first
    df_all = _warn_dupes_pd(df_all, ["area_code", "year"])
    df_all = df_all.sort_values(["year", "area_code"]).reset_index(drop=True)
    return df_all


# =========================
# Public API
# =========================
def get_value(
    area_code: int,
    year: int,
    item_id: str,
    *,
    base_dir: Union[str, Path] = "data",
) -> Optional[Tuple[Optional[float], Optional[str]]]:
    """
    Return (value, quality_flag) for an exact key or None if key not found.
    Raises NotFound if the item file does not exist.
    """
    if _HAS_POLARS:
        df = _read_values_for_item_pl(base_dir, item_id, years=[year], columns=["area_code", "year", "value", "quality_flag"])
        df = df.filter(pl.col("area_code") == int(area_code))
        if df.height == 0:
            return None
        row = df.row(0, named=True)
        return (row.get("value"), row.get("quality_flag"))
    else:
        df = _read_values_for_item_pd(base_dir, item_id, years=[year], columns=["area_code", "year", "value", "quality_flag"])
        df = df[df["area_code"] == int(area_code)]
        if df.empty:
            return None
        r = df.iloc[0]
        return (r.get("value"), r.get("quality_flag"))


def get_series(
    area_code: int,
    years: Union[Iterable[int], slice],
    item_id: str,
    *,
    base_dir: Union[str, Path] = "data",
):
    """
    Return a DataFrame with columns: year, value, quality_flag (ascending by year).
    Missing years are not padded; only present rows are returned.
    """
    ys = _years_to_list(years)
    if _HAS_POLARS:
        df = _read_values_for_item_pl(base_dir, item_id, years=ys, columns=["area_code", "year", "value", "quality_flag"])
        df = df.filter(pl.col("area_code") == int(area_code)).select(["year", "value", "quality_flag"]).sort("year")
        return df
    else:
        df = _read_values_for_item_pd(base_dir, item_id, years=ys, columns=["area_code", "year", "value", "quality_flag"])
        df = df[df["area_code"] == int(area_code)][["year", "value", "quality_flag"]].sort_values("year").reset_index(drop=True)
        return df


def get_panel(
    years: Union[Iterable[int], slice],
    item_id: Union[str, List[str]],
    *,
    base_dir: Union[str, Path] = "data",
    drop_any_null: bool = False,
    exclude_city_ending_00: bool = False,
):
    """
    Panel for one or multiple items.

    When item_id is a string:
      -> long form: columns [area_code, year, value, quality_flag]
    When item_id is a list[str]:
      -> wide form: [area_code, year, value__A, qflag__A, value__B, qflag__B, ...]
         Merge key is (area_code, year) via inner join.
    """
    ys = _years_to_list(years)

    def _single(item: str):
        if _HAS_POLARS:
            df = _read_values_for_item_pl(base_dir, item, years=ys, columns=["area_code", "year", "value", "quality_flag"])
            # sort ensured
            return df
        else:
            df = _read_values_for_item_pd(base_dir, item, years=ys, columns=["area_code", "year", "value", "quality_flag"])
            return df

    if isinstance(item_id, str):
        df = _single(item_id)
        # Exclude parent city (area_code ending with 00)
        if exclude_city_ending_00:
            if _HAS_POLARS:
                df = df.filter((pl.col("area_code") % 100) != 0)
            else:
                df = df[df["area_code"] % 100 != 0].reset_index(drop=True)
        # Drop rows where value is null
        if drop_any_null:
            if _HAS_POLARS:
                df = df.filter(~pl.col("value").is_null() & ~pl.col("value").is_nan())
            else:
                df = df[df["value"].notna()].reset_index(drop=True)
        # Ensure ordering
        if _HAS_POLARS:
            return df.sort(["year", "area_code"])
        else:
            return df.sort_values(["year", "area_code"]).reset_index(drop=True)

    # Multiple items -> wide form
    items = list(item_id)
    if len(items) == 0:
        # empty -> return empty frame
        if _HAS_POLARS:
            return pl.DataFrame(schema={"area_code": pl.Int32, "year": pl.Int16})
        else:
            import pandas as pd  # type: ignore
            return pd.DataFrame(columns=["area_code", "year"])

    # Build base with first item
    base = _single(items[0])
    if _HAS_POLARS:
        base = base.rename({"value": f"value__{items[0]}", "quality_flag": f"qflag__{items[0]}"})
    else:
        base = base.rename(columns={"value": f"value__{items[0]}", "quality_flag": f"qflag__{items[0]}"})

    # Iteratively join others
    for it in items[1:]:
        df = _single(it)
        if _HAS_POLARS:
            df = df.rename({"value": f"value__{it}", "quality_flag": f"qflag__{it}"})
            base = base.join(df, on=["area_code", "year"], how="inner")
        else:
            df = df.rename(columns={"value": f"value__{it}", "quality_flag": f"qflag__{it}"})
            base = base.merge(df, on=["area_code", "year"], how="inner")

    # Apply filters
    if exclude_city_ending_00:
        if _HAS_POLARS:
            base = base.filter((pl.col("area_code").cast(pl.Int32) % 100) != 0)
        else:
            base = base[base["area_code"].astype(int) % 100 != 0].reset_index(drop=True)

    if drop_any_null:
        # find value columns
        if _HAS_POLARS:
            val_cols = [c for c in base.columns if c.startswith("value__")]
            if val_cols:
                cond = pl.all_horizontal([~pl.col(c).is_null() & ~pl.col(c).is_nan() for c in val_cols])
                base = base.filter(cond)
        else:
            val_cols = [c for c in base.columns if c.startswith("value__")]
            if val_cols:
                mask = True
                for c in val_cols:
                    mask = mask & base[c].notna()
                base = base[mask].reset_index(drop=True)

    # Sort
    if _HAS_POLARS:
        base = base.sort(["year", "area_code"])
    else:
        base = base.sort_values(["year", "area_code"]).reset_index(drop=True)
    return base


def get_item_ids(base_dir: Union[str, Path] = "data") -> List[str]:
    """
    Enumerate item_id from catalog/items.parquet if present.
    Falls back to scanning muni_by_item directories.
    """
    items_path = _catalog_items_path(base_dir)
    if _HAS_POLARS:
        if items_path.exists():
            df = pl.read_parquet(str(items_path), columns=["item_id"])
            df = df.filter(~pl.col("item_id").is_null())
            ids = sorted({str(x) for x in df.get_column("item_id").to_list()})
            return ids
        # fallback
        root = _base_path(base_dir) / "muni_by_item"
        if not root.exists():
            return []
        ids = []
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("item="):
                ids.append(p.name.split("=", 1)[1])
        return sorted(ids)
    else:
        if items_path.exists():
            import pandas as pd  # type: ignore
            df = pd.read_parquet(items_path, columns=["item_id"])
            df = df[df["item_id"].notna()]
            return sorted({str(x) for x in df["item_id"].tolist()})
        root = _base_path(base_dir) / "muni_by_item"
        if not root.exists():
            return []
        ids = []
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("item="):
                ids.append(p.name.split("=", 1)[1])
        return sorted(ids)


def get_years_for_item(item_id: str, base_dir: Union[str, Path]) -> List[int]:
    """
    Read years list for an item from catalog/items.parquet first.
    If not available, scan the item's values.parquet for distinct years.
    """
    items_path = _catalog_items_path(base_dir)
    if _HAS_POLARS:
        if items_path.exists():
            df = pl.read_parquet(str(items_path))
            if "years" in df.columns:
                match = df.filter(pl.col("item_id") == item_id)
                if match.height > 0:
                    ys = match.select("years").to_series().to_list()[0]
                    # ensure ints and sorted unique
                    return sorted({int(y) for y in ys})
        # fallback to values.parquet
        df = _read_values_for_item_pl(base_dir, item_id, years=None, columns=["year"])
        return sorted({int(y) for y in df.get_column("year").to_list()})
    else:
        import pandas as pd  # type: ignore
        if items_path.exists():
            df = pd.read_parquet(items_path)
            if "years" in df.columns:
                row = df[df["item_id"] == item_id]
                if not row.empty:
                    ys = row.iloc[0]["years"]
                    if isinstance(ys, (list, tuple)):
                        return sorted({int(y) for y in ys})
        # fallback
        df = _read_values_for_item_pd(base_dir, item_id, years=None, columns=["year"])
        return sorted({int(y) for y in df["year"].tolist()})


def item_id_to_name( item_id: str, base_dir: Union[str, Path]) -> Optional[str]:
    """
    Map item_id to human-readable name using catalog/items.parquet.
    Returns None if not available.
    """
    items_path = _catalog_items_path(base_dir)
    if _HAS_POLARS:
        if not items_path.exists():
            return None
        df = pl.read_parquet(str(items_path))
        if "name" not in df.columns:
            return None
        match = df.filter(pl.col("item_id") == item_id)
        if match.height == 0:
            return None
        return match.item(0, "name")
    else:
        import pandas as pd  # type: ignore
        if not items_path.exists():
            return None
        df = pd.read_parquet(items_path)
        if "name" not in df.columns:
            return None
        row = df[df["item_id"] == item_id]
        if row.empty:
            return None
        return str(row.iloc[0]["name"])


__all__ = [
    "NotFound",
    "get_value",
    "get_series",
    "get_panel",
    "get_item_ids",
    "get_years_for_item",
    "item_id_to_name",
]
