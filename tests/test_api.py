# tests/test_api.py
import os
from pathlib import Path
import math
import pytest

# 常に利用できる書き出し系（pyarrow経由）
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from api import (
    NotFound,
    get_value,
    get_series,
    get_panel,
    get_item_ids,
    get_years_for_item,
)

# -----------------------
# Helpers
# -----------------------

def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)

def df_like_to_records(df):
    """Polars/Pandas どちらでもレコードの list[dict] に変換"""
    # Polars
    if hasattr(df, "to_dicts"):
        return df.to_dicts()
    # Pandas
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    raise TypeError("Unsupported DF type")

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture
def sample_item_id() -> str:
    return "A110101"

@pytest.fixture
def values_parquet(base: Path, sample_item_id: str) -> Path:
    """標準の values.parquet を作る（重複・NULLを含む）"""
    path = base / "muni_by_item" / f"item={sample_item_id}" / "values.parquet"
    # 行順は「先勝ち」検証のために重要
    pdf = pd.DataFrame(
        {
            "area_code": [13101, 13101, 13101, 13102, 13101, 13103],
            "year":      [2010, 2011, 2011, 2011, 2012, 2010],
            "item_id":   [sample_item_id]*6,
            "value":     [100.0, 200.0, 201.0, math.nan, math.nan, 300.0],
            "quality_flag": [None, "X", "dup", None, "…", None],
            "statsDataId":  ["0001","0001","0001","0001","0002","0001"],
            "class_keys":   ['{"sex":"T"}']*6,
        }
    )
    write_parquet(pdf, path)
    return path

@pytest.fixture
def another_item_parquet(base: Path) -> tuple[str, Path]:
    """別 item の values.parquet（年集合が異なる）"""
    item = "B220202"
    path = base / "muni_by_item" / f"item={item}" / "values.parquet"
    pdf = pd.DataFrame(
        {
            "area_code": [13101, 13102, 13101],
            "year":      [2000, 2000, 2005],
            "item_id":   [item]*3,
            "value":     [1.0, 2.0, 3.0],
            "quality_flag": [None, None, None],
        }
    )
    write_parquet(pdf, path)
    return item, path

@pytest.fixture
def items_catalog(base: Path) -> Path:
    path = base / "catalog" / "items.parquet"
    pdf = pd.DataFrame(
        {
            "item_id": ["B220202", "A110101", "A110101", None],
            "name":    ["x", "y", "dup", "nan"],
        }
    )
    write_parquet(pdf, path)
    return path

# -----------------------
# Tests: get_item_ids / get_years_for_item
# -----------------------

def test_get_item_ids_ok(base: Path, items_catalog: Path):
    ids = get_item_ids(str(base))
    assert ids == ["A110101", "B220202"]  # 昇順・重複/NULL除外

def test_get_item_ids_not_found(base: Path):
    (base / "catalog").mkdir()
    with pytest.raises(NotFound):
        get_item_ids(str(base))

def test_get_years_for_item_ok(base: Path, values_parquet: Path, another_item_parquet):
    years = get_years_for_item(str(base), "A110101")
    assert years == [2010, 2011, 2012]
    # 別 item
    years_b = get_years_for_item(str(base), another_item_parquet[0])
    assert years_b == [2000, 2005]

def test_get_years_for_item_not_found(base: Path):
    with pytest.raises(NotFound):
        get_years_for_item(str(base), "NOPE")

# -----------------------
# Tests: get_value
# -----------------------

def test_get_value_hit_first_occurrence(base: Path, values_parquet: Path):
    # 2011 は重複行（quality_flag: "X" と "dup"）。先勝ちで "X" を返す
    v = get_value(13101, 2011, "A110101", base_dir=base)
    assert v == (200.0, "X")

def test_get_value_nan_and_none(base: Path, values_parquet: Path):
    # value NaN -> None になる（api側の正規化仕様）。2012 は value NaN, qflag "…"
    v = get_value(13101, 2012, "A110101", base_dir=base)
    assert v == (None, "…")
    # area/year 不一致 -> None
    assert get_value(99999, 2011, "A110101", base_dir=base) is None

def test_get_value_not_found(base: Path):
    with pytest.raises(NotFound):
        get_value(13101, 2010, "NOPE", base_dir=base)

# -----------------------
# Tests: get_series
# -----------------------

def test_get_series_years_slice_inclusive(base: Path, values_parquet: Path):
    # slice は両端含む
    df = get_series(13101, slice(2010, 2012), "A110101", base_dir=base)
    recs = df_like_to_records(df)
    assert [r["year"] for r in recs] == [2010, 2011, 2012]
    # 並び昇順、重複は先勝ちで1行のみ
    assert recs[1]["value"] == 200.0 and recs[1]["quality_flag"] == "X"

def test_get_series_years_iterable(base: Path, values_parquet: Path):
    df = get_series(13101, [2012, 2010], "A110101", base_dir=base)
    years = [r["year"] for r in df_like_to_records(df)]
    assert years == [2010, 2012]

def test_get_series_optional_cols(base: Path, values_parquet: Path):
    df = get_series(13101, slice(2010, 2012), "A110101", base_dir=base, include_optional=True)
    recs = df_like_to_records(df)
    assert set(recs[0].keys()) >= {"year", "value", "quality_flag", "statsDataId", "class_keys"}
    # quality_flag の None 正規化（Pandas/Polars両対応のため None 許容で確認）
    assert recs[0]["quality_flag"] is None  # 2010 の qflag は None

# -----------------------
# Tests: get_panel
# -----------------------

def test_get_panel_slice_and_sort(base: Path, values_parquet: Path):
    df = get_panel(slice(2010, 2011), "A110101", base_dir=base)
    recs = df_like_to_records(df)
    # (year, area_code) の昇順
    assert [(r["year"], r["area_code"]) for r in recs] == [(2010, 13101), (2010, 13103), (2011, 13101), (2011, 13102)]
    # 重複 (13101,2011) は先勝ち（value 200.0, qflag "X"）
    row_2011_13101 = [r for r in recs if r["year"] == 2011 and r["area_code"] == 13101][0]
    assert row_2011_13101["value"] == 200.0 and row_2011_13101["quality_flag"] == "X"

def test_get_panel_iterable_years_and_optional(base: Path, values_parquet: Path):
    df = get_panel([2011, 2010], "A110101", base_dir=base, include_optional=True)
    recs = df_like_to_records(df)
    # year 2010 と 2011 のみ、昇順
    assert [r["year"] for r in recs] == [2010, 2010, 2011, 2011]
    # optional 列あり
    assert set(recs[0].keys()) >= {"statsDataId", "class_keys"}

# -----------------------
# Edge cases
# -----------------------

def test_missing_columns_guard(base: Path, sample_item_id: str):
    # year 列欠損の values.parquet
    bad = base / "muni_by_item" / f"item={sample_item_id}" / "values.parquet"
    pdf = pd.DataFrame({"area_code": [1], "value": [1.0]})
    write_parquet(pdf, bad)
    with pytest.raises(ValueError):
        get_years_for_item(str(base), sample_item_id)
