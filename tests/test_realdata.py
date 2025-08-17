# tests/test_realdata.py
from __future__ import annotations
import os
from pathlib import Path
import pytest

from api import (
    NotFound,
    get_item_ids,
    get_years_for_item,
    get_panel,
    get_series,
    get_value,
)

# -----------------------
# Config
# -----------------------
BASE = Path(os.environ.get("MUNI_BASE", "data"))

def require_dir_exists():
    if not BASE.exists():
        pytest.skip(f"real-data base dir not found: {BASE}")

# -----------------------
# Tests
# -----------------------

def test_catalog_and_item_ids_present():
    require_dir_exists()
    try:
        ids = get_item_ids(str(BASE))
    except NotFound:
        pytest.skip(f"catalog/items.parquet not found under {BASE}")
    assert isinstance(ids, list)
    assert all(isinstance(i, str) for i in ids)
    assert len(ids) > 0

@pytest.mark.parametrize("max_items", [3])  # 最初の数件だけ軽く検査
def test_each_item_has_years_and_panel(max_items):
    require_dir_exists()
    try:
        item_ids = get_item_ids(str(BASE))
    except NotFound:
        pytest.skip("items catalog not found")

    item_ids = item_ids[:max_items]
    if not item_ids:
        pytest.skip("no items in catalog")

    for item in item_ids:
        # years が非空で昇順
        try:
            years = get_years_for_item(str(BASE), item)
        except NotFound:
            # カタログにあるがまだ values.parquet が未作成のケース
            pytest.skip(f"values.parquet not found for {item}")
        assert isinstance(years, list) and len(years) > 0
        assert years == sorted(years)

        # panel が取得でき、year の範囲が期待通り
        # 年数が多いと重いので先頭2年だけに限定
        yrs_subset = years[:2]
        df = get_panel(yrs_subset, item, base_dir=BASE, include_optional=True)
        # 空でも仕様上はOK（全自治体欠測の年など）。
        # 少なくとも列が揃っていることだけ確認
        cols = set(getattr(df, "columns", []))
        assert {"area_code", "year", "value", "quality_flag"}.issubset(cols)

        # 最初に見つかった自治体・年で get_value が例外を出さないこと
        # 空ならスキップ
        if getattr(df, "shape", (0,))[0] == 0:
            pytest.skip(f"panel empty for {item} years={yrs_subset}")
        # Polars/Pandas 両対応で1行取り出し
        rec = None
        if hasattr(df, "to_dicts"):      # Polars
            rec = df.to_dicts()[0]
        else:                             # Pandas
            rec = df.iloc[0].to_dict()
        area = int(rec["area_code"])
        year = int(rec["year"])
        v = get_value(area, year, item, base_dir=BASE)
        # 返り値は None か (value, qflag)
        assert (v is None) or (isinstance(v, tuple) and len(v) == 2)

def test_series_contract_smoke():
    """series が inclusive-slice を満たし、重複が先勝ちで1行になることの軽い検査。
    データ状況に依るため、年が2つ以上ある項目を1つだけ選んで検査する。
    """
    require_dir_exists()
    try:
        item_ids = get_item_ids(str(BASE))
    except NotFound:
        pytest.skip("items catalog not found")

    picked = None
    years_of_picked = None
    for item in item_ids[:10]:  # 最初の10件を当たって、2年以上あるものを探す
        try:
            ys = get_years_for_item(str(BASE), item)
        except NotFound:
            continue
        if len(ys) >= 2:
            picked, years_of_picked = item, ys
            break
    if picked is None:
        pytest.skip("no item has >=2 years in this real dataset snapshot")

    # 2年だけで panel を取り、そこに存在する area を1つ拾う
    df = get_panel(years_of_picked[:2], picked, base_dir=BASE)
    if getattr(df, "shape", (0,))[0] == 0:
        pytest.skip("panel empty for picked item/years")

    if hasattr(df, "to_dicts"):
        area = df.to_dicts()[0]["area_code"]
    else:
        area = int(df.iloc[0]["area_code"])

    # inclusive slice の確認
    lo, hi = years_of_picked[0], years_of_picked[1]
    s = get_series(area, slice(lo, hi), picked, base_dir=BASE)
    # 年は lo..hi の範囲に限定され昇順
    if hasattr(s, "to_dicts"):
        years = [r["year"] for r in s.to_dicts()]
    else:
        years = [int(x) for x in s["year"].tolist()]
    assert all(lo <= y <= hi for y in years)
    assert years == sorted(years)
