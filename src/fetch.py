#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch items from an Excel list and save per-item Parquet.

- Partition per item:  out_dir / item=<ITEM_ID> / values.parquet
- Columns: area_code, year, item_id, value, quality_flag, statsDataId, time_code
- Resolves all time codes via getMetaInfo
- Does NOT set lvArea; instead, filters to municipal/ward area codes on client side
- Finds statsDataId per ITEM by scanning getStatsList(searchWord="市区町村データ 基礎データ")
  and confirming cat01 membership in getMetaInfo (cached for speed).

Usage:
  python fetch.py --appid $ESTAT_APPID --excel kiso_shi.xlsx --out data/muni_by_item --skip-existing

Notes:
- Requires: pandas, requests, (pyarrow or fastparquet recommended for Parquet I/O)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# Optional: pyarrow for fast row-count check
try:
    import pyarrow.parquet as _pq  # type: ignore
except Exception:
    _pq = None

ESTAT_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"

# Item ID pattern like "A4200", "B120103"
ITEM_RE = re.compile(r'^[A-J]\d{4,}$')

CACHE_DIR = Path(".estat_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
CATALOG_FILE = CACHE_DIR / "item_to_stats.json"  # {item_id: statsDataId}


# -----------------------------
# HTTP helpers
# -----------------------------
def _http_get(endpoint: str, params: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    url = f"{ESTAT_BASE}/{endpoint}"
    headers = {"User-Agent": "estat-fetch/1.1"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    # unwrap inner envelope if present
    return js


def get_meta_info(appid: str, stats_id: str) -> Dict[str, Any]:
    return _http_get("getMetaInfo", {"appId": appid, "statsDataId": stats_id})


def get_stats_list(appid: str, search_word: str, collect_area: Optional[str] = "3",
                   limit: int = 100, max_pages: int = 20) -> List[Dict[str, Any]]:
    """List candidate tables via searchWord; optionally restrict to municipal collectArea=3."""
    start = 1
    out: List[Dict[str, Any]] = []
    for _ in range(max_pages):
        params = {"appId": appid, "searchWord": search_word, "limit": limit, "startPosition": start}
        if collect_area:
            params["collectArea"] = collect_area
        js = _http_get("getStatsList", params)
        lst = js.get("GET_STATS_LIST", {}).get("DATALIST_INF", {}).get("TABLE_INF", [])
        if isinstance(lst, dict):
            lst = [lst]
        if not lst:
            break
        out.extend(lst)
        if len(lst) < limit:
            break
        start += len(lst)
    return out


# -----------------------------
# Catalog: item_id -> statsDataId
# -----------------------------
def _load_catalog() -> Dict[str, str]:
    if CATALOG_FILE.exists():
        try:
            return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_catalog(mp: Dict[str, str]) -> None:
    try:
        CATALOG_FILE.write_text(json.dumps(mp, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def resolve_stats_id_for_item(appid: str, item_id: str) -> str:
    """
    Resolve statsDataId for an item by scanning City '基礎データ' tables and checking cat01 membership.
    Cached on disk for speed.
    """
    catalog = _load_catalog()
    if item_id in catalog:
        return catalog[item_id]

    # 1) Narrow by searchWord to the "市区町村データ 基礎データ" universe
    #    and also try without collectArea restriction as fallback.
    for collect_area in ("3", None):
        tables = get_stats_list(appid, search_word="市区町村データ 基礎データ", collect_area=collect_area)
        for t in tables:
            sid = t.get("@id")
            if not sid:
                continue
            # 2) Check cat01 membership
            meta = get_meta_info(appid, sid)
            try:
                objs = meta["GET_META_INFO"]["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"]
            except KeyError:
                continue
            objs = [objs] if isinstance(objs, dict) else objs
            for obj in objs:
                if obj.get("@id") != "cat01":
                    continue
                cls = obj.get("CLASS")
                cls = [cls] if isinstance(cls, dict) else cls
                codes = {c.get("@code") for c in (cls or []) if c.get("@code")}
                if item_id in codes:
                    catalog[item_id] = sid
                    _save_catalog(catalog)
                    return sid
    raise RuntimeError(f"Unable to resolve statsDataId for item {item_id} via getStatsList+getMetaInfo")


# -----------------------------
# Meta parsing
# -----------------------------
def extract_time_pairs(meta: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Return list of (year:int, time_code:str) sorted asc. Accepts 'YYYY年度' etc."""
    out: List[Tuple[int, str]] = []
    try:
        objs = meta["GET_META_INFO"]["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"]
    except KeyError:
        return out
    objs = [objs] if isinstance(objs, dict) else objs
    for obj in objs:
        if obj.get("@id") != "time":
            continue
        cls = obj.get("CLASS")
        cls = [cls] if isinstance(cls, dict) else cls
        for c in cls or []:
            code = c.get("@code")
            name = (c.get("@name") or "")
            if not code:
                continue
            if len(name) >= 4 and name[:4].isdigit():
                out.append((int(name[:4]), code))
            else:
                sc = str(code)
                if len(sc) >= 4 and sc[:4].isdigit():
                    out.append((int(sc[:4]), code))
    out.sort(key=lambda x: x[0])
    return out


def _extract_values(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    vals = (js.get("GET_STATS_DATA", {})
              .get("STATISTICAL_DATA", {})
              .get("DATA_INF", {})
              .get("VALUE", []))
    if isinstance(vals, dict):
        vals = [vals]
    return vals or []


# -----------------------------
# Value parsing & filters
# -----------------------------
def parse_numeric(raw: Any) -> Optional[float]:
    """Normalize and parse numeric strings; return None for '-', '…', 'X', '' or parse failure."""
    if raw in (None, "", "-", "…", "X"):
        return None
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if s.endswith("%"):
        s = s[:-1]
    s = s.replace(",", "").replace("，", "").replace("−", "-")
    try:
        return float(s)
    except Exception:
        return None


def is_city_or_ward_code(code: Optional[str]) -> bool:
    """Keep only 5-digit area codes that do not end with '000' (exclude prefectures & nation)."""
    if not code or len(code) != 5:
        return False
    return not code.endswith("000")


# -----------------------------
# I/O helpers
# -----------------------------
def _parquet_num_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    if _pq is not None:
        try:
            return _pq.ParquetFile(str(path)).metadata.num_rows  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        df = pd.read_parquet(path, columns=["year"])
        return len(df)
    except Exception:
        return None


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to CSV if parquet engine missing
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[warn] Parquet engine missing; wrote CSV instead: {csv_path}")


# -----------------------------
# Excel item list
# -----------------------------
def extract_items_from_excel(excel_path: Path) -> List[str]:
    xl = pd.ExcelFile(excel_path)
    found: set[str] = set()
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str)
        for col in df.columns:
            for val in df[col].dropna().astype(str):
                val = val.strip()
                if ITEM_RE.match(val):
                    found.add(val)
    items = sorted(found)
    print(f"[info] items detected: {len(items)}")
    if items[:10]:
        print("[info] first items:", items[:10])
    return items


# -----------------------------
# Fetch core
# -----------------------------
def fetch_item_all_years(appid: str, item_id: str, out_dir: Path, save_raw: bool,
                         chunk_size: int = 50) -> int:
    stats_id = resolve_stats_id_for_item(appid, item_id)
    print(f"[info] item={item_id} -> statsDataId={stats_id}")

    meta = get_meta_info(appid, stats_id)
    pairs = extract_time_pairs(meta)
    if not pairs:
        print(f"[warn] no time codes for statsDataId={stats_id}")
        return 0

    raw_dir = out_dir / f"item={item_id}" / "_raw"
    records: List[Dict[str, Any]] = []

    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i+chunk_size]
        cdtime = ",".join(code for (_, code) in chunk)
        year_map = {code: y for (y, code) in chunk}

        start = 1
        limit = 10000
        page = 1
        while True:
            params = {
                "appId": appid,
                "statsDataId": stats_id,
                "cdCat01": item_id,
                "cdTime": cdtime,
                # lvArea intentionally omitted
                "startPosition": start,
                "limit": limit,
            }
            js = _http_get("getStatsData", params)
            if save_raw:
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{item_id}_chunk{i//chunk_size+1}_p{page}.json").write_text(
                    json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            vals = _extract_values(js)
            if not vals:
                break

            for v in vals:
                area = v.get("@area")
                if not is_city_or_ward_code(area):
                    continue
                tcode = v.get("@time")
                y = year_map.get(tcode)
                raw = v.get("$")
                num = parse_numeric(raw)
                rec = {
                    "area_code": area,
                    "year": y,
                    "item_id": item_id,
                    "value": num,
                    "quality_flag": None if num is not None else (raw if raw not in (None, "") else None),
                    "statsDataId": stats_id,
                    "time_code": tcode,
                }
                records.append(rec)

            if len(vals) < limit:
                break
            start += limit
            page += 1

    if not records:
        print(f"[warn] no municipal/ward records found for item={item_id}")
        return 0

    df = pd.DataFrame.from_records(records)
    df.sort_values(["area_code", "year"], inplace=True)
    out_item_dir = out_dir / f"item={item_id}"
    save_parquet(df, out_item_dir / "values.parquet")
    print(f"[ok] saved {len(df)} rows -> {out_item_dir/'values.parquet'}")
    return len(df)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appid", required=True, help="e-Stat API appId")
    ap.add_argument("--excel", required=True, help="Excel file listing items")
    ap.add_argument("--base_dir", required=True, help="Base directory for output")
    ap.add_argument("--chunk-size", type=int, default=50, help="Number of time codes per request")
    ap.add_argument("--save-raw", action="store_true", help="Save raw JSON per request for diagnostics")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip items whose values.parquet already exists and has >0 rows")
    args = ap.parse_args()

    excel_path = Path(args.excel)
    out_dir = Path(args.base_dir) / "muni_by_item"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = extract_items_from_excel(excel_path)

    def existing_rows(path: Path) -> Optional[int]:
        return _parquet_num_rows(path)

    total_rows = 0
    for idx, item in enumerate(items, 1):
        pq_path = out_dir / f"item={item}" / "values.parquet"
        if args.skip_existing:
            nrows = existing_rows(pq_path)
            if nrows and nrows > 0:
                print(f"[skip-existing] {item} -> {pq_path} rows={nrows}")
                continue
        print(f"[{idx}/{len(items)}] fetching item: {item}")
        try:
            total_rows += fetch_item_all_years(args.appid, item, out_dir, args.save_raw, args.chunk_size)
        except Exception as e:
            print(f"[error] item={item} err={e}")
            # continue with next item

    print(f"[summary] total_rows={total_rows}")


if __name__ == "__main__":
    main()
