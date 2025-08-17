
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =======================
# Optional dependencies
# =======================
try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:
    _HAS_POLARS = False
    import pandas as pd  # type: ignore
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

import urllib.parse
import urllib.request


# =======================
# Endpoints (official)
# =======================
# e-Stat getMetaInfo (JSON)
ESTAT_META_ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/json/getMetaInfo"
# e-Stat LOD SPARQL
SPARQL_ENDPOINT = "http://data.e-stat.go.jp/lod/sparql/alldata/query"


# =======================
# IO helpers
# =======================
def _read_parquet(path: Path, columns: Optional[List[str]] = None):
    if _HAS_POLARS:
        return pl.read_parquet(str(path), columns=columns)
    else:
        table = pq.read_table(str(path), columns=columns)
        return table.to_pandas()


def _write_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_POLARS:
        df.write_parquet(str(path), compression="snappy")
    else:
        if hasattr(df, "to_arrow"):
            table = df.to_arrow()
        else:
            table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path), compression="snappy")


def _list_item_dirs(base: Path) -> List[Tuple[str, Path]]:
    """Return list of (item_id, values.parquet path)."""
    root = base / "muni_by_item"
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        return out
    for p in sorted(root.glob("item=*/values.parquet")):
        m = re.search(r"item=([^/\\\\]+)", str(p))
        if m:
            out.append((m.group(1), p))
    return out


# =======================
# Progress helper
# =======================
def _progress_update(desc: str, idx: int, total: int):
    if _HAS_TQDM:
        return  # tqdm bar handles progress
    step = max(1, total // 20) if total else 1  # ~5% steps
    if idx == 1 or idx == total or (idx % step == 0):
        pct = int(idx * 100 / total) if total else 100
        print(f"{desc} {idx}/{total} ({pct}%)")


# =======================
# Item helpers
# =======================
def _years_statsdata_from_values(path: Path) -> Tuple[List[int], Optional[str]]:
    """Extract years (unique, sorted) and mode statsDataId from a values.parquet.
    - Years exclude any year in which all rows have null/NaN 'value' (ie. no actual data).
    """
    df = _read_parquet(path)
    cols = df.columns if _HAS_POLARS else df.columns.tolist()
    years: List[int] = []
    stats_mode: Optional[str] = None

    # Years from non-null values
    if "year" in cols:
        if "value" in cols:
            if _HAS_POLARS:
                df_nonnull = df.filter(pl.col("value").is_not_null() & (~pl.col("value").is_nan()))
                years = (
                    df_nonnull.select(pl.col("year").cast(pl.Int32))
                              .drop_nulls()
                              .unique()
                              .sort("year")
                              .to_series()
                              .to_list()
                )
            else:
                years = (
                    df.loc[~df["value"].isna(), "year"]
                      .dropna().astype("int32")
                      .drop_duplicates().sort_values()
                      .tolist()
                )
        else:
            # No value column; keep any year present
            if _HAS_POLARS:
                years = (
                    df.select(pl.col("year").cast(pl.Int32))
                      .drop_nulls()
                      .unique()
                      .sort("year")
                      .to_series()
                      .to_list()
                )
            else:
                years = (
                    df["year"].dropna().astype("int32")
                      .drop_duplicates().sort_values().tolist()
                )

    # statsDataId mode
    if "statsDataId" in cols:
        if _HAS_POLARS:
            ser = df.select(pl.col("statsDataId").cast(pl.Utf8)).to_series().to_list()
        else:
            ser = df["statsDataId"].astype(str).tolist()
        cnt = Counter(x for x in ser if x and x.lower() not in ("none", "nan"))
        stats_mode = cnt.most_common(1)[0][0] if cnt else None

    return years, stats_mode


def _gcd_cycle(years: List[int]) -> str:
    if not years or len(years) < 2:
        return "unknown"
    diffs = [b - a for a, b in zip(years, years[1:]) if b - a > 0]
    if not diffs:
        return "unknown"
    g = diffs[0]
    for d in diffs[1:]:
        g = math.gcd(g, d)
    if g == 1:
        return "annual"
    if g in (2, 3, 4, 5, 10):
        return f"{g}y"
    return "irregular"


def _estat_get_meta(appid: Optional[str], statsDataId: Optional[str], timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Call e-Stat getMetaInfo JSON; return parsed payload or None."""
    if not appid or not statsDataId:
        return None
    try:
        params = {"appId": appid, "statsDataId": statsDataId, "lang": "J"}
        url = f"{ESTAT_META_ENDPOINT}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "catalog_build/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        j = json.loads(data.decode("utf-8"))
        root = j.get("GET_META_INFO") or j.get("GET_METAINFO") or j
        status = None
        try:
            status = root.get("RESULT", {}).get("STATUS")
        except Exception:
            pass
        if status not in (None, "0", 0):
            print(f"[warn] getMetaInfo STATUS={status} statsDataId={statsDataId}")
        return root
    except Exception as e:
        print(f"[warn] getMetaInfo failed statsDataId={statsDataId}: {e}")
        return None


def _extract_name_unit_from_meta(meta: Dict[str, Any], item_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Find CLASS where code==item_id, prefer CLASS_OBJ id='cat01'. Return (name, unit)."""
    try:
        class_inf = meta.get("METADATA_INF", {}).get("CLASS_INF", {})
        class_objs = class_inf.get("CLASS_OBJ", [])
        if isinstance(class_objs, dict):
            class_objs = [class_objs]

        def _iter_classes(obj):
            classes = obj.get("CLASS", [])
            if isinstance(classes, dict):
                classes = [classes]
            for c in classes:
                yield c

        # Prefer cat01
        for obj in class_objs:
            if obj.get("@id") == "cat01":
                for c in _iter_classes(obj):
                    code = c.get("@code") or c.get("code")
                    if code and code.lstrip("#") == item_id:
                        name = c.get("@name") or c.get("name")
                        unit = c.get("@unit") or c.get("unit")
                        return name, unit

        # Fallback search
        for obj in class_objs:
            for c in _iter_classes(obj):
                code = c.get("@code") or c.get("code")
                if code and code.lstrip("#") == item_id:
                    name = c.get("@name") or c.get("name")
                    unit = c.get("@unit") or c.get("unit")
                    return name, unit
    except Exception:
        pass
    return None, None


# =======================
# Scope detection helpers
# =======================
TOKYO_23_RANGE = (13101, 13123)
SEIREI_BASES = [
    1100,  # 01100 Sapporo
    4100,  # 04100 Sendai
    11100, # 11100 Saitama
    12100, # 12100 Chiba
    14100, # 14100 Yokohama
    14130, # 14130 Kawasaki
    14210, # 14210 Sagamihara
    15100, # 15100 Niigata
    22100, # 22100 Shizuoka
    22130, # 22130 Hamamatsu
    23100, # 23100 Nagoya
    26100, # 26100 Kyoto
    27100, # 27100 Osaka
    27140, # 27140 Sakai
    28100, # 28100 Kobe
    33100, # 33100 Okayama
    34100, # 34100 Hiroshima
    40100, # 40100 Kitakyushu
    40130, # 40130 Fukuoka
    43100, # 43100 Kumamoto
]

def _nonnull_area_codes(path: Path) -> List[int]:
    """Return unique area_code list where value is not null/NaN."""
    try:
        df = _read_parquet(path, columns=["area_code", "value"])
    except Exception:
        df = _read_parquet(path, columns=["area_code"])
    if _HAS_POLARS:
        if "value" in df.columns:
            df = df.filter(pl.col("value").is_not_null() & (~pl.col("value").is_nan()))
        if "area_code" not in df.columns:
            return []
        return sorted(set(int(x) for x in df["area_code"].to_list()))
    else:
        if "value" in df.columns:
            df = df.loc[~df["value"].isna()]
        if "area_code" not in df.columns:
            return []
        return sorted(set(int(x) for x in df["area_code"].astype(int).tolist()))

def _detect_scope_flags(area_codes: List[int]) -> Dict[str, Any]:
    has_pref = any((c % 1000) == 0 for c in area_codes)
    has_tokyo = any(TOKYO_23_RANGE[0] <= c <= TOKYO_23_RANGE[1] for c in area_codes)
    def is_seirei_ward(c: int) -> bool:
        for base in SEIREI_BASES:
            if base < c < base + 100 and (c % 100) != 0:
                return True
        return False
    has_seirei = any(is_seirei_ward(c) for c in area_codes)
    has_muni = any((c % 1000) != 0 for c in area_codes)  # includes wards
    if has_tokyo or has_seirei:
        geo_min = "ward"
    elif has_muni:
        geo_min = "municipality"
    elif has_pref:
        geo_min = "prefecture"
    else:
        geo_min = "national"
    return {
        "geo_min_level": geo_min,
        "has_tokyo_23": bool(has_tokyo),
        "has_seirei_wards": bool(has_seirei),
        "has_municipal": bool(has_muni),
        "has_prefecture": bool(has_pref),
    }


# =======================
# Builders
# =======================
def build_items(base: Path, appid: Optional[str] = None, sleep_sec: float = 0.25):
    pairs = _list_item_dirs(base)
    rows: List[Dict[str, Any]] = []
    total = len(pairs)
    pbar = tqdm(total=total, desc="[items]", unit="item") if _HAS_TQDM else None

    for idx, (item_id, vpath) in enumerate(pairs, start=1):
        _progress_update("[items]", idx, total)

        years, stats_mode = _years_statsdata_from_values(vpath)
        cycle = _gcd_cycle(years)

        # Meta lookup
        name = None
        unit = None
        meta = _estat_get_meta(appid, stats_mode)
        if meta:
            nm, un = _extract_name_unit_from_meta(meta, item_id)
            name = nm or name
            unit = un or unit
        elif appid and stats_mode:
            print(f"[info] meta not found for item_id={item_id} statsDataId={stats_mode}")
        if appid:
            time.sleep(max(0.0, sleep_sec))  # polite rate limit

        # Scope flags
        flags = _detect_scope_flags(_nonnull_area_codes(vpath))

        rows.append({
            "item_id": item_id,
            "years": [int(y) for y in years],
            "years_len": int(len(years)),
            "first_year": int(years[0]) if years else None,
            "last_year": int(years[-1]) if years else None,
            "statsDataId": stats_mode,
            "cycle": cycle,
            "name": name,
            "unit": unit,
            "source": "e-Stat",
            **flags,
        })

        if pbar is not None:
            pbar.update(1)

    # to DF
    if _HAS_POLARS:
        df = pl.DataFrame(rows)
        df = df.with_columns(
            pl.col("item_id").cast(pl.Utf8),
            pl.col("years").cast(pl.List(pl.Int16)),
            pl.col("years_len").cast(pl.Int16, strict=False),
            pl.col("first_year").cast(pl.Int16, strict=False),
            pl.col("last_year").cast(pl.Int16, strict=False),
            pl.col("statsDataId").cast(pl.Utf8),
            pl.col("cycle").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
            pl.col("unit").cast(pl.Utf8),
            pl.col("source").cast(pl.Utf8),
            pl.col("geo_min_level").cast(pl.Utf8),
            pl.col("has_tokyo_23").cast(pl.Boolean),
            pl.col("has_seirei_wards").cast(pl.Boolean),
            pl.col("has_municipal").cast(pl.Boolean),
            pl.col("has_prefecture").cast(pl.Boolean),
        ).sort(["item_id"])
    else:
        df = pd.DataFrame(rows).sort_values(["item_id"])

    _write_parquet(df, base / "catalog" / "items.parquet")
    if pbar is not None:
        pbar.close()
    print(f"[items] done: {len(rows)} items")


def _sparql_post(query: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            SPARQL_ENDPOINT,
            data=data,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "catalog_build/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
        return json.loads(payload.decode("utf-8"))
    except Exception as e:
        print(f"[warn] SPARQL failed: {e}")
        return None


def _fetch_municipalities_snapshot() -> Optional[List[Dict[str, Any]]]:
    """Fetch latest municipalities: return list of dict(area_code, area_name, pref_name)."""
    q = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX sacs: <http://data.e-stat.go.jp/lod/terms/sacs#>

SELECT ?code5 ?name_ja ?pref_ja WHERE {
  ?plain a sacs:PlainStandardAreaCode ;
         sacs:latestCode ?std .
  ?std dcterms:identifier ?code5 ;
       rdfs:label ?name_ja ;
       sacs:administrativeClass ?aclass ;
       dcterms:isPartOf ?prefStd .
  FILTER(LANG(?name_ja)='ja')
  VALUES ?aclass { sacs:City sacs:Town sacs:Village sacs:SpecialWard sacs:Ward }
  ?prefStd rdfs:label ?pref_ja .
  FILTER(LANG(?pref_ja)='ja')
}
ORDER BY ?code5
"""
    res = _sparql_post(q)
    print("[debug] SPARQL response:", json.dumps(res)[:1000])  # 先頭1000文字だけ表示
    if not res:
        print("[debug] SPARQL response is None")
        return None
    rows = res.get("results", {}).get("bindings", [])
    print(f"[debug] SPARQL bindings count: {len(rows)}")
    out: List[Dict[str, Any]] = []
    for b in rows:
        code5 = b.get("CODE5", {}).get("value")
        name_ja = b.get("NAME_JA", {}).get("value")
        pref_ja = b.get("PREF_JA", {}).get("value")
        if code5 and name_ja and pref_ja:
            try:
                out.append({"area_code": int(code5), "area_name": name_ja, "pref_name": pref_ja})
            except Exception:
                print(f"[debug] failed to parse code5={code5}")
                continue
        else:
            print(f"[debug] missing field: code5={code5}, name_ja={name_ja}, pref_ja={pref_ja}")
    print(f"[debug] parsed municipalities: {len(out)}")
    return out


def build_areas(base: Path, no_remote: bool = False):
    data = None if no_remote else _fetch_municipalities_snapshot()
    if not data:
        # fallback: discover codes from values files only
        codes = set()
        pairs = _list_item_dirs(base)
        total = len(pairs)
        pbar = tqdm(total=total, desc="[areas-scan]", unit="item") if _HAS_TQDM else None
        for idx, (_, vp) in enumerate(pairs, start=1):
            _progress_update("[areas-scan]", idx, total)
            df = _read_parquet(vp, columns=["area_code"])
            if _HAS_POLARS:
                codes.update(df["area_code"].to_list())
            else:
                codes.update(df["area_code"].astype(int).tolist())
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        rows = [{"area_code": int(c), "area_name": None, "pref_code": int(c)//1000, "pref_name": None, "status": None} for c in sorted(codes)]
        print(f"[areas] discovered {len(rows)} codes (fallback)")
    else:
        def detect_area_type(code: int, name: str) -> int:
            # 0:政令指定都市, 1:市, 2:区, 3:町, 4:村
            # 政令指定都市コードリスト
            seirei_codes = [1100,4100,11100,12100,14100,14130,14210,15100,22100,22130,23100,26100,27100,27140,28100,33100,34100,40100,40130,43100]
            # 区判定
            if name.endswith("区"):
                return 2
            # 政令指定都市
            if any(code // 100 == c // 100 for c in seirei_codes):
                return 0
            # 市
            if name.endswith("市"):
                return 1
            # 町
            if name.endswith("町"):
                return 3
            # 村
            if name.endswith("村"):
                return 4
            return -1

        rows = []
        for rec in data:
            code = int(rec["area_code"])
            area_type = detect_area_type(code, rec["area_name"])
            rows.append({
                "area_code": code,
                "area_name": rec["area_name"],
                "area_type": area_type,
                "pref_code": code // 1000,
                "pref_name": rec["pref_name"],
                "status": None,
            })

    if _HAS_POLARS:
        df = pl.DataFrame(rows).with_columns(
            pl.col("area_code").cast(pl.Int32),
            pl.col("area_name").cast(pl.Utf8),
            pl.col("pref_code").cast(pl.Int16),
            pl.col("pref_name").cast(pl.Utf8),
            pl.col("status").cast(pl.Utf8),
            pl.col("area_type").cast(pl.Int8),
        ).sort(["area_code"])
    else:
        df = pd.DataFrame(rows).sort_values(["area_code"])

    _write_parquet(df, base / "catalog" / "areas.parquet")
    print(f"[areas] written: {len(rows)} rows")


# =======================
# CLI
# =======================
def main():
    ap = argparse.ArgumentParser(description="Build catalog/items.parquet and catalog/areas.parquet with progress and auto-fill from e-Stat.")
    ap.add_argument("--base", default="data", help="Base directory")
    ap.add_argument("--appid", default=None, help="e-Stat API appId for getMetaInfo (optional; needed for name/unit)")
    ap.add_argument("--no-remote", action="store_true", help="Disable remote calls (SPARQL/meta).")
    ap.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between meta requests")
    args = ap.parse_args()

    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)

    print(f"[catalog_build] scanning items under {base} ...")
    appid = None if args.no_remote else args.appid
    build_items(base, appid=appid, sleep_sec=args.sleep)
    print(f"[catalog_build] items.parquet written")

    print(f"[catalog_build] building areas (SPARQL {'disabled' if args.no_remote else 'enabled'}) ...")
    build_areas(base, no_remote=args.no_remote)
    print(f"[catalog_build] areas.parquet written")


if __name__ == "__main__":
    main()
