"""Reproducible stage-one audit, including auxiliary coverage and join safety."""
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from data_cleaning import REPO_ROOT, TARGET, audit_report, clean_data, normalise_block, normalise_text


def run_audit():
    train, test = clean_data()
    data_dir = REPO_ROOT / "data"
    result = {"core": audit_report(train, test), "versions": {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    result["sha256"] = {str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(data_dir.rglob("*.csv")) if "processed" not in p.parts}
    features = [c for c in train if c != TARGET]
    keys = ["TOWN", "BLOCK", "STREET"]
    shared = sorted(set(train.RENT_APPROVAL_DATE) & set(test.RENT_APPROVAL_DATE))
    result["shared_months"] = {m: {"train": int((train.RENT_APPROVAL_DATE == m).sum()),
                                       "test": int((test.RENT_APPROVAL_DATE == m).sum())} for m in shared}
    result["test_rows_with_seen_features"] = int(test.merge(
        train[features].drop_duplicates(), on=features, how="left", indicator=True,
        validate="many_to_one")["_merge"].eq("both").sum())
    result["unseen_categories"] = {c: {"values": sorted(set(test[c]) - set(train[c])),
        "rows": int((~test[c].isin(train[c])).sum())}
        for c in ["TOWN", "STREET", "FLAT_TYPE", "FLAT_MODEL"]}
    result["numeric_ranges"] = {s: df.select_dtypes("number").agg(["min", "median", "max"]).to_dict()
                                for s, df in [("train", train), ("test", test)]}
    sample = pd.read_csv(data_dir / "example-submission.csv")
    assert list(sample) == ["Id", "Predicted"] and len(sample) == len(test)
    assert np.array_equal(sample.Id, np.arange(len(test)))
    result["submission"] = {"rows": len(sample), "ids_equal_test_row_order": True}
    aux = {p.stem: pd.read_csv(p) for p in sorted((data_dir / "auxiliary").glob("*.csv"))}
    result["auxiliary"] = {}
    for name, df in aux.items():
        info = {"rows": len(df), "missing_by_column": df.isna().sum().to_dict(),
                "exact_duplicate_excess": int(df.duplicated().sum())}
        info["placeholder_by_column"] = {
            c: int(df[c].astype("string").str.strip().str.lower().isin(
                ["nil", "n/a", "na", "none", "null", "unknown", "-", ""]).sum())
            for c in df.columns}
        if "LATITUDE" in df:
            info["invalid_coordinates"] = int((~df.LATITUDE.between(-90, 90) |
                                                ~df.LONGITUDE.between(-180, 180)).sum())
            info["coordinate_ranges"] = df[["LATITUDE", "LONGITUDE"]].agg(["min", "max"]).to_dict()
        result["auxiliary"][name] = info
    blocks = aux["sg-hdb-block"]
    for c in keys:
        blocks[c] = normalise_block(blocks[c]) if c == "BLOCK" else normalise_text(blocks[c]).str.lower()
    result["hdb_duplicate_key_excess"] = int(blocks.duplicated(keys).sum())
    result["hdb_invalid_numeric_fields"] = {
        c: int(blocks[c].le(0).sum()) for c in ["MAX_FLOOR", "YEAR_COMPLETED"]}
    # A match-rate alone cannot detect row multiplication.
    for s, df in [("train", train), ("test", test)]:
        joined = df.merge(blocks, on=keys, how="left", validate="many_to_one", indicator=True)
        assert len(joined) == len(df)
        result[f"hdb_{s}_unmatched_rows"] = int(joined._merge.eq("left_only").sum())
        valid_completion = joined.YEAR_COMPLETED.gt(0)
        result[f"consistency_{s}"] = {
            "invalid_block_format": int((~df.BLOCK.str.fullmatch(r"[0-9]+[A-Z]?")).sum()),
            "unexpected_flat_type": int((~df.FLAT_TYPE.isin(
                ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"])).sum()),
            "invalid_completion_or_floor_rows": int((joined.YEAR_COMPLETED.le(0) | joined.MAX_FLOOR.le(0)).sum()),
            "approval_before_known_completion": int((valid_completion & joined.RENT_APPROVAL_YEAR.lt(joined.YEAR_COMPLETED)).sum()),
            "lease_before_known_completion": int((valid_completion & joined.LEASE_COMMENCE_DATE.lt(joined.YEAR_COMPLETED)).sum()),
            "blocks_with_multiple_lease_years": int(df.groupby(keys).LEASE_COMMENCE_DATE.nunique().gt(1).sum()),
        }
        result[f"negative_age_context_{s}"] = joined.loc[joined.FLAT_AGE.lt(0),
            keys + ["RENT_APPROVAL_DATE", "LEASE_COMMENCE_DATE", "YEAR_COMPLETED", "FLAT_AGE"]].to_dict("records")
    result["mrt_status"] = aux["sg-mrt-stations"].STATUS.value_counts().to_dict()
    stock = aux["sg-stock-prices"]
    complete_ohlc = stock[["OPEN", "HIGH", "LOW", "CLOSE"]].notna().all(axis=1)
    result["stock_numeric_checks"] = {
        "nonpositive_close": int(stock.CLOSE.le(0).sum()),
        "negative_volume": int(stock.VOLUME.lt(0).sum()),
        "inconsistent_ohlc_rows": int((complete_ohlc & (
            stock.HIGH.lt(stock[["OPEN", "LOW", "CLOSE"]].max(axis=1)) |
            stock.LOW.gt(stock[["OPEN", "HIGH", "CLOSE"]].min(axis=1)))).sum()),
    }
    stock["month"] = pd.to_datetime(stock.DATE, errors="raise").dt.strftime("%Y-%m")
    coe = aux["sg-coe-prices"]
    coe["month"] = pd.to_datetime(coe.YEAR.astype(str) + "-" + coe.MONTH,
                                  format="%Y-%b", errors="raise").dt.strftime("%Y-%m")
    result["macro_coverage"] = {}
    for name, df, group, key in [("stock", stock, "SYMBOL", ["SYMBOL", "DATE"]),
                                  ("coe", coe, "CATEGORY", ["YEAR", "MONTH", "CATEGORY", "ROUND"])]:
        result["macro_coverage"][name] = {
            "month_span": [df.month.min(), df.month.max()],
            "duplicate_key_excess": int(df.duplicated(key).sum()),
            "missing_test_months": sorted(set(test.RENT_APPROVAL_DATE) - set(df.month)),
            "coverage_by_series": {str(g): {"start": x.month.min(), "end": x.month.max(),
                "missing_test_months": sorted(set(test.RENT_APPROVAL_DATE) - set(x.month))}
                for g, x in df.groupby(group)}}
    out = REPO_ROOT / "outputs" / "audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit saved: {out}")
    return result


if __name__ == "__main__":
    run_audit()
