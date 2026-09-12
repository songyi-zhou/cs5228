"""Data cleaning for the CS5228 HDB rental prediction project.

Audits and cleans the raw Kaggle files (``data/train.csv`` / ``data/test.csv``)
and writes ``data/processed/processed_train.csv`` / ``processed_test.csv``.

Cleaning rules:
1. Text normalisation: trim, collapse internal whitespace; TOWN/STREET/FLAT_MODEL
   lowercased; FLAT_TYPE mapped to canonical hyphenated labels; BLOCK letter
   suffix uppercased (e.g. "604a" -> "604A").
2. Constant columns are dropped. In the raw data FURNISHED is always "yes" and
   FEE is always 0 in BOTH train and test, so they carry no signal.
3. RENT_APPROVAL_DATE is validated against the YYYY-MM format and split into
   RENT_APPROVAL_YEAR / RENT_APPROVAL_MONTH (plus a datetime column for plots).
4. Derived column FLAT_AGE = approval year - LEASE_COMMENCE_DATE (one row has
   -1, kept as-is and documented).
5. Keep duplicates by default: these records identify blocks, not individual
   units or contracts. Equal rows are not sufficient evidence of data errors.
   Optional train deduplication is available for a validation ablation only.

Usage (CLI):
    python src/data_cleaning.py
Usage (import):
    from src.data_cleaning import clean_data
    train, test = clean_data()          # also writes the processed CSVs
    train, test = clean_data(write=False)  # in-memory only
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAIN = REPO_ROOT / "data" / "train.csv"
RAW_TEST = REPO_ROOT / "data" / "test.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TARGET = "MONTHLY_RENT"

DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# "4 room" / "4-room" / (any case) -> canonical lowercase hyphenated label.
FLAT_TYPE_PATTERN = re.compile(r"^(\d)\s*-?\s*room$", re.IGNORECASE)

# Canonical column order of the processed outputs.
PROCESSED_COLUMNS = [
    "RENT_APPROVAL_DATE",
    "RENT_APPROVAL_YEAR",
    "RENT_APPROVAL_MONTH",
    "RENT_APPROVAL_DT",
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "FLOOR_AREA_SQM",
    "LEASE_COMMENCE_DATE",
    "FLAT_AGE",
]


# --------------------------------------------------------------------------- #
# Field-level normalisers
# --------------------------------------------------------------------------- #
def normalise_text(series: pd.Series) -> pd.Series:
    """Trim and collapse internal whitespace without changing case."""
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def normalise_flat_type(series: pd.Series) -> pd.Series:
    """Map raw flat-type variants onto canonical labels.

    Raw variants observed: "4 room" vs "4-room" (and case differences in
    general). "executive" passes through unchanged.
    """

    return normalise_text(series).str.lower().str.replace(
        FLAT_TYPE_PATTERN, r"\1-room", regex=True
    )


def normalise_block(series: pd.Series) -> pd.Series:
    """Uppercase the letter suffix of block numbers ("604a" -> "604A")."""
    return (
        normalise_text(series)
        .str.replace(r"([a-z])$", lambda m: m.group(1).upper(), regex=True)
    )


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate RENT_APPROVAL_DATE and derive year/month/datetime columns."""
    bad = ~df["RENT_APPROVAL_DATE"].astype(str).map(lambda v: bool(DATE_PATTERN.match(v)))
    if bad.any():
        raise ValueError(f"Unexpected RENT_APPROVAL_DATE values: {df.loc[bad, 'RENT_APPROVAL_DATE'].unique()[:5]}")

    date = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="%Y-%m")
    df["RENT_APPROVAL_DT"] = date
    df["RENT_APPROVAL_YEAR"] = date.dt.year.astype(int)
    df["RENT_APPROVAL_MONTH"] = date.dt.month.astype(int)
    return df


# --------------------------------------------------------------------------- #
# Audit helpers
# --------------------------------------------------------------------------- #
def audit_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Collect data-quality facts shown to the user and reused in the docs."""
    feature_cols = [c for c in train.columns if c != TARGET and c in test.columns]
    dup_train_full = int(train.duplicated().sum())
    dup_train_features = int(train.duplicated(subset=feature_cols).sum())
    conflicting = 0
    grouped = train.groupby(feature_cols, dropna=False, observed=True)[TARGET]
    conflicting = int((grouped.nunique() > 1).sum())

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "missing_train": int(train.isna().sum().sum()),
        "missing_test": int(test.isna().sum().sum()),
        "exact_duplicates_train": dup_train_full,
        "feature_duplicates_train": dup_train_features,
        "feature_groups_with_conflicting_rent": conflicting,
        "exact_duplicates_test": int(test.duplicated().sum()),
        "rent_min": int(train[TARGET].min()) if TARGET in train else None,
        "rent_max": int(train[TARGET].max()) if TARGET in train else None,
        "rent_median": float(train[TARGET].median()) if TARGET in train else None,
        "rent_under_500": int((train[TARGET] < 500).sum()) if TARGET in train else None,
        "rent_over_6000": int((train[TARGET] > 6000).sum()) if TARGET in train else None,
        "negative_flat_age": int((train["FLAT_AGE"] < 0).sum()) if "FLAT_AGE" in train else None,
        "train_date_span": (train["RENT_APPROVAL_DATE"].min(), train["RENT_APPROVAL_DATE"].max()),
        "test_date_span": (test["RENT_APPROVAL_DATE"].min(), test["RENT_APPROVAL_DATE"].max()),
        "flat_types": sorted(train["FLAT_TYPE"].unique()),
    }
    return report


def print_report(report: dict, dropped_constants: list[str], dropped_rows: int) -> None:
    print("=" * 64)
    print("Data audit & cleaning report")
    print("=" * 64)
    print(f"rows: train={report['train_rows']:,}  test={report['test_rows']:,}")
    print(f"missing values: train={report['missing_train']}  test={report['missing_test']}")
    print(f"constant columns dropped: {dropped_constants or 'none'}")
    print(f"exact duplicate excess rows in train: {report['exact_duplicates_train']:,}; dropped: {dropped_rows:,}")
    print(f"duplicate feature excess rows in train: {report['feature_duplicates_train']:,}; "
          f"feature groups with conflicting rent: {report['feature_groups_with_conflicting_rent']:,}")
    print(f"exact duplicate rows in test (kept, one prediction per Id): {report['exact_duplicates_test']:,}")
    print(f"MONTHLY_RENT: min={report['rent_min']} median={report['rent_median']:.0f} max={report['rent_max']} "
          f"| <500 SGD: {report['rent_under_500']} | >6000 SGD: {report['rent_over_6000']}")
    print(f"FLAT_AGE < 0 (kept, documented): {report['negative_flat_age']}")
    print(f"train dates: {report['train_date_span'][0]} .. {report['train_date_span'][1]}")
    print(f"test  dates: {report['test_date_span'][0]} .. {report['test_date_span'][1]}")
    print(f"flat types after normalisation ({len(report['flat_types'])}): {', '.join(report['flat_types'])}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def clean_data(
    train_path: str | Path = RAW_TRAIN,
    test_path: str | Path = RAW_TEST,
    output_dir: str | Path | None = PROCESSED_DIR,
    write: bool = True,
    drop_exact_duplicates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, audit, and clean the raw Kaggle data.

    Parameters
    ----------
    train_path / test_path : raw CSV locations.
    output_dir : directory for the processed CSVs (only used when ``write``).
    write : write processed_train.csv / processed_test.csv to ``output_dir``.
    drop_exact_duplicates : remove train rows identical across ALL columns
        (features + target). Duplicates are never dropped from test because
        every row needs one prediction.

    Returns
    -------
    (train, test) : cleaned DataFrames (test columns == train minus target).
    """
    train = pd.read_csv(train_path, dtype={"BLOCK": "string"})
    test = pd.read_csv(test_path, dtype={"BLOCK": "string"})

    # --- schema sanity ----------------------------------------------------- #
    if TARGET not in train.columns:
        raise ValueError(f"train is missing the target column {TARGET!r}")
    if TARGET in test.columns:
        raise ValueError(f"test unexpectedly contains the target column {TARGET!r}")
    feature_cols = [c for c in train.columns if c != TARGET]
    if feature_cols != list(test.columns):
        raise ValueError("train/test feature columns differ or are reordered")
    required = {"RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE",
                "FLAT_MODEL", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FURNISHED", "FEE"}
    if set(feature_cols) != required:
        raise ValueError(f"Unexpected input schema: {feature_cols}")
    for name, df in (("train", train), ("test", test)):
        if df.empty:
            raise ValueError(f"{name} is empty")
        df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
        if df.isna().any().any():
            raise ValueError(f"{name} has missing/blank values; define fold-fitted handling before modelling")
        for col in ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE"] + ([TARGET] if name == "train" else []):
            df[col] = pd.to_numeric(df[col], errors="raise")
            if df[col].isin([float("inf"), -float("inf")]).any():
                raise ValueError(f"{name}.{col} contains non-finite values")
        if (df["FLOOR_AREA_SQM"] <= 0).any() or (name == "train" and (df[TARGET] <= 0).any()):
            raise ValueError(f"{name} contains non-positive area/rent")
        if (df["LEASE_COMMENCE_DATE"] % 1 != 0).any():
            raise ValueError(f"{name} contains fractional lease years")

    # --- drop columns that are constant in BOTH train and test ------------- #
    # Only these known nuisance columns may be removed. Fail on changed data
    # rather than silently discard an informative field or a required key.
    constants = [c for c in ["FURNISHED", "FEE"]
                 if pd.concat([train[c], test[c]], ignore_index=True).nunique(dropna=False) == 1]
    if len(constants) != 2:
        raise ValueError("FURNISHED/FEE are no longer jointly constant; revise the output schema")
    if constants:
        train = train.drop(columns=constants)
        test = test.drop(columns=constants)

    # --- text normalisation ------------------------------------------------ #
    for df in (train, test):
        df["TOWN"] = normalise_text(df["TOWN"]).str.lower()
        df["STREET"] = normalise_text(df["STREET"]).str.lower()
        df["FLAT_MODEL"] = normalise_text(df["FLAT_MODEL"]).str.lower()
        df["FLAT_TYPE"] = normalise_flat_type(df["FLAT_TYPE"])
        df["BLOCK"] = normalise_block(df["BLOCK"])
        add_temporal_columns(df)
        df["FLAT_AGE"] = df["RENT_APPROVAL_YEAR"] - df["LEASE_COMMENCE_DATE"].astype(int)

    # --- duplicates -------------------------------------------------------- #
    # NOTE: normalisation can merge rows ("4 room" vs "4-room"), so the count
    # of exact duplicates must be measured AFTER normalisation, BEFORE removal.
    report = audit_report(train, test)
    n_exact_dup_train = report["exact_duplicates_train"]
    if drop_exact_duplicates:
        train = train.drop_duplicates().reset_index(drop=True)
    test = test.reset_index(drop=True)

    # --- final ordering / dtypes ------------------------------------------- #
    display_cols = PROCESSED_COLUMNS
    train = train[display_cols + [TARGET]]
    test = test[display_cols]
    train["FLOOR_AREA_SQM"] = train["FLOOR_AREA_SQM"].astype(float)
    test["FLOOR_AREA_SQM"] = test["FLOOR_AREA_SQM"].astype(float)
    # Preserve fractional rents if a later dataset supplies them.
    print_report(report, constants, n_exact_dup_train if drop_exact_duplicates else 0)

    if write:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / "processed_train.csv", index=False)
        test.to_csv(out / "processed_test.csv", index=False)
        print(f"\nwrote: {out / 'processed_train.csv'} "
              f"({len(train):,} rows x {train.shape[1]} cols)")
        print(f"wrote: {out / 'processed_test.csv'} "
              f"({len(test):,} rows x {test.shape[1]} cols)")

    return train, test


if __name__ == "__main__":
    clean_data()
