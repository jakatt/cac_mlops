"""
3-level schema validation for ONISR raw CSV files.

Level 1 — FORMAT     : file readable, correct separator, non-empty
Level 2 — SCHEMA     : required columns present, types compatible
Level 3 — QUALITY    : distributions, NaN rates, value ranges

Result levels : CRITICAL (stop pipeline) / WARNING (log + continue) / OK
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from .schema import QUALITY_BOUNDS, REQUIRED_COLUMNS, TABLE_SCHEMAS
from .import_raw_data import discover_raw_files, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class ValidationMessage:
    level: str          # "CRITICAL" | "WARNING" | "INFO"
    table: str
    check: str
    detail: str


@dataclass
class ValidationReport:
    year: int
    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def overall_level(self) -> str:
        levels = {m.level for m in self.messages}
        if "CRITICAL" in levels:
            return "CRITICAL"
        if "WARNING" in levels:
            return "WARNING"
        return "OK"

    def add(self, level: str, table: str, check: str, detail: str) -> None:
        msg = ValidationMessage(level, table, check, detail)
        self.messages.append(msg)
        log_fn = logger.critical if level == "CRITICAL" else (
            logger.warning if level == "WARNING" else logger.info
        )
        log_fn("[%s] year=%d table=%s check=%s — %s", level, self.year, table, check, detail)

    def summary(self) -> str:
        lines = [f"ValidationReport year={self.year} → {self.overall_level}"]
        for m in self.messages:
            lines.append(f"  [{m.level}] {m.table}.{m.check}: {m.detail}")
        return "\n".join(lines)


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Try common separators/encodings. Return None if unreadable."""
    for sep in (";", ",", "\t"):
        for enc in ("utf-8", "latin-1", "utf-8-sig"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, nrows=5)
                if len(df.columns) > 1:
                    return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except Exception:
                continue
    return None


# ── Level 1 ───────────────────────────────────────────────────────────────────

def _validate_level1(year: int, raw_dir: Path, report: ValidationReport) -> bool:
    """Return True if all 4 files are discoverable, readable and non-empty."""
    try:
        files = discover_raw_files(year, raw_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        report.add("CRITICAL", "all", "file_discovery", str(exc))
        return False

    all_ok = True
    for table, path in files.items():
        if path.stat().st_size == 0:
            report.add("CRITICAL", table, "file_nonempty", f"'{path.name}' is empty")
            all_ok = False
            continue
        df = _read_csv_safe(path)
        if df is None:
            report.add("CRITICAL", table, "file_readable",
                       f"'{path.name}' cannot be parsed (encoding/separator unknown)")
            all_ok = False
            continue
        if len(df) == 0:
            report.add("CRITICAL", table, "file_nonempty", f"'{path.name}' has 0 rows after parse")
            all_ok = False
    return all_ok


# ── Level 2 ───────────────────────────────────────────────────────────────────

def _validate_level2(
    year: int, raw_dir: Path, report: ValidationReport
) -> dict[str, pd.DataFrame]:
    """Return dict of loaded DataFrames. CRITICAL if required columns missing."""
    try:
        files = discover_raw_files(year, raw_dir)
    except (FileNotFoundError, RuntimeError):
        return {}  # already flagged in Level 1
    dfs: dict[str, pd.DataFrame] = {}

    for table, path in files.items():
        df = _read_csv_safe(path)
        if df is None:
            continue  # already flagged in Level 1

        # 2022+ : ONISR renamed Num_Acc → Accident_Id in caracteristiques
        if table == "caracteristiques" and "Accident_Id" in df.columns and "Num_Acc" not in df.columns:
            df = df.rename(columns={"Accident_Id": "Num_Acc"})

        # ── required columns ─────────────────────────────────────────────────
        required = set(REQUIRED_COLUMNS.get(table, []))
        missing = required - set(df.columns)
        if missing:
            report.add(
                "CRITICAL", table, "required_columns",
                f"Missing required columns: {sorted(missing)}"
            )
        else:
            report.add("INFO", table, "required_columns", "all required columns present")

        # ── unknown columns ───────────────────────────────────────────────────
        schema_cols = set(TABLE_SCHEMAS[table].columns.keys())
        unknown = set(df.columns) - schema_cols - required
        if unknown:
            report.add(
                "WARNING", table, "unknown_columns",
                f"New columns not in schema (ignored): {sorted(unknown)}"
            )

        # ── pandera type validation ───────────────────────────────────────────
        try:
            TABLE_SCHEMAS[table].validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            # Collect type errors; column-presence errors already handled above
            # exc.schema_errors is a list of SchemaError objects (pandera >= 0.14)
            type_errors = [
                e for e in exc.schema_errors
                if "column_in_dataframe" not in str(getattr(e, "check", ""))
            ]
            if type_errors:
                detail = "; ".join(
                    str(getattr(e, "failure_cases", e)) for e in type_errors[:5]
                )
                report.add("WARNING", table, "type_check", f"Type mismatches: {detail}")
        except Exception as exc:
            report.add("WARNING", table, "pandera", str(exc))

        dfs[table] = df

    return dfs


# ── Level 3 ───────────────────────────────────────────────────────────────────

def _validate_level3(
    year: int, dfs: dict[str, pd.DataFrame], report: ValidationReport
) -> None:
    """Quality checks on merged/raw data."""
    bounds = QUALITY_BOUNDS

    # Accident count (from caracteristiques table)
    if "caracteristiques" in dfs:
        n = len(dfs["caracteristiques"])
        if n < bounds["accident_count_min"] or n > bounds["accident_count_max"]:
            report.add(
                "WARNING", "caracteristiques", "accident_count",
                f"{n} accidents — outside expected range "
                f"[{bounds['accident_count_min']}, {bounds['accident_count_max']}]"
            )
        else:
            report.add("INFO", "caracteristiques", "accident_count",
                       f"{n} accidents (within expected range)")

    # NaN rate per table
    for table, df in dfs.items():
        for col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate > bounds["nan_rate_warning"]:
                report.add(
                    "WARNING", table, f"nan_rate_{col}",
                    f"{col} has {nan_rate:.1%} NaN (threshold: {bounds['nan_rate_warning']:.0%})"
                )

    # grav values (usagers)
    if "usagers" in dfs:
        invalid_grav = ~dfs["usagers"]["grav"].isin([-1, 1, 2, 3, 4])
        if invalid_grav.any():
            report.add(
                "WARNING", "usagers", "grav_values",
                f"{invalid_grav.sum()} rows with grav outside {{-1,1,2,3,4}}"
            )

    # lat/long range (caracteristiques, if already float)
    if "caracteristiques" in dfs:
        df_c = dfs["caracteristiques"].copy()
        try:
            df_c["lat"] = df_c["lat"].astype(str).str.replace(",", ".").astype(float)
            df_c["long"] = df_c["long"].astype(str).str.replace(",", ".").astype(float)
            out_of_range = (
                (df_c["lat"] < bounds["lat_min"]) | (df_c["lat"] > bounds["lat_max"]) |
                (df_c["long"] < bounds["lon_min"]) | (df_c["long"] > bounds["lon_max"])
            )
            n_bad = out_of_range.sum()
            if n_bad > 0:
                report.add(
                    "WARNING", "caracteristiques", "lat_long_range",
                    f"{n_bad} rows with lat/long outside metropolitan France bounding box"
                )
        except Exception:
            pass  # lat/long format issues already caught in Level 2


# ── Public API ────────────────────────────────────────────────────────────────

def validate(year: int, raw_dir: Path | None = None) -> ValidationReport:
    """
    Run all 3 validation levels for *year*.

    raw_dir defaults to data/raw/{year}/ (or data/production/{year}/ for 2024).
    """
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / str(year)

    report = ValidationReport(year=year)
    logger.info("=== Schema validation year=%d (dir=%s) ===", year, raw_dir)

    # Level 1 — abort if files unreadable
    l1_ok = _validate_level1(year, raw_dir, report)
    if not l1_ok:
        logger.critical("Level 1 FAILED — pipeline must stop for year=%d", year)
        return report

    # Level 2 — load DataFrames
    dfs = _validate_level2(year, raw_dir, report)

    # Level 3 — quality
    _validate_level3(year, dfs, report)

    logger.info("=== Validation complete: %s ===", report.overall_level)
    return report
