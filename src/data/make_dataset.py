"""
Preprocess raw ONISR CSV files for a given year (or cumulative years).

Usage:
    python make_dataset.py --year 2021
    python make_dataset.py --year 2023 --cumul   # cumulates 2021+2022+2023

Output: data/preprocessed/{output_dir}/X_train.csv X_test.csv y_train.csv y_test.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .import_raw_data import FILENAMES, PROJECT_ROOT, TRAINING_YEARS

logger = logging.getLogger(__name__)


def _raw_dir(year: int) -> Path:
    return PROJECT_ROOT / "data" / "raw" / str(year)


def _load_year(year: int, raw_dir: Path | None = None) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load the 4 raw CSVs for *year*. Returns (df_users, df_caract, df_places, df_veh)."""
    d = raw_dir or _raw_dir(year)
    fn = FILENAMES[year]

    df_caract = pd.read_csv(d / fn["caracteristiques"], sep=";", low_memory=False)
    df_places  = pd.read_csv(d / fn["lieux"],            sep=";", encoding="utf-8")
    df_users   = pd.read_csv(d / fn["usagers"],          sep=";")
    df_veh     = pd.read_csv(d / fn["vehicules"],        sep=";")

    return df_users, df_caract, df_places, df_veh


def _engineer(
    df_users: pd.DataFrame,
    df_caract: pd.DataFrame,
    df_places: pd.DataFrame,
    df_veh: pd.DataFrame,
) -> pd.DataFrame:
    """All feature engineering and merging for one year's DataFrames."""

    # ── derived columns ────────────────────────────────────────────────────────
    nb_victim    = pd.crosstab(df_users.Num_Acc, "count").reset_index()
    nb_vehicules = pd.crosstab(df_veh.Num_Acc,   "count").reset_index()

    df_users["year_acc"]   = df_users["Num_Acc"].astype(str).str[:4].astype(int)
    df_users["victim_age"] = df_users["year_acc"] - df_users["an_nais"]
    df_users.loc[
        (df_users["victim_age"] > 120) | (df_users["victim_age"] < 0), "victim_age"
    ] = np.nan

    df_caract["hour"] = df_caract["hrmn"].astype(str).str[:-3]
    df_caract.drop(columns=["hrmn", "an"], inplace=True, errors="ignore")
    df_users.drop(columns=["an_nais"], inplace=True, errors="ignore")

    # ── recodings ─────────────────────────────────────────────────────────────
    df_users["grav"] = df_users["grav"].replace([1, 2, 3, 4], [1, 3, 4, 2])
    df_caract.rename(columns={"agg": "agg_"}, inplace=True)

    df_caract["dep"] = df_caract["dep"].astype(str).str.replace("2A", "201").str.replace("2B", "202")
    df_caract["com"] = df_caract["com"].astype(str).str.replace("2A", "201").str.replace("2B", "202")
    df_caract[["dep", "com", "hour"]] = df_caract[["dep", "com", "hour"]].astype(int)

    df_caract["lat"]  = df_caract["lat"].astype(str).str.replace(",", ".").astype(float)
    df_caract["long"] = df_caract["long"].astype(str).str.replace(",", ".").astype(float)

    dico_atm = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
    df_caract["atm"] = df_caract["atm"].replace(dico_atm)

    catv_old = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99]
    catv_new = [0,1,1,2,1,1,6,2,5,5,5,5,5,4,4,4,4,4,3,3,4,4,1,1,1,1,1,6,6,3,3,3,3,1,1,1,1,1,0,0]
    df_veh["catv"] = df_veh["catv"].replace(catv_old, catv_new)

    # ── merges ─────────────────────────────────────────────────────────────────
    fusion = (
        df_users
        .merge(df_veh,    on=["Num_Acc", "num_veh", "id_vehicule"], how="inner")
        .sort_values("grav", ascending=False)
        .drop_duplicates(subset=["Num_Acc"], keep="first")
        .merge(df_places, on="Num_Acc", how="left")
        .merge(df_caract, on="Num_Acc", how="left")
        .merge(nb_victim,    on="Num_Acc", how="inner")
        .rename(columns={"count": "nb_victim"})
        .merge(nb_vehicules, on="Num_Acc", how="inner")
        .rename(columns={"count": "nb_vehicules"})
    )

    # ── target variable ────────────────────────────────────────────────────────
    fusion["grav"] = fusion["grav"].replace([2, 3, 4], [0, 1, 1])

    # ── NaN handling ──────────────────────────────────────────────────────────
    cols_minus1_to_nan = ["trajet", "secu1", "catv", "obsm", "motor", "circ", "surf", "situ", "vma", "atm", "col"]
    cols_zero_to_nan   = ["trajet", "catv", "motor"]
    fusion[cols_minus1_to_nan] = fusion[cols_minus1_to_nan].replace(-1, np.nan)
    fusion[cols_zero_to_nan]   = fusion[cols_zero_to_nan].replace(0, np.nan)

    # ── drop columns ──────────────────────────────────────────────────────────
    drop_cols = [
        "senc", "larrout", "actp", "manv", "choc", "nbv", "prof", "plan",
        "Num_Acc", "id_vehicule", "num_veh", "pr", "pr1", "voie", "trajet",
        "secu2", "secu3", "adr", "v1", "lartpc", "occutc", "v2", "vosp",
        "locp", "etatp", "infra", "obs",
    ]
    fusion.drop(columns=[c for c in drop_cols if c in fusion.columns], inplace=True)

    # ── drop rows with critical NaN ───────────────────────────────────────────
    critical_cols = ["catv", "vma", "secu1", "obsm", "atm"]
    existing_critical = [c for c in critical_cols if c in fusion.columns]
    fusion.dropna(subset=existing_critical, inplace=True)

    return fusion


def process_years(
    years: list[int],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge and preprocess all *years*, return (X_train, X_test, y_train, y_test).
    Saves CSVs to output_dir (creates it if necessary).
    """
    frames: list[pd.DataFrame] = []
    for year in sorted(years):
        logger.info("Loading year %d…", year)
        dfs = _load_year(year)
        df = _engineer(*dfs)
        logger.info("  year=%d → %d rows × %d cols", year, len(df), df.shape[1])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    target = combined["grav"]
    feats  = combined.drop(columns=["grav"])

    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.3, random_state=42
    )

    # ── impute remaining NaN (on train stats) ────────────────────────────────
    impute_cols = ["surf", "circ", "col", "motor"]
    existing_impute = [c for c in impute_cols if c in X_train.columns]
    mode_vals = X_train[existing_impute].mode().iloc[0]
    X_train[existing_impute] = X_train[existing_impute].fillna(mode_vals)
    X_test[existing_impute]  = X_test[existing_impute].fillna(mode_vals)

    logger.info(
        "Dataset ready: %d train / %d test — %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )

    # ── save ─────────────────────────────────────────────────────────────────
    if output_dir is None:
        label = "_".join(str(y) for y in sorted(years))
        if len(years) == 1:
            output_dir = PROJECT_ROOT / "data" / "preprocessed" / label
        else:
            output_dir = PROJECT_ROOT / "data" / "preprocessed" / f"cumul_{label}"

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir  / "X_test.csv",  index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir  / "y_test.csv",  index=False)
    logger.info("Saved preprocessed data to %s", output_dir)

    return X_train, X_test, y_train, y_test


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Preprocess ONISR data")
    parser.add_argument("--year", type=int, required=True,
                        help="Most recent year to include")
    parser.add_argument("--cumul", action="store_true",
                        help="Cumulate all training years up to --year")
    args = parser.parse_args()

    if args.cumul:
        years = [y for y in TRAINING_YEARS if y <= args.year]
    else:
        years = [args.year]

    process_years(years)


if __name__ == "__main__":
    main()
