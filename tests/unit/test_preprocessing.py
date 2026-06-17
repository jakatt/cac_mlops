"""Unit tests for make_dataset preprocessing logic."""
import pandas as pd
import numpy as np
import pytest

from src.data.make_dataset import _engineer


def _minimal_dfs() -> tuple:
    """Return (df_users, df_caract, df_places, df_veh) with 3 accidents."""
    df_users = pd.DataFrame({
        "Num_Acc":     ["202100001", "202100002", "202100003"],
        "id_vehicule": ["A", "B", "C"],
        "num_veh":     ["01", "01", "01"],
        "place":       [1, 2, 1],
        "catu":        [1, 2, 1],
        "grav":        [2, 3, 4],
        "sexe":        [1, 2, 1],
        "an_nais":     [1970.0, 1985.0, 2000.0],
        "secu1":       [1.0, 2.0, 1.0],
        "secu2":       [0.0, 0.0, 0.0],
        "secu3":       [0.0, 0.0, 0.0],
        "trajet":      [1, 2, 1],
        "locp":        [0, 0, 0],
        "actp":        [0, 0, 0],
        "etatp":       [1, 1, 1],
    })
    df_caract = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002", "202100003"],
        "jour":    [1, 15, 7],
        "mois":    [3, 7, 11],
        "hrmn":    ["1430", "0800", "2200"],
        "an":      [2021, 2021, 2021],
        "lum":     [1, 3, 5],
        "dep":     ["77", "75", "2A"],
        "com":     ["77317", "75056", "2A247"],
        "agg":     [2, 1, 2],
        "int":     [1, 2, 1],
        "atm":     [1, 0, 1],
        "col":     [6, 3, 1],
        "lat":     ["48,60", "48,85", "43,30"],
        "long":    ["2,89", "2,35", "5,40"],
        "adr":     ["", "", ""],
    })
    df_places = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002", "202100003"],
        "catr":    [3, 4, 3],
        "voie":    [None, None, None],
        "circ":    [2, 1, 2],
        "nbv":     [2, 2, 2],
        "prof":    [1, 1, 1],
        "plan":    [1, 1, 1],
        "surf":    [1, 1, 1],
        "infra":   [0, 0, 0],
        "situ":    [1, 3, 1],
        "vma":     [50, 30, 70],
        "lartpc":  [None, None, None],
        "larrout": [None, None, None],
        "vosp":    [None, None, None],
        "v1":      [None, None, None],
        "v2":      [None, None, None],
        "pr":      [None, None, None],
        "pr1":     [None, None, None],
    })
    df_veh = pd.DataFrame({
        "Num_Acc":     ["202100001", "202100002", "202100003"],
        "id_vehicule": ["A", "B", "C"],
        "num_veh":     ["01", "01", "01"],
        "senc":        [0, 0, 0],
        "catv":        [2, 7, 1],
        "obs":         [0, 0, 0],
        "obsm":        [1, 0, 1],
        "choc":        [1, 0, 1],
        "manv":        [1, 2, 1],
        "motor":       [1, 1, 1],
        "occutc":      [0, 0, 0],
    })
    return df_users, df_caract, df_places, df_veh


class TestEngineer:
    def test_returns_one_row_per_accident(self):
        df = _engineer(*_minimal_dfs())
        assert len(df) == 3

    def test_grav_is_binary(self):
        df = _engineer(*_minimal_dfs())
        assert set(df["grav"].unique()).issubset({0, 1})

    def test_target_recoding_blessure_grave(self):
        """grav=3 (blessé grave) → 1 after recoding; grav=2 (blessé léger) → 0."""
        df = _engineer(*_minimal_dfs())
        # grav original [2, 3, 4] → recode step 1 [3, 4, 2] → target [0, 1, 1]
        assert sorted(df["grav"].tolist()) == [0, 1, 1]

    def test_hrmn_converted_to_hour(self):
        df = _engineer(*_minimal_dfs())
        assert "hour" in df.columns
        assert "hrmn" not in df.columns
        assert df["hour"].dtype in (int, np.int64, np.int32)

    def test_victim_age_computed(self):
        df = _engineer(*_minimal_dfs())
        assert "victim_age" in df.columns
        assert df["victim_age"].notna().any()

    def test_corse_dep_normalized(self):
        df = _engineer(*_minimal_dfs())
        assert 201 in df["dep"].values  # "2A" → 201

    def test_lat_long_are_float(self):
        df = _engineer(*_minimal_dfs())
        assert df["lat"].dtype == float
        assert df["long"].dtype == float

    def test_28_features_plus_target(self):
        df = _engineer(*_minimal_dfs())
        expected_features = {
            "place", "catu", "sexe", "secu1", "year_acc", "victim_age",
            "catv", "obsm", "motor", "catr", "circ", "surf", "situ", "vma",
            "jour", "mois", "lum", "dep", "com", "agg_", "int", "atm", "col",
            "lat", "long", "hour", "nb_victim", "nb_vehicules", "grav",
        }
        assert expected_features.issubset(set(df.columns)), (
            f"Missing columns: {expected_features - set(df.columns)}"
        )

    def test_no_id_columns_in_output(self):
        df = _engineer(*_minimal_dfs())
        forbidden = {"Num_Acc", "id_vehicule", "num_veh", "an_nais", "hrmn", "an"}
        overlap = forbidden & set(df.columns)
        assert not overlap, f"ID/raw columns leaked: {overlap}"
