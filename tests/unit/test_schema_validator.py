"""Unit tests for 3-level schema validation."""
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data.schema_validator import (
    _validate_level1,
    _validate_level3,
    ValidationReport,
)
from src.data.import_raw_data import FILENAMES


@pytest.fixture
def tmp_2021(tmp_path: Path) -> Path:
    """Create a minimal valid 2021 dataset in tmp_path."""
    caract = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002"],
        "jour": [1, 15], "mois": [3, 7],
        "hrmn": ["1430", "0800"],
        "lum": [1, 3], "dep": ["77", "75"], "com": ["77317", "75056"],
        "agg": [2, 1], "int": [1, 2], "atm": [1, 0], "col": [6, 3],
        "lat": ["48,60", "48,85"], "long": ["2,89", "2,35"],
        "an": [2021, 2021], "adr": ["", ""],
    })
    lieux = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002"],
        "catr": [3, 4], "circ": [2, 1], "surf": [1, 1],
        "situ": [1, 3], "vma": [50, 30],
    })
    usagers = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002"],
        "id_vehicule": ["A", "B"], "num_veh": ["01", "01"],
        "place": [1, 2], "catu": [1, 1], "grav": [2, 3],
        "sexe": [1, 2], "an_nais": [1970.0, 1985.0],
        "secu1": [1.0, 2.0],
    })
    vehicules = pd.DataFrame({
        "Num_Acc": ["202100001", "202100002"],
        "id_vehicule": ["A", "B"], "num_veh": ["01", "01"],
        "catv": [2, 7], "obsm": [1, 0], "motor": [1, 1],
    })

    for table, df in [
        ("caracteristiques", caract),
        ("lieux",            lieux),
        ("usagers",          usagers),
        ("vehicules",        vehicules),
    ]:
        df.to_csv(tmp_path / FILENAMES[2021][table], sep=";", index=False)

    return tmp_path


# ── Level 1 ───────────────────────────────────────────────────────────────────

class TestLevel1:
    def test_all_files_present_returns_true(self, tmp_2021):
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_2021, report)
        assert ok is True
        criticals = [m for m in report.messages if m.level == "CRITICAL"]
        assert len(criticals) == 0

    def test_missing_file_raises_critical(self, tmp_2021):
        (tmp_2021 / FILENAMES[2021]["lieux"]).unlink()
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_2021, report)
        assert ok is False
        critical_checks = [m.check for m in report.messages if m.level == "CRITICAL"]
        assert "file_exists" in critical_checks

    def test_empty_file_raises_critical(self, tmp_2021):
        (tmp_2021 / FILENAMES[2021]["vehicules"]).write_bytes(b"")
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_2021, report)
        assert ok is False

    def test_2024_naming_convention(self, tmp_path):
        """2024 uses Caract_2024.csv (uppercase + underscore) — must be in FILENAMES."""
        assert "Caract_2024.csv" == FILENAMES[2024]["caracteristiques"]
        assert "Lieux_2024.csv"  == FILENAMES[2024]["lieux"]

    def test_2023_naming_convention(self):
        """2023 uses caract-2023.csv (abbreviated)."""
        assert "caract-2023.csv" == FILENAMES[2023]["caracteristiques"]

    def test_2021_typo_in_filename(self):
        """2021 has 'carcteristiques' (missing 'a') — typo from ONISR preserved."""
        assert "carcteristiques-2021.csv" == FILENAMES[2021]["caracteristiques"]
        assert "caracteristiques-2021.csv" not in FILENAMES[2021].values()


# ── Level 3 ───────────────────────────────────────────────────────────────────

class TestLevel3:
    def _make_caract(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Num_Acc": [str(i) for i in range(n)],
            "lat":  ["48,60"] * n,
            "long": ["2,89"] * n,
            "atm":  [1] * n,
        })

    def _make_usagers(self, n: int, grav_values=None) -> pd.DataFrame:
        if grav_values is None:
            grav_values = [2] * n
        return pd.DataFrame({
            "Num_Acc": [str(i) for i in range(n)],
            "grav":    grav_values,
        })

    def test_low_accident_count_triggers_warning(self):
        dfs = {"caracteristiques": self._make_caract(100)}  # way below 40k
        report = ValidationReport(year=2021)
        _validate_level3(2021, dfs, report)
        warns = [m for m in report.messages if m.level == "WARNING" and "accident_count" in m.check]
        assert len(warns) == 1

    def test_normal_count_no_warning(self):
        dfs = {"caracteristiques": self._make_caract(55_000)}
        report = ValidationReport(year=2021)
        _validate_level3(2021, dfs, report)
        warns = [m for m in report.messages if "accident_count" in m.check and m.level == "WARNING"]
        assert len(warns) == 0

    def test_invalid_grav_value_triggers_warning(self):
        dfs = {"usagers": self._make_usagers(10, grav_values=[5, 2, 2, 2, 2, 2, 2, 2, 2, 2])}
        report = ValidationReport(year=2021)
        _validate_level3(2021, dfs, report)
        warns = [m for m in report.messages if "grav_values" in m.check]
        assert len(warns) == 1
