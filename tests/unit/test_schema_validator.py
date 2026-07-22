"""Unit tests for 3-level schema validation."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema_validator import (
    _validate_level1,
    _validate_level2,
    _validate_level3,
    ValidationReport,
    load_and_validate_year,
)

# Filenames that match CATEGORY_KEYWORDS for year 2021 (used in fixtures)
_TEST_FILES = {
    "caracteristiques": "caract-2021.csv",
    "lieux":            "lieux-2021.csv",
    "usagers":          "usagers-2021.csv",
    "vehicules":        "vehicules-2021.csv",
}


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

    for table, filename in _TEST_FILES.items():
        df = {"caracteristiques": caract, "lieux": lieux,
              "usagers": usagers, "vehicules": vehicules}[table]
        df.to_csv(tmp_path / filename, sep=";", index=False)

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
        (tmp_2021 / _TEST_FILES["lieux"]).unlink()
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_2021, report)
        assert ok is False
        assert any(m.level == "CRITICAL" for m in report.messages)

    def test_empty_file_raises_critical(self, tmp_2021):
        (tmp_2021 / _TEST_FILES["vehicules"]).write_bytes(b"")
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_2021, report)
        assert ok is False

    def test_nonexistent_dir_raises_critical(self, tmp_path):
        report = ValidationReport(year=2021)
        ok = _validate_level1(2021, tmp_path / "nonexistent", report)
        assert ok is False
        assert any(m.level == "CRITICAL" for m in report.messages)


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


# ── ValidationReport ──────────────────────────────────────────────────────────

class TestValidationReport:
    def test_overall_level_ok_when_no_messages(self):
        report = ValidationReport(year=2021)
        assert report.overall_level == "OK"

    def test_overall_level_critical_after_critical_message(self, tmp_2021):
        report = ValidationReport(year=2021)
        _validate_level1(2021, tmp_2021 / "nonexistent", report)
        assert report.overall_level == "CRITICAL"

    def test_summary_contains_year(self):
        report = ValidationReport(year=2022)
        assert "2022" in report.summary()


# ── Level 2 ───────────────────────────────────────────────────────────────────

class TestLevel2:
    def test_missing_required_column_raises_critical(self, tmp_2021):
        """Supprimer une colonne requise de caracteristiques → CRITICAL level 2."""
        caract_path = tmp_2021 / _TEST_FILES["caracteristiques"]
        df = pd.read_csv(caract_path, sep=";")
        df = df.drop(columns=["dep"])
        df.to_csv(caract_path, sep=";", index=False)

        report = ValidationReport(year=2021)
        _validate_level1(2021, tmp_2021, report)
        _validate_level2(2021, tmp_2021, report)

        criticals = [m for m in report.messages if m.level == "CRITICAL"]
        assert len(criticals) >= 1


# ── Auto-correction (renommage + coercion Pandera) ─────────────────────────────

class TestAutoCorrected:
    def test_rename_reported_as_auto_corrected(self, tmp_2021):
        caract_path = tmp_2021 / _TEST_FILES["caracteristiques"]
        df = pd.read_csv(caract_path, sep=";").rename(columns={"Num_Acc": "Accident_Id"})
        df.to_csv(caract_path, sep=";", index=False)

        report = ValidationReport(year=2021)
        _validate_level1(2021, tmp_2021, report)
        _validate_level2(2021, tmp_2021, report)

        auto = [m for m in report.messages
                if m.level == "AUTO_CORRECTED" and m.check == "column_rename"]
        assert len(auto) == 1
        assert report.overall_level in ("OK", "WARNING")  # AUTO_CORRECTED n'escalade pas

    def test_coercible_value_reported_as_auto_corrected(self, tmp_2021):
        """Colonne numérique lue en object (artefact \\xa0) → coercée + reportée.

        Note : Num_Acc est systématiquement coercé (int64 lu depuis un CSV
        purement numérique → object attendu par le schéma) sur les 4 tables,
        même sans cet artefact — c'est le comportement voulu. On restreint
        donc l'assertion à la table 'caracteristiques' et à la colonne 'jour'.
        """
        caract_path = tmp_2021 / _TEST_FILES["caracteristiques"]
        df = pd.read_csv(caract_path, sep=";")
        df["jour"] = df["jour"].astype(str) + "\xa0"
        df.to_csv(caract_path, sep=";", index=False)

        report = ValidationReport(year=2021)
        _validate_level1(2021, tmp_2021, report)
        dfs = _validate_level2(2021, tmp_2021, report)

        caract_auto = [m for m in report.messages
                       if m.level == "AUTO_CORRECTED" and m.check == "type_coercion"
                       and m.table == "caracteristiques"]
        assert len(caract_auto) == 1
        assert "jour" in caract_auto[0].detail
        assert dfs["caracteristiques"]["jour"].dtype.kind == "i"  # coercé en int

    def test_uncoercible_value_stays_warning_not_critical(self, tmp_2021):
        caract_path = tmp_2021 / _TEST_FILES["caracteristiques"]
        df = pd.read_csv(caract_path, sep=";")
        df["jour"] = df["jour"].astype(object)
        df.loc[0, "jour"] = "abc"
        df.to_csv(caract_path, sep=";", index=False)

        report = ValidationReport(year=2021)
        _validate_level1(2021, tmp_2021, report)
        _validate_level2(2021, tmp_2021, report)

        warns = [m for m in report.messages if m.level == "WARNING" and m.check == "type_check"]
        assert len(warns) == 1
        assert not any(m.level == "CRITICAL" for m in report.messages)


# ── load_and_validate_year (pipeline unique validation + preprocessing) ────────

class TestLoadAndValidateYear:
    def test_returns_dfs_for_all_4_tables(self, tmp_2021):
        dfs, report = load_and_validate_year(2021, tmp_2021)
        assert set(dfs.keys()) == {"caracteristiques", "lieux", "usagers", "vehicules"}
        assert report.overall_level in ("OK", "WARNING")

    def test_critical_returns_empty_dfs(self, tmp_path):
        dfs, report = load_and_validate_year(2021, tmp_path / "nonexistent")
        assert dfs == {}
        assert report.overall_level == "CRITICAL"
