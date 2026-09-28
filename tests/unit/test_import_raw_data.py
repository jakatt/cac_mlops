"""Unit tests for discover_raw_files (matching strict + fallback fuzzy)."""
from pathlib import Path

import pytest

from src.data.import_raw_data import discover_raw_files, _strip_year_suffix


def _touch(d: Path, name: str) -> None:
    (d / name).write_text("Num_Acc;col\n1;2\n")


class TestStripYearSuffix:
    def test_strips_trailing_year(self):
        assert _strip_year_suffix("carcteristiques-2021") == "carcteristiques"

    def test_strips_underscore_year(self):
        assert _strip_year_suffix("Caract_2024") == "caract"

    def test_no_year_suffix_unchanged(self):
        assert _strip_year_suffix("lieux") == "lieux"


class TestDiscoverRawFiles:
    def test_strict_match_canonical_names(self, tmp_path):
        for name in ["caracteristiques-2021.csv", "lieux-2021.csv",
                     "usagers-2021.csv", "vehicules-2021.csv"]:
            _touch(tmp_path, name)

        result = discover_raw_files(2021, tmp_path)
        assert set(result.keys()) == {"caracteristiques", "lieux", "usagers", "vehicules"}
        assert result["caracteristiques"].name == "caracteristiques-2021.csv"

    def test_fuzzy_fallback_catches_typo(self, tmp_path, caplog):
        """Reproduit l'incident carcteristiques-2021 (faute de frappe source ONISR)."""
        _touch(tmp_path, "carcteristiques-2021.csv")  # typo — ne matche pas "caract"
        _touch(tmp_path, "lieux-2021.csv")
        _touch(tmp_path, "usagers-2021.csv")
        _touch(tmp_path, "vehicules-2021.csv")

        result = discover_raw_files(2021, tmp_path)
        assert result["caracteristiques"].name == "carcteristiques-2021.csv"
        assert any("filename_fuzzy_match" in r.message for r in caplog.records)

    def test_unmatchable_file_raises_runtime_error(self, tmp_path):
        _touch(tmp_path, "xyz-completely-unrelated-2021.csv")
        _touch(tmp_path, "lieux-2021.csv")
        _touch(tmp_path, "usagers-2021.csv")
        _touch(tmp_path, "vehicules-2021.csv")

        with pytest.raises(RuntimeError, match="Cannot identify all 4"):
            discover_raw_files(2021, tmp_path)

    def test_nonexistent_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_raw_files(2021, tmp_path / "nope")


# ── _validate_onisr_csv : contrôle d'entrée tolérant (auto-correction) ────────
from src.data.import_raw_data import _validate_onisr_csv  # noqa: E402

_BODY = ("x" * 80 + "\n") * 8000   # > 500 Ko (seuil anti page d'erreur)


def _csv(tmp_path: Path, header: str) -> Path:
    p = tmp_path / "caracteristiques-2099.csv"
    p.write_text(header + "\n" + _BODY)
    return p


class TestValidateOnisrCsv:
    def test_accident_id_2022_accepted_via_known_rename(self, tmp_path):
        # ONISR 2022 : Num_Acc renommé Accident_Id — renommage connu (known_fixes)
        _validate_onisr_csv(_csv(tmp_path, '"Accident_Id";"jour";"mois";"an";"lum"'), "caracteristiques")

    @pytest.mark.parametrize("sep", ["\t", ","])
    def test_other_separator_accepted(self, tmp_path, sep):
        _validate_onisr_csv(_csv(tmp_path, sep.join(["Num_Acc", "jour", "mois", "an", "lum"])), "caracteristiques")

    def test_unknown_separator_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="séparateur inconnu"):
            _validate_onisr_csv(_csv(tmp_path, "Num_Acc jour mois an lum"), "caracteristiques")

    def test_missing_required_column_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="colonnes manquantes"):
            _validate_onisr_csv(_csv(tmp_path, '"Num_Acc";"jour";"mois";"an"'), "caracteristiques")

    def test_too_small_file_rejected(self, tmp_path):
        p = tmp_path / "caracteristiques-2099.csv"
        p.write_text('"Num_Acc";"jour";"mois";"an";"lum"\n1;2;3;4;5\n')
        with pytest.raises(ValueError, match="trop petit"):
            _validate_onisr_csv(p, "caracteristiques")
