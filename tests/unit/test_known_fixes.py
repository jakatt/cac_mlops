"""Unit tests for the centralized known-fixes registry."""
import pandas as pd

from src.data.known_fixes import apply_known_fixes


class TestApplyKnownFixes:
    def test_renames_accident_id_to_num_acc(self):
        df = pd.DataFrame({"Accident_Id": ["1", "2"], "jour": [1, 2]})
        out, renamed = apply_known_fixes(df, "caracteristiques")
        assert "Num_Acc" in out.columns
        assert "Accident_Id" not in out.columns
        assert renamed == ["Accident_Id"]

    def test_no_rename_when_num_acc_already_present(self):
        """Ne pas écraser Num_Acc si les deux colonnes existent (cas improbable
        mais évite une perte de données silencieuse)."""
        df = pd.DataFrame({"Accident_Id": ["1"], "Num_Acc": ["999"]})
        out, renamed = apply_known_fixes(df, "caracteristiques")
        assert renamed == []
        assert out["Num_Acc"].tolist() == ["999"]

    def test_no_rename_for_other_tables(self):
        df = pd.DataFrame({"Num_Acc": ["1"], "catv": [2]})
        out, renamed = apply_known_fixes(df, "vehicules")
        assert renamed == []
        assert list(out.columns) == ["Num_Acc", "catv"]

    def test_strips_nbsp_and_whitespace(self):
        df = pd.DataFrame({"dep": ["77\xa0", " 75 ", "2A"]})
        out, _ = apply_known_fixes(df, "lieux")
        assert out["dep"].tolist() == ["77", "75", "2A"]

    def test_numeric_columns_untouched(self):
        df = pd.DataFrame({"jour": [1, 2, 3]})
        out, _ = apply_known_fixes(df, "caracteristiques")
        assert out["jour"].tolist() == [1, 2, 3]
