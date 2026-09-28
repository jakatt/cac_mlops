"""
Pandera schemas for the 4 ONISR CSV files (format stable 2021-2024).

strict=False  → unknown columns trigger WARNING, not CRITICAL
strict=True   → would raise SchemaError on any unknown column
coerce=True   → tente de convertir vers le type déclaré (ex. colonne 'int'
                lue comme string à cause d'un artefact de formatage) ; lève
                une SchemaErrors explicite si la conversion est impossible
                (jamais de NaN silencieux) — cf. schema_validator._validate_level2
                qui capture le DataFrame coercé et logue ce qui a été corrigé.
"""
import pandera.pandas as pa

# ── caracteristiques ──────────────────────────────────────────────────────────
CARACTERISTIQUES_SCHEMA = pa.DataFrameSchema(
    columns={
        "Num_Acc": pa.Column(object, nullable=False),
        "jour":    pa.Column(int, pa.Check.in_range(1, 31)),
        "mois":    pa.Column(int, pa.Check.in_range(1, 12)),
        "hrmn":    pa.Column(object, nullable=True),        # "HHMM" string → hour int
        "lum":     pa.Column(int, pa.Check.isin([-1, 1, 2, 3, 4, 5])),
        "dep":     pa.Column(object, nullable=False),       # string: "77", "2A", "2B"
        "com":     pa.Column(object, nullable=True),
        "agg":     pa.Column(int, pa.Check.isin([-1, 1, 2])),
        "int":     pa.Column(int),
        "atm":     pa.Column(int, pa.Check.in_range(-1, 9), nullable=True),
        "col":     pa.Column(int, nullable=True),
        "lat":     pa.Column(object, nullable=True),        # "48,60" before normaliz.
        "long":    pa.Column(object, nullable=True),
    },
    strict=False,       # extra columns (adr, an…) → WARNING, not CRITICAL
    coerce=True,
)

# ── lieux ─────────────────────────────────────────────────────────────────────
LIEUX_SCHEMA = pa.DataFrameSchema(
    columns={
        "Num_Acc": pa.Column(object, nullable=False),
        "catr":    pa.Column(int, nullable=True),
        "circ":    pa.Column(int, nullable=True),
        "surf":    pa.Column(int, nullable=True),
        "situ":    pa.Column(int, nullable=True),
        "vma":     pa.Column(int, nullable=True),
    },
    strict=False,
    coerce=True,
)

# ── usagers ───────────────────────────────────────────────────────────────────
USAGERS_SCHEMA = pa.DataFrameSchema(
    columns={
        "Num_Acc":      pa.Column(object, nullable=False),
        "id_vehicule":  pa.Column(object, nullable=False),
        "num_veh":      pa.Column(object, nullable=False),
        "place":        pa.Column(int, nullable=True),
        "catu":         pa.Column(int, pa.Check.isin([-1, 1, 2, 3, 4])),
        "grav":         pa.Column(int, pa.Check.isin([-1, 1, 2, 3, 4])),
        "sexe":         pa.Column(int, pa.Check.isin([-1, 1, 2])),
        "an_nais":      pa.Column(float, nullable=True),
        "secu1":        pa.Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)

# ── vehicules ─────────────────────────────────────────────────────────────────
VEHICULES_SCHEMA = pa.DataFrameSchema(
    columns={
        "Num_Acc":      pa.Column(object, nullable=False),
        "id_vehicule":  pa.Column(object, nullable=False),
        "num_veh":      pa.Column(object, nullable=False),
        "catv":         pa.Column(int, nullable=True),
        "obsm":         pa.Column(int, nullable=True),
        "motor":        pa.Column(int, nullable=True),
    },
    strict=False,
    coerce=True,
)

# ── mapping table → schema ────────────────────────────────────────────────────
TABLE_SCHEMAS = {
    "caracteristiques": CARACTERISTIQUES_SCHEMA,
    "lieux":            LIEUX_SCHEMA,
    "usagers":          USAGERS_SCHEMA,
    "vehicules":        VEHICULES_SCHEMA,
}

# Required columns per table (CRITICAL if missing)
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "caracteristiques": ["Num_Acc", "jour", "mois", "lum", "dep", "agg", "int", "lat", "long"],
    "lieux":            ["Num_Acc", "catr", "circ", "surf", "situ", "vma"],
    "usagers":          ["Num_Acc", "id_vehicule", "num_veh", "catu", "grav", "sexe", "secu1"],
    "vehicules":        ["Num_Acc", "id_vehicule", "num_veh", "catv", "obsm", "motor"],
}

# Colonnes ONISR connues mais volontairement non utilisées par le modèle
# (présentes chaque année depuis 2021) : signalées en INFO, pas en WARNING.
# Seule une colonne réellement NOUVELLE (absente d'ici et du schéma) lève un
# WARNING — sans cette liste, chaque ETL finissait en WARNING (alerte email)
# pour les mêmes colonnes, signal devenu du bruit (constaté 2026-09-28).
KNOWN_UNUSED_COLUMNS: dict[str, list[str]] = {
    "caracteristiques": ["adr", "an"],
    "lieux":            ["infra", "larrout", "lartpc", "nbv", "plan", "pr", "pr1",
                         "prof", "v1", "v2", "voie", "vosp"],
    "usagers":          ["actp", "etatp", "id_usager", "locp", "secu2", "secu3", "trajet"],
    "vehicules":        ["choc", "manv", "obs", "occutc", "senc"],
}

# Colonnes codifiées ONISR où -1 signifie « non renseigné » : une valeur hors
# nomenclature (nouveau code, texte parasite) y est ramenée à -1 plutôt que de
# laisser toute la table non typée (cf. schema_validator._fix_out_of_nomenclature).
# Colonnes float nullables : ramenées à NaN.
NOMENCLATURE_INT_COLUMNS = {
    "lum", "agg", "int", "atm", "col", "catr", "circ", "surf", "situ", "vma",
    "place", "catu", "grav", "sexe", "catv", "obsm", "motor",
}
NOMENCLATURE_FLOAT_COLUMNS = {"secu1", "an_nais"}

# Quality bounds for Level-3 checks
QUALITY_BOUNDS = {
    "accident_count_min": 40_000,
    "accident_count_max": 90_000,
    "nan_rate_warning":   0.30,
    # Bounding box France métropolitaine + Corse. Les accidents en DROM/COM
    # (Guadeloupe, Réunion…) sont légitimes et en dehors (~5-6 % par an) :
    # WARNING seulement au-delà de la part tolérée ci-dessous.
    "lat_min": 41.0,  "lat_max": 51.5,
    "lon_min": -5.5,  "lon_max": 9.6,
    "outside_metropole_share_warning": 0.10,
}
