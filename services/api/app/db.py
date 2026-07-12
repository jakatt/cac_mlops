"""Async prediction logging to PostgreSQL via asyncpg."""
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_pool = None

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    model_version TEXT,
    prediction    INT,
    probability   FLOAT,
    place         INT,
    catu          INT,
    sexe          INT,
    secu1         FLOAT,
    year_acc      INT,
    victim_age    FLOAT,
    catv          INT,
    obsm          INT,
    motor         INT,
    catr          INT,
    circ          FLOAT,
    surf          FLOAT,
    situ          FLOAT,
    vma           FLOAT,
    jour          INT,
    mois          INT,
    lum           INT,
    dep           INT,
    com           INT,
    agg_          INT,
    intersection_type INT,
    atm           FLOAT,
    col           FLOAT,
    lat           FLOAT,
    long          FLOAT,
    hour          INT,
    nb_victim     INT,
    nb_vehicules  INT
);
"""

_INSERT = """
INSERT INTO predictions (
    model_version, prediction, probability,
    place, catu, sexe, secu1, year_acc, victim_age, catv, obsm, motor,
    catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
    intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules
) VALUES (
    $1, $2, $3,
    $4, $5, $6, $7, $8, $9, $10, $11, $12,
    $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
    $24, $25, $26, $27, $28, $29, $30, $31
);
"""

# Variante avec created_at explicite — utilisée par simulate_production en mode multi-cycle
# pour que chaque cycle de simulation soit daté dans sa propre année (drift distinct par cycle)
_INSERT_WITH_DATE = """
INSERT INTO predictions (
    created_at,
    model_version, prediction, probability,
    place, catu, sexe, secu1, year_acc, victim_age, catv, obsm, motor,
    catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
    intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules
) VALUES (
    $1,
    $2, $3, $4,
    $5, $6, $7, $8, $9, $10, $11, $12, $13,
    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24,
    $25, $26, $27, $28, $29, $30, $31, $32
);
"""


def _build_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "mlops")
    pwd  = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST", "postgresql")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "mlops")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


async def init_db() -> None:
    global _pool
    try:
        import asyncpg
        # timeout court et explicite : le pool par défaut d'asyncpg attend 60s
        # avant d'abandonner, ce qui grille presque toute la fenêtre de
        # readiness K8s (30s + 6×10s = 90s, cf. k8s/api/deployment.yaml) et a
        # fait échouer 2 rollouts d'affilée le 2026-07-12 — la connexion
        # traverse le Tailscale jusqu'au Postgres du VPS depuis K8s (locale
        # et quasi instantanée sur le VPS lui-même, d'où l'incident invisible
        # jusqu'ici). Ce chemin est déjà "best effort" (dégradation propre
        # ci-dessous) : pas de raison d'attendre une minute pleine.
        _pool = await asyncpg.create_pool(_build_dsn(), min_size=1, max_size=3, timeout=5)
        async with _pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
        logger.info("DB pool ready — predictions table ensured")
    except Exception:
        logger.warning("DB init failed — prediction logging disabled", exc_info=True)
        _pool = None


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def log_prediction(
    features: dict[str, Any],
    prediction: int,
    probability: float,
    model_version: str,
    sim_date: str | None = None,
) -> None:
    """Log prediction to DB. sim_date='YYYY-MM' overrides created_at for simulation cycles."""
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            feat_vals = (
                features.get("place"), features.get("catu"), features.get("sexe"),
                features.get("secu1"), features.get("year_acc"), features.get("victim_age"),
                features.get("catv"), features.get("obsm"), features.get("motor"),
                features.get("catr"), features.get("circ"), features.get("surf"),
                features.get("situ"), features.get("vma"), features.get("jour"),
                features.get("mois"), features.get("lum"), features.get("dep"),
                features.get("com"), features.get("agg_"), features.get("int"),
                features.get("atm"), features.get("col"), features.get("lat"),
                features.get("long"), features.get("hour"),
                features.get("nb_victim"), features.get("nb_vehicules"),
            )
            if sim_date:
                year, month = sim_date.split("-")
                created_at = datetime(int(year), int(month), 15, 12, 0, 0, tzinfo=timezone.utc)
                await conn.execute(
                    _INSERT_WITH_DATE,
                    created_at, model_version, prediction, probability, *feat_vals,
                )
            else:
                await conn.execute(
                    _INSERT,
                    model_version, prediction, probability, *feat_vals,
                )
    except Exception:
        logger.debug("Failed to log prediction to DB", exc_info=True)
