"""Async prediction logging to PostgreSQL via asyncpg."""
import asyncio
import logging
import os
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
        _pool = await asyncpg.create_pool(_build_dsn(), min_size=1, max_size=3)
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
) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                _INSERT,
                model_version, prediction, probability,
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
    except Exception:
        logger.debug("Failed to log prediction to DB", exc_info=True)
