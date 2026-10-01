"""Import des prédictions de l'API K8s (déposées sur S3) dans la table predictions du VPS.

L'API K8s dépose ses prédictions réelles par lots dans
s3://cac-mlops-data/k8s-predictions/pending/ (services/api/app/s3_sink.py).
Avant chaque calcul de drift, ce module les insère dans la table predictions
avec source='k8s', puis range chaque fichier dans k8s-predictions/processed/.

Reprise sans doublon : un fichier déjà présent dans processed/ a déjà été
inséré — il est seulement retiré de pending/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

BUCKET = "cac-mlops-data"
PENDING = "k8s-predictions/pending/"
PROCESSED = "k8s-predictions/processed/"

# Même ordre que la table predictions (services/api/app/db.py) ; "int" est
# l'alias JSON de intersection_type côté API.
_FEATURES = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv", "obsm", "motor",
    "catr", "circ", "surf", "situ", "vma", "jour", "mois", "lum", "dep", "com", "agg_",
    "int", "atm", "col", "lat", "long", "hour", "nb_victim", "nb_vehicules",
]
_INSERT = """
INSERT INTO predictions (
    created_at, model_version, prediction, probability,
    place, catu, sexe, secu1, year_acc, victim_age, catv, obsm, motor,
    catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
    intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules, source
) VALUES (""" + ", ".join(f"${i}" for i in range(1, 34)) + ");"


def to_record(row: dict[str, Any]) -> tuple:
    """Une ligne JSON déposée par l'API K8s → tuple pour _INSERT."""
    return (
        datetime.fromisoformat(row["created_at"]),
        row.get("model_version"), row.get("prediction"), row.get("probability"),
        *(row.get(f) for f in _FEATURES),
        "k8s",
    )


def parse_lines(body: bytes) -> list[tuple]:
    return [to_record(json.loads(line)) for line in body.decode().splitlines() if line.strip()]


def _client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url="https://s3.fr-par.scw.cloud", region_name="fr-par",
        aws_access_key_id=os.getenv("SCW_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SCW_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


def _exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


async def _insert(records: list[tuple]) -> None:
    import asyncpg

    from services.api.app.db import _CREATE_TABLE, _build_dsn
    conn = await asyncpg.connect(_build_dsn())
    try:
        await conn.execute(_CREATE_TABLE)  # garantit la colonne source
        async with conn.transaction():
            await conn.executemany(_INSERT, records)
    finally:
        await conn.close()


def import_pending() -> dict[str, int]:
    """Importe tous les lots en attente. Ne lève jamais : un échec est journalisé
    et le fichier reste dans pending/ pour le prochain passage."""
    s3 = _client()
    files = rows = 0
    try:
        keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PENDING)
                for o in page.get("Contents", [])]
    except Exception:
        logger.warning("Lecture de s3://%s/%s impossible — import K8s ignoré", BUCKET, PENDING, exc_info=True)
        return {"files": 0, "rows": 0}

    for key in sorted(keys):
        done_key = PROCESSED + key[len(PENDING):]
        try:
            if not _exists(s3, done_key):
                records = parse_lines(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
                if records:
                    asyncio.run(_insert(records))
                s3.copy_object(Bucket=BUCKET, Key=done_key, CopySource={"Bucket": BUCKET, "Key": key})
                files += 1
                rows += len(records)
            s3.delete_object(Bucket=BUCKET, Key=key)
        except Exception:
            logger.warning("Import de %s échoué — réessayé au prochain passage", key, exc_info=True)
    logger.info("Prédictions K8s importées : %d ligne(s) depuis %d fichier(s)", rows, files)
    return {"files": files, "rows": rows}
