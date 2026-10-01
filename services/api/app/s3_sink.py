"""Enregistrement des prédictions de l'API K8s sur S3 (PREDICTIONS_SINK=s3).

K8s n'a pas de base de données et ne doit pas dépendre du VPS (le PostgreSQL
du VPS n'est d'ailleurs pas exposé). Les prédictions réelles sont donc gardées
en mémoire et déposées par lots dans le bucket S3 Scaleway, un fichier JSON
Lines par lot :

    s3://cac-mlops-data/k8s-predictions/pending/<horodatage>_<pod>_<id>.jsonl

Le flow prediction-drift-check du VPS les importe dans la table predictions
avant chaque calcul de drift (services/monitoring/import_k8s_predictions.py).
Si le VPS est en panne, les fichiers attendent simplement dans S3.

Seule perte possible : le lot en mémoire si le pod est tué brutalement
(au plus FLUSH_EVERY_S secondes de prédictions).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

BUCKET = os.getenv("PREDICTIONS_S3_BUCKET", "cac-mlops-data")
PREFIX = "k8s-predictions/pending/"
ENDPOINT = os.getenv("S3_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
FLUSH_EVERY_S = int(os.getenv("PREDICTIONS_FLUSH_EVERY_S", "300"))
FLUSH_ROWS = int(os.getenv("PREDICTIONS_FLUSH_ROWS", "100"))
MAX_BUFFER = 20_000  # garde-fou mémoire si S3 reste injoignable longtemps

_buffer: list[dict[str, Any]] = []
_lock = asyncio.Lock()
_task: asyncio.Task | None = None


def _client():
    import boto3
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name="fr-par")


def build_row(features: dict[str, Any], prediction: int, probability: float,
              model_version: str) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "prediction": prediction,
        "probability": probability,
        **features,
    }


def object_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{PREFIX}{now.strftime('%Y%m%dT%H%M%S')}_{socket.gethostname()}_{uuid.uuid4().hex[:8]}.jsonl"


def _put(rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    _client().put_object(Bucket=BUCKET, Key=object_key(), Body=body, ContentType="application/x-ndjson")


async def flush() -> int:
    """Dépose le lot en mémoire sur S3. En cas d'échec, le lot est conservé
    pour le prochain essai. Retourne le nombre de lignes déposées."""
    async with _lock:
        if not _buffer:
            return 0
        rows = list(_buffer)
        try:
            await asyncio.to_thread(_put, rows)
        except Exception:
            logger.warning("Dépôt S3 de %d prédiction(s) échoué — nouvel essai au prochain lot", len(rows), exc_info=True)
            return 0
        del _buffer[:len(rows)]
        logger.info("%d prédiction(s) déposée(s) sur s3://%s/%s", len(rows), BUCKET, PREFIX)
        return len(rows)


async def add(row: dict[str, Any]) -> None:
    async with _lock:
        _buffer.append(row)
        if len(_buffer) > MAX_BUFFER:
            del _buffer[: len(_buffer) - MAX_BUFFER]
        full = len(_buffer) >= FLUSH_ROWS
    if full:
        await flush()


async def _periodic() -> None:
    while True:
        await asyncio.sleep(FLUSH_EVERY_S)
        await flush()


def start() -> None:
    global _task
    if _task is None:
        _task = asyncio.get_running_loop().create_task(_periodic())
        logger.info("Prédictions enregistrées sur S3 (s3://%s/%s), par lots de %d ou toutes les %d s",
                    BUCKET, PREFIX, FLUSH_ROWS, FLUSH_EVERY_S)


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
    await flush()
