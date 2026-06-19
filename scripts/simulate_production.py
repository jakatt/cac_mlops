"""
simulate_production.py — Replay 2024 ONISR data via POST /predict.

Simulates a real production load: sends the ~55k accidents 2024 in
monthly batches so the predictions table fills up for Evidently analysis.

Usage:
    python scripts/simulate_production.py
    python scripts/simulate_production.py --api-url http://localhost:8090 --token mytoken
    python scripts/simulate_production.py --month 2024-03   # one month only
    python scripts/simulate_production.py --dry-run         # count rows, don't send
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_2024 = Path("data/production/2024")
PREPROCESSED_2024 = Path("data/preprocessed/2024")

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "int", "atm", "col", "lat", "long",
    "hour", "nb_victim", "nb_vehicules",
]


def _get_token(api_url: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{api_url}/token",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _load_production_data() -> pd.DataFrame:
    """Load preprocessed 2024 data. Downloads + preprocesses if needed."""
    x_path = PREPROCESSED_2024 / "X_test.csv"

    if x_path.exists():
        logger.info("Loading preprocessed 2024 data from %s", x_path)
        return pd.read_csv(x_path)

    # Try raw data
    if not RAW_2024.exists() or not any(RAW_2024.iterdir()):
        logger.info("Downloading 2024 raw data from data.gouv.fr…")
        os.system(f"{sys.executable} -m src.data.import_raw_data --year 2024")

    logger.info("Preprocessing 2024 data…")
    os.system(f"{sys.executable} -m src.data.make_dataset --year 2024")

    if not x_path.exists():
        logger.error("Preprocessing failed — %s not found", x_path)
        sys.exit(1)

    return pd.read_csv(x_path)


def simulate(
    api_url: str,
    token: str,
    month: str | None = None,
    dry_run: bool = False,
    delay_ms: int = 0,
) -> dict:
    df = _load_production_data()

    # Keep only feature columns (handle missing ones gracefully)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing columns in 2024 data: %s — filling with 0", missing)
        for c in missing:
            df[c] = 0
    df = df[FEATURE_COLS].copy()

    # Map feature names: 'int' column (intersection_type alias)
    if "int" not in df.columns and "intersection_type" in df.columns:
        df = df.rename(columns={"intersection_type": "int"})

    # Assign synthetic month using 'mois' column (1-12)
    if month:
        target_month = int(month.split("-")[1])
        df = df[df["mois"] == target_month].copy()
        logger.info("Filtered to month %02d: %d rows", target_month, len(df))

    if df.empty:
        logger.warning("No rows to send")
        return {"sent": 0, "errors": 0}

    if dry_run:
        logger.info("DRY RUN — would send %d predictions (month=%s)", len(df), month or "all")
        return {"sent": 0, "errors": 0, "would_send": len(df)}

    headers = {"Authorization": f"Bearer {token}"}
    sent = errors = 0

    logger.info("Sending %d predictions to %s…", len(df), api_url)
    for i, (_, row) in enumerate(df.iterrows()):
        payload = {col: (None if pd.isna(v) else v) for col, v in row.items()}
        # Convert numpy types to Python native for JSON
        payload = {k: (int(v) if hasattr(v, 'item') and isinstance(v.item(), int) else
                       float(v) if hasattr(v, 'item') else v)
                   for k, v in payload.items()}
        try:
            resp = requests.post(
                f"{api_url}/predict",
                json=payload,
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                sent += 1
            else:
                errors += 1
                if errors <= 3:
                    logger.warning("Request %d: HTTP %d — %s", i, resp.status_code, resp.text[:120])
        except Exception as exc:
            errors += 1
            if errors <= 3:
                logger.warning("Request %d failed: %s", i, exc)

        if delay_ms:
            time.sleep(delay_ms / 1000)

        if (i + 1) % 1000 == 0:
            logger.info("  Progress: %d/%d (errors=%d)", i + 1, len(df), errors)

    logger.info("Done — sent=%d errors=%d", sent, errors)
    return {"sent": sent, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay 2024 accidents via POST /predict")
    parser.add_argument("--api-url",  default=os.getenv("API_URL", "http://localhost:8090"))
    parser.add_argument("--username", default=os.getenv("API_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("API_PASSWORD", "changeme"))
    parser.add_argument("--token",    default=None, help="JWT token (skips /token call)")
    parser.add_argument("--month",    default=None, help="YYYY-MM filter on 'mois' column")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--delay-ms", type=int, default=0, help="ms between requests")
    args = parser.parse_args()

    token = args.token or _get_token(args.api_url, args.username, args.password)
    result = simulate(
        api_url=args.api_url,
        token=token,
        month=args.month,
        dry_run=args.dry_run,
        delay_ms=args.delay_ms,
    )
    sys.exit(0 if result.get("errors", 0) == 0 else 1)
