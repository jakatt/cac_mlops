"""
Full retrain flow — chains all training cycles from scratch.
Auto-détecte les années disponibles dans data/raw/ (toutes, y compris la
plus récente — elle sert de test set temporel dans process_years, elle
n'est plus exclue de l'entraînement).
Benchmarks RF / XGBoost / LGBM each cycle, promotes the champion.

Cycle sequence (exemple avec 2021-2024 disponibles) :
  Cycle 1 : year=2021 cumul=false → benchmark → promote champion
  Cycle 2 : year=2022 cumul=true  → benchmark → promote → restart API
            → simulate 2022 (démo monitoring) → drift 2022 vs [2021]
  Cycle 3 : year=2023 cumul=true  → idem → drift 2023 vs [2021,2022]
  Cycle 4 : year=2024 cumul=true  → idem → drift 2024 vs [2021,2022,2023]

Le drift de chaque cycle est un drift de features pur (comparaison de la
nouvelle année vs les précédentes, cf. drift_detection.py) — indépendant du
modèle. simulate_task() sert uniquement à peupler des métriques de
monitoring réalistes (Grafana/API), plus à générer la donnée du drift.

Chaque cycle promeut son meilleur qualifié (gate KPI absolue uniquement,
compare_to_production=False) — pas de comparaison à @Production : celui-ci
serait fixé par le cycle précédent de ce même run compressé, pas une vraie
référence de production stable (bloquerait à tort les cycles suivants sur
une régression mineure et non représentative).
"""
import logging
import os
import time
from datetime import datetime

import requests as _req
from prefect import flow, task

from src.data.import_raw_data import discover_available_years, training_years_up_to
from src.flows.drift_monitoring_flow import drift_monitoring_flow
from src.flows.etl_flow import etl_flow
from src.flows.train_flow import train_flow

logger = logging.getLogger(__name__)


@task(name="detect-dvc-cycles")
def detect_cycles_task() -> list[tuple[int, bool]]:
    """
    Détecte les cycles d'entraînement depuis data/raw/.
    Training years = toutes les années disponibles (la plus récente sert de
    test set temporel dans process_years, cf. get_training_years()).
    Exemple : [2021, 2022, 2023, 2024] dispo → cycles sur les 4 années.
    """
    available = discover_available_years()
    if len(available) < 2:
        raise RuntimeError(
            f"full_retrain_flow requiert >= 2 années dans data/raw/. Disponibles : {available}"
        )
    cycles = [(y, i > 0) for i, y in enumerate(sorted(available))]
    logger.info("Années disponibles : %s — cycles : %s", available, cycles)
    return cycles


@task(name="restart-api")
def restart_api_task() -> None:
    """Restart the API container via Docker SDK so it loads the new @Production model."""
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "cac_mlops-api-1"})
        if not containers:
            logger.warning("API container not found — skipping restart")
            return
        containers[0].restart(timeout=30)
        logger.info("API container restarted — waiting for healthcheck...")
    except Exception as exc:
        logger.warning("Docker restart failed (%s) — continuing anyway", exc)
        return

    api_url = os.getenv("API_URL", "http://api:8000")
    for _ in range(12):  # max 60s
        try:
            if _req.get(f"{api_url}/health", timeout=3).status_code == 200:
                logger.info("API ready")
                return
        except Exception:
            pass
        time.sleep(5)
    logger.warning("API healthcheck timeout — continuing anyway")


@task(name="simulate-predictions", task_run_name="simulate-{sim_year}", retries=1, retry_delay_seconds=60)
def simulate_task(sim_year: int, sim_month: str, max_rows: int = 100) -> dict:
    """Send sim_year predictions to /predict with X-Sim-Date override for per-cycle drift isolation."""
    import requests
    from scripts.simulate_production import simulate

    api_url  = os.getenv("API_URL", "http://api:8000")
    username = os.getenv("API_USERNAME", "admin")
    password = os.getenv("API_PASSWORD", "changeme")

    resp = requests.post(
        f"{api_url}/token",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]

    result = simulate(
        api_url=api_url,
        token=token,
        year=sim_year,
        sim_month=sim_month,
        max_rows=max_rows,
    )
    logger.info("Simulation done (year=%d sim_month=%s): %s", sim_year, sim_month, result)
    return result


@flow(
    name="full-retrain-flow",
    flow_run_name=lambda: f"full-retrain-{datetime.now().strftime('%Y-%m-%d')}",
    log_prints=True,
)
def full_retrain_flow(max_sim_rows: int = 100) -> None:
    """
    Full from-scratch retrain pipeline.
    Pour chaque cycle détecté : ETL → train (année incluse, test set temporel)
    → promote → restart API → simulate (démo monitoring) → drift de features
    (année du cycle vs années précédentes, indépendant du modèle).
    Cycle 1 (première année) saute simulate/drift : aucune référence antérieure.
    Run reset-flow first to clear predictions table and drift reports.
    """
    cycles = detect_cycles_task()

    for i, (year, cumul) in enumerate(cycles):
        logger.info(
            "=== Cycle %d/%d — year=%d cumul=%s ===",
            i + 1, len(cycles), year, cumul,
        )

        # explicit_years = replay historique incrémental (pas auto-detect)
        explicit_years = training_years_up_to(year)
        etl_flow(year=year, cumul=cumul, explicit_years=explicit_years)
        # compare_to_production=False : chaque cycle promeut son meilleur qualifié
        # sans comparer à l'@Production fixé par le cycle précédent de ce même
        # replay compressé (pas une vraie référence stable — bloquerait à tort
        # les cycles suivants sur une régression mineure et non représentative).
        train_flow(year=year, cumul=cumul, promote=True,
                   require_improvement=False, compare_to_production=False)
        restart_api_task()

        if i > 0:
            # Simulation trafic (démo monitoring Grafana/API) — indépendante du drift
            simulate_task(sim_year=year, sim_month=f"{year}-06", max_rows=max_sim_rows)
            # Drift de features : year vs années précédentes (même dossier cumulatif)
            drift_monitoring_flow(year=year)

    logger.info("Full retrain complete — %d cycles", len(cycles))
