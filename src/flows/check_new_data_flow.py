"""
Weekly detection flow — checks data.gouv.fr for new ONISR accident data.

What it does:
  - Polls data.gouv.fr API for the ONISR dataset
  - Detects years not yet in git DVC tags (data-v1, data-v2, ...)
  - Fuzzy-matches the 4 CSV files (caracteristiques, lieux, usagers, vehicules) by URL
  - If 4/4 matched → auto-trigger: etl → train → deploy-vps (gate manuelle) → deploy-kapsule

What it does NOT do automatically:
  - If < 4/4 match → email alerte + stop (ONISR changed naming convention)

Full automation chain (nouvelle data → prod):
  check-new-data-flow
    → etl-flow(year=N, urls={...})       — download par URL, pas de FILENAMES
    → train-flow(year=N, cumul=True)     — benchmark, champion sélectionné sans promote
                                            (N inclus dans l'entraînement — test set temporel)
    → deploy-vps-flow(champion, metrics) — gate manuelle dans Prefect UI
    → deploy-kapsule-flow()              — rolling update automatique si Kapsule actif
    → drift-monitoring-flow(year=N)      — drift de features N vs années précédentes
                                            (indépendant du modèle, alerte seulement)
"""
import requests
from prefect import flow, task, get_run_logger

from src.data.import_raw_data import CATEGORY_KEYWORDS, _DATASET_ID, training_years_up_to
from src.utils.email_utils import send_alert

_DATA_GOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{_DATASET_ID}/"


def _versioned_years() -> set[int]:
    """Years with downloaded raw data (4+ CSV files in data/raw/{year}/).

    Remplace l'ancienne détection par git tags (data-v*) qui échouait
    silencieusement dans le container Prefect (pas de .git à /app).
    """
    from src.data.import_raw_data import PROJECT_ROOT, FIRST_TRAINING_YEAR
    raw_root = PROJECT_ROOT / "data" / "raw"
    known: set[int] = set()
    if raw_root.exists():
        for d in raw_root.iterdir():
            if d.is_dir() and d.name.isdigit() and FIRST_TRAINING_YEAR <= int(d.name) <= 2030:
                if len(list(d.glob("*.csv"))) >= 4:
                    known.add(int(d.name))
    return known


@task(name="fetch-datagouv-resources")
def fetch_resources_task() -> list[dict]:
    """Return all resources from the ONISR dataset on data.gouv.fr."""
    log = get_run_logger()
    resp = requests.get(_DATA_GOUV_API, timeout=15)
    resp.raise_for_status()
    resources = resp.json().get("resources", [])
    log.info("data.gouv.fr — %d resources found in ONISR dataset", len(resources))
    return resources


@task(name="detect-new-year")
def detect_new_year_task(resources: list[dict]) -> tuple[int | None, dict[str, str]]:
    """
    Check if a year beyond current DVC tags is available on data.gouv.fr.
    Returns (new_year, {category: url}) or (None, {}).
    """
    log = get_run_logger()
    known = _versioned_years()
    max_known = max(known) if known else 2023

    for year in range(max_known + 1, max_known + 3):
        year_str = str(year)
        year_resources = [r for r in resources if year_str in r.get("title", "")]

        if not year_resources:
            continue

        matched: dict[str, str] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            for r in year_resources:
                title = r.get("title", "").lower()
                if any(kw in title for kw in keywords):
                    matched[category] = r.get("url", "")
                    break

        if len(matched) == 4:
            log.info("New year %d detected — all 4 files matched", year)
            return year, matched

        missing = set(CATEGORY_KEYWORDS) - set(matched)
        log.warning(
            "Year %d found on data.gouv.fr but only %d/4 files matched "
            "(missing: %s) — manual review needed.\nAll titles for %d: %s",
            year, len(matched), missing, year,
            [r.get("title") for r in year_resources],
        )
        return year, matched  # partial match returned for alert

    log.info("No new year detected beyond %d — known: %s", max_known, sorted(known))
    return None, {}


@flow(name="check-new-data-flow", log_prints=True)
def check_new_data_flow() -> None:
    """
    Weekly: detect new ONISR accident data.
    If 4/4 files found → auto-trigger full chain: ETL → train → deploy (gate manuelle).
    If < 4/4 → email alerte + stop.
    """
    from src.flows.etl_flow import etl_flow
    from src.flows.train_flow import train_flow
    from src.flows.deploy_vps_flow import deploy_vps_flow
    from src.flows.drift_monitoring_flow import drift_monitoring_flow

    log = get_run_logger()
    resources = fetch_resources_task()
    new_year, matched_urls = detect_new_year_task(resources)

    if new_year is None:
        log.info("Nothing to do — dataset is up to date.")
        return

    if len(matched_urls) < 4:
        missing = set(CATEGORY_KEYWORDS) - set(matched_urls)
        msg = (
            f"Année {new_year} partiellement disponible ({len(matched_urls)}/4 fichiers).\n"
            f"Catégories manquantes : {missing}\n"
            f"Fichiers trouvés : {matched_urls}\n"
            f"→ Consulter data.gouv.fr manuellement."
        )
        log.warning("event=alert severity=warning topic=onisr_partial_match year=%d matched=%d", new_year, len(matched_urls))
        send_alert(f"Données ONISR {new_year} — revue manuelle requise", msg)
        return

    # ── Toutes les données disponibles → chaîne complète ──────────────────────
    log.info(
        "\n═══════════════════════════════════════════════════════════\n"
        "  NOUVELLE ANNEE ONISR : %d — 4/4 fichiers matchés\n"
        "  Lancement de la chaîne ETL → Train → Deploy\n"
        "═══════════════════════════════════════════════════════════",
        new_year,
    )

    # ETL : download par URLs (pas de FILENAMES), preprocess cumul 2021→N
    etl_flow(year=new_year, cumul=True, urls=matched_urls)

    # Train : benchmark + sélection champion, sans promote (gate d'abord)
    # require_improvement=False (Trigger 1) : gate KPI absolue + tolérance de
    # régression ≤1 métrique vs @Production (nouvelles données = valeur ajoutée
    # même sans strict dépassement du F1 actuel — cf. select_champion_task).
    result = train_flow(year=new_year, cumul=True, promote=False, require_improvement=False)

    if result["champion"] is None:
        msg = (
            f"Training année {new_year} terminé mais aucun algorithme ne passe la gate KPI "
            f"absolue, ou tous régressent sur ≥2 métriques vs @Production.\n"
            f"Métriques : {result['metrics']}"
        )
        log.warning("Training ONISR %d conclu sans champion (cf. event=alert topic=no_champion émis par select-champion-task)", new_year)
        send_alert(f"Training ONISR {new_year} — aucun champion promu", msg)
        return

    # Deploy VPS avec gate manuelle + promote après validation + deploy Kapsule
    deploy_vps_flow(
        champion=result["champion"],
        run_ids=result["run_ids"],
        metrics=result["metrics"],
        year=new_year,
    )

    # Drift de features new_year vs années précédentes — indépendant du modèle,
    # calculé même si le champion n'a pas encore été validé/promu (alerte seule,
    # jamais de retrain automatique).
    drift_monitoring_flow(year=str(new_year))
