"""
ETL flow — download raw ONISR data, validate, push to DVC, preprocess,
push the preprocessed (clean) dataset to DVC.

Triggered manually, from check-new-data-flow (with pre-resolved URLs),
or as the first step of full_retrain_flow (once per cycle — each cycle's
year-combination gets its own versioned clean dataset).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.data.import_raw_data import download_year, training_years_up_to, get_training_years

GITHUB_REPO = os.getenv("GITHUB_REPO", "jakatt/cac_mlops")


@task(name="download-raw-data", retries=2, retry_delay_seconds=30)
def download_task(year: int, urls: dict[str, str] | None = None) -> None:
    log = get_run_logger()
    log.info("Downloading ONISR data for year %d", year)
    download_year(year, urls=urls)
    log.info("Download complete for year %d", year)


@task(name="validate-schema")
def validate_task(year: int) -> None:
    """3-level schema validation. Raises RuntimeError on CRITICAL — stoppe le pipeline."""
    from src.data.schema_validator import validate
    log = get_run_logger()
    report = validate(year)
    level = report.overall_level

    if level == "CRITICAL":
        log.error("event=alert severity=critical topic=schema_validation year=%d", year)
        raise RuntimeError(
            f"Schema validation CRITICAL pour year={year} — pipeline stoppé.\n"
            f"{report.summary()}"
        )

    if level == "WARNING":
        log.warning("event=alert severity=warning topic=schema_validation year=%d", year)

    log.info("Validation level=%s year=%d", level, year)


def _fetch_gh_pat(log) -> str | None:
    """Récupère le PAT GitHub depuis S3 (secrets/gh_pat) plutôt qu'une variable
    d'environnement figée à la création du conteneur.

    Pourquoi : /app (image api, utilisée par prefect-worker) ne contient jamais
    .git — le Dockerfile ne fait que des COPY sélectifs — et prefect-worker ne
    peut pas se recréer lui-même pour appliquer un changement docker-compose.yml
    (la tâche qui le ferait tourne dans le conteneur qu'elle recréerait,
    tuant le flow en cours). Lire le PAT depuis S3 à chaque exécution rend le
    mécanisme indépendant du cycle de vie du conteneur et du pipeline de
    déploiement : rotation du PAT = un nouvel upload S3, jamais de redémarrage.
    """
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url="https://s3.fr-par.scw.cloud",
            aws_access_key_id=os.environ["SCW_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["SCW_SECRET_ACCESS_KEY"],
        )
        obj = s3.get_object(Bucket="cac-mlops-data", Key="secrets/gh_pat")
        return obj["Body"].read().decode().strip()
    except Exception as exc:
        log.warning("Impossible de récupérer GH_PAT depuis S3 : %s", exc)
        return None


def _dvc_push_and_git_commit(path: str, commit_message: str, log) -> None:
    """dvc add --no-commit + dvc push + commit/push du .dvc — le tout dans un
    clone git jetable, créé et détruit à chaque exécution.

    Aucune dépendance à un .git présent dans /app (jamais le cas, cf.
    _fetch_gh_pat) : *path* (ex. data/raw/2024) pointe vers les données
    réelles montées dans /app — un lien symbolique dans le clone permet à dvc
    de calculer les hash sur les vrais fichiers sans les dupliquer.
    """
    pat = _fetch_gh_pat(log)
    if not pat:
        log.warning("GH_PAT indisponible — %s non versionné dans DVC/git.", path)
        return

    real_path = Path("/app") / path
    if not real_path.exists():
        log.warning("Chemin introuvable, rien à versionner : %s", real_path)
        return

    with tempfile.TemporaryDirectory(prefix="dvc-sync-") as tmp:
        clone_dir = Path(tmp) / "repo"
        repo_url = f"https://oauth2:{pat}@github.com/{GITHUB_REPO}.git"

        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo_url, str(clone_dir)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log.warning("git clone failed : %s", r.stderr.strip())
            return

        target = clone_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
        target.symlink_to(real_path)

        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "prefect-worker",
            "GIT_AUTHOR_EMAIL": "ci@cac-mlops.fr",
            "GIT_COMMITTER_NAME": "prefect-worker",
            "GIT_COMMITTER_EMAIL": "ci@cac-mlops.fr",
        }

        r = subprocess.run(
            ["dvc", "add", "--no-commit", path],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning("dvc add failed (%s) : %s", path, r.stderr.strip())
            return

        dvc_file = f"{path}.dvc"
        r = subprocess.run(
            ["dvc", "push", dvc_file],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning("dvc push failed (%s) : %s", path, r.stderr.strip())
            return
        log.info("dvc push OK — %s → Scaleway S3", path)

        subprocess.run(["git", "add", dvc_file], cwd=clone_dir, capture_output=True, env=env)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        ).stdout.strip()
        if not staged:
            log.info("%s déjà à jour dans git — rien à commiter", dvc_file)
            return

        subprocess.run(["git", "commit", "-m", commit_message], cwd=clone_dir, capture_output=True, env=env)
        r = subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning("git push failed : %s", r.stderr.strip())
        else:
            log.info("git push OK — %s → origin/main", dvc_file)


def _preprocessed_path(years: list[int]) -> str:
    """Chemin data/preprocessed/ correspondant à *years* — miroir de
    src.data.make_dataset._preprocessed_dir (mais en chemin relatif str,
    pratique pour les commandes dvc/git de ce module)."""
    label = "_".join(str(y) for y in sorted(years))
    subdir = label if len(years) == 1 else f"cumul_{label}"
    return f"data/preprocessed/{subdir}"


@task(name="dvc-sync-raw", retries=1, retry_delay_seconds=30)
def dvc_push_task(year: int) -> None:
    """Track raw data with DVC, push to Scaleway S3 et commite le .dvc dans git
    (source de vérité partagée) — clone jetable, cf. _dvc_push_and_git_commit."""
    log = get_run_logger()
    _dvc_push_and_git_commit(
        f"data/raw/{year}",
        f"data: DVC track {year} raw ONISR data [skip ci]",
        log,
    )


@task(name="preprocess-data")
def preprocess_task(years: list[int]) -> None:
    from src.data.make_dataset import process_years
    log = get_run_logger()
    log.info("Preprocessing years: %s", years)
    process_years(years)
    log.info("Preprocessing complete — %d years", len(years))


@task(name="dvc-sync-preprocessed", retries=1, retry_delay_seconds=30)
def dvc_push_preprocessed_task(years: list[int]) -> None:
    """Track le dataset préprocessé (clean) avec DVC, push sur Scaleway S3 et
    commite le .dvc dans git — permet à tout consommateur (DS local, VPS) de
    `dvc pull` le résultat exact de ce cycle ETL sans rejouer make_dataset."""
    log = get_run_logger()
    path = _preprocessed_path(years)
    label = path.rsplit("/", maxsplit=1)[-1]
    _dvc_push_and_git_commit(
        path,
        f"data: DVC track preprocessed {label} [skip ci]",
        log,
    )


@flow(name="etl-flow", flow_run_name="etl-year{year}", log_prints=True)
def etl_flow(
    year: int = 2023,
    cumul: bool = True,
    urls: dict[str, str] | None = None,
    explicit_years: list[int] | None = None,
) -> None:
    """
    Download, validate, push to DVC remote and preprocess ONISR data.

    urls: pre-resolved {category: download_url} — passed by check-new-data-flow.
    explicit_years: liste d'années à préprocesser (full_retrain_flow — replay historique).
                    Si None en mode cumul : auto-détection via get_training_years()
                    (toutes les années disponibles — la plus récente sert de test set
                    temporel dans process_years, sans être exclue du pipeline).
    """
    download_task(year, urls=urls)
    validate_task(year)
    dvc_push_task(year)
    if explicit_years is not None:
        years = explicit_years
    elif cumul:
        # Production : toutes les années disponibles (cf. get_training_years())
        years = get_training_years()
    else:
        years = [year]
    preprocess_task(years)
    dvc_push_preprocessed_task(years)
