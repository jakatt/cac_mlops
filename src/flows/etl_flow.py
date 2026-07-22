"""
ETL flow — download raw ONISR data, validate, push to DVC, preprocess,
push the preprocessed (clean) dataset to DVC.

Triggered manually, from check-new-data-flow (with pre-resolved URLs),
or as the first step of full_retrain_flow (once per cycle — each cycle's
year-combination gets its own versioned clean dataset).
"""
import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.data.import_raw_data import download_year, training_years_up_to, get_training_years


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


def _dvc_track_and_push(path: str, log) -> bool:
    """dvc add --no-commit + dvc push pour *path*. Retourne True si le push a réussi."""
    dvc_file = Path(f"{path}.dvc")

    r = subprocess.run(
        ["dvc", "add", "--no-commit", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.warning("dvc add failed (%s): %s", path, r.stderr.strip())
        return False

    push_target = str(dvc_file) if dvc_file.exists() else path
    r = subprocess.run(
        ["dvc", "push", push_target],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.warning("dvc push failed (%s): %s", path, r.stderr.strip())
        return False

    log.info("dvc push OK — %s → Scaleway S3", path)
    return True


def _git_commit_dvc_file(path: str, commit_message: str, log) -> None:
    """Commite {path}.dvc dans git et push vers origin/main.

    Requiert GH_PAT dans l'env : token GitHub avec scope 'repo' (contents write).
    Le commit utilise [skip ci] — deploy.yml a aussi un paths-ignore data/**
    pour qu'un commit .dvc ne déclenche jamais le CD (Trigger 1 = Prefect only).
    """
    import os

    dvc_file = f"{path}.dvc"
    pat = os.getenv("GH_PAT", "")
    if not pat:
        log.warning(
            "GH_PAT non défini — %s non commité dans git. "
            "Le DS ne pourra pas dvc pull cette donnée.",
            dvc_file,
        )
        return

    r = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    remote = r.stdout.strip()
    if "https://" in remote:
        auth_remote = remote.replace("https://", f"https://oauth2:{pat}@")
    else:
        repo_path = remote.split(":")[-1].removesuffix(".git")
        auth_remote = f"https://oauth2:{pat}@github.com/{repo_path}.git"

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "prefect-worker",
        "GIT_AUTHOR_EMAIL": "ci@cac-mlops.fr",
        "GIT_COMMITTER_NAME": "prefect-worker",
        "GIT_COMMITTER_EMAIL": "ci@cac-mlops.fr",
    }

    subprocess.run(["git", "add", dvc_file], capture_output=True, env=env)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, env=env,
    ).stdout.strip()
    if not staged:
        log.info("%s déjà tracké dans git — rien à commiter", dvc_file)
        return

    subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, env=env)
    r = subprocess.run(
        ["git", "push", auth_remote, "HEAD:main"],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        log.warning("git push failed: %s", r.stderr.strip())
    else:
        log.info("git push OK — %s → origin/main", dvc_file)


def _preprocessed_path(years: list[int]) -> str:
    """Chemin data/preprocessed/ correspondant à *years* — miroir de
    src.data.make_dataset._preprocessed_dir (mais en chemin relatif str,
    pratique pour les commandes dvc/git de ce module)."""
    label = "_".join(str(y) for y in sorted(years))
    subdir = label if len(years) == 1 else f"cumul_{label}"
    return f"data/preprocessed/{subdir}"


@task(name="dvc-push", retries=1, retry_delay_seconds=30)
def dvc_push_task(year: int) -> None:
    """Track raw data with DVC and push to Scaleway S3 (source de vérité partagée)."""
    log = get_run_logger()
    _dvc_track_and_push(f"data/raw/{year}", log)


@task(name="dvc-git-commit", retries=1, retry_delay_seconds=10)
def dvc_git_commit_task(year: int) -> None:
    log = get_run_logger()
    _git_commit_dvc_file(
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


@task(name="dvc-push-preprocessed", retries=1, retry_delay_seconds=30)
def dvc_push_preprocessed_task(years: list[int]) -> None:
    """Track le dataset préprocessé (clean) avec DVC et push sur Scaleway S3 —
    permet à tout consommateur (DS local, VPS) de `dvc pull` le résultat exact
    de ce cycle ETL sans avoir à rejouer make_dataset localement."""
    log = get_run_logger()
    _dvc_track_and_push(_preprocessed_path(years), log)


@task(name="dvc-git-commit-preprocessed", retries=1, retry_delay_seconds=10)
def dvc_git_commit_preprocessed_task(years: list[int]) -> None:
    log = get_run_logger()
    path = _preprocessed_path(years)
    label = path.rsplit("/", maxsplit=1)[-1]
    _git_commit_dvc_file(
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
    dvc_git_commit_task(year)
    if explicit_years is not None:
        years = explicit_years
    elif cumul:
        # Production : toutes les années disponibles (cf. get_training_years())
        years = get_training_years()
    else:
        years = [year]
    preprocess_task(years)
    dvc_push_preprocessed_task(years)
    dvc_git_commit_preprocessed_task(years)
