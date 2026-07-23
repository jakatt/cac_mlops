"""
Extrait les hyperparamètres d'un run champion, met à jour config/model_params.yml
et pousse jusqu'à la création de la PR (commit + push + gh pr create).

Deux usages :
  1. Avec run_id (recommandé — une seule commande, pas d'aller-retour UI) :
       python -m src.scripts.extract_blueprint <run_id>
     Tague ce run export_to_prod=true puis extrait directement. Le run_id
     s'obtient dans le tableau affiché par train_model.py en fin de training
     ("Pour promouvoir ce run...") — pas besoin d'aller le chercher dans l'UI.

  2. Sans run_id (ancien flux — tag posé à la main dans l'UI MLflow au préalable) :
       python -m src.scripts.extract_blueprint
     Cherche le dernier run tagué export_to_prod=true dans accidents_severity_dev.

Usage (outil DS local — non appelé par le pipeline de prod) :
    python -m src.scripts.extract_blueprint [run_id] [--dry-run] [--no-pr]

--dry-run : affiche sans rien écrire ni committer.
--no-pr   : écrit config/model_params.yml mais s'arrête là (pas de git/PR) —
            pour relire/ajuster le YAML à la main avant de committer soi-même.

Invocation via -m (pas python src/scripts/extract_blueprint.py) : déclenche
src/__init__.py (charge .env — MLFLOW_TRACKING_URI y compris). Sans ça, le
défaut http://mlflow:5000 (hostname interne conteneur, injoignable en local)
fait planter en silence (hang réseau, pas d'erreur claire).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import yaml

from src.models.train_model import KPI_THRESHOLDS, compute_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

EXPLORE_EXPERIMENT = "accidents_severity_dev"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_params.yml"
REPO_ROOT = CONFIG_PATH.parent.parent
KNOWN_ALGOS = {"rf", "xgboost", "lgbm"}

# Params loggés par train() en plus du blueprint (mlflow.log_param direct,
# pas depuis config/model_params.yml) — à exclure de l'extraction.
_META_PARAMS = {"algorithm", "years", "n_train", "n_test", "n_features"}


def _coerce(value: str) -> object:
    """Reconvertit une valeur MLflow (toujours stockée en string) vers son
    type réel — même ordre que _max_features_type côté CLI (train_model.py) :
    bool/None explicites, puis int avant float (sinon '5' redeviendrait 5.0),
    sinon string telle quelle ('sqrt', 'gini', 'balanced'...)."""
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _find_export_run(client: mlflow.MlflowClient) -> mlflow.entities.Run | None:
    """Retourne le run le plus récent avec tag export_to_prod=true."""
    try:
        exp = client.get_experiment_by_name(EXPLORE_EXPERIMENT)
        if exp is None:
            logger.warning("Expérience '%s' introuvable dans MLflow", EXPLORE_EXPERIMENT)
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.export_to_prod = 'true'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("Aucun run tagué export_to_prod=true dans '%s'", EXPLORE_EXPERIMENT)
            return None
        return runs[0]
    except Exception as exc:
        logger.error("Erreur lors de la recherche du run champion : %s", exc)
        return None


def _run_git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def _current_branch() -> str:
    return _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def _build_pr_body(algo: str, run_id: str, comp: dict) -> str:
    """Corps de PR Markdown — reprend le tableau de comparaison affiché par
    train_model.py (mêmes données, via compute_comparison), pas une recopie
    manuelle du texte terminal."""
    rows_md = "\n".join(
        f"| {r['metric']} | {r['prod']:.4f} | {r['exp']:.4f} | {r['delta']:+.4f} |"
        if r["prod"] is not None else f"| {r['metric']} | — | {r['exp']:.4f} | — |"
        for r in comp["rows"]
    )
    prod_label = comp["prod_model_name"] or "aucun @Production existant"
    global_label = "N/A" if comp["all_better"] is None else ("OUI" if comp["all_better"] else "NON")
    verdict = "OUI" if comp["send_to_prod"] else "NON"

    if comp["prod_model_name"] is None:
        model_line = f"**Modèle proposé :** `{comp['exp_model_name']}` (aucun @Production existant)"
    elif comp["prod_model_name"] == comp["exp_model_name"]:
        model_line = f"**Modèle proposé :** `{comp['exp_model_name']}` (même famille que l'actuel @Production)"
    else:
        model_line = (
            f"**Modèle proposé :** `{comp['exp_model_name']}` "
            f"— ⚠️ remplacerait `{comp['prod_model_name']}` (actuellement @Production)"
        )

    decision_note = (
        f"✅ **Send to prod ? OUI**  ({comp['reason']}) — ce run dépasse le seuil de promotion automatique "
        "(Trigger 3 : gate KPI + f1 ≥ +0.01 vs @Production). Rappel de la règle : `update-model-flow` réentraîne "
        "et compare à nouveau après merge (pas une simple relecture de ce fichier) — sur les mêmes données, ce "
        "run devrait donc être confirmé et promu automatiquement, sans être filtré par le pipeline."
        if comp["send_to_prod"]
        else f"⚠️ **Send to prod ? NON**  ({comp['reason']}) — ce run **ne dépasse pas** le seuil de promotion "
             "automatique (Trigger 3 : gate KPI + f1 ≥ +0.01 vs @Production). Il est soumis malgré la "
             "recommandation — `update-model-flow` réentraînera et comparera après merge, et ne promouvra "
             "**pas** ce blueprint s'il ne dépasse pas réellement ce seuil à ce moment-là. Décision du DS à "
             "documenter ici si pertinent (ex. amélioration jugée significative sur les autres métriques, "
             "contexte métier)."
    )

    return f"""## Résumé

Nouveau blueprint **{algo}** proposé suite à une session d'exploration DS — run MLflow [`{run_id}`](http://{os.getenv("MLFLOW_TRACKING_URI", "").replace("http://", "")}/#/experiments/{EXPLORE_EXPERIMENT}/runs/{run_id}) tagué `export_to_prod=true`.

{model_line}

## Comparaison vs @Production ({prod_label})

| Métrique | Prod | Expérience | Delta |
|---|---|---|---|
{rows_md}

**Global (4 métriques meilleures) :** {global_label}
**Send to prod ?** {verdict}  ({comp["reason"]})

## Décision DS

{decision_note}

## Ce qui se passe après merge

`deploy.yml` détecte le changement de `config/model_params.yml` et déclenche `update-model-flow` (Trigger 3) : réentraîne les 3 algos sur les données de prod à ce moment-là, compare à `@Production`, et ne promeut que si le nouveau champion dépasse réellement le seuil — **indépendamment du tableau ci-dessus**, calculé sur le jeu de données local de la session d'exploration (potentiellement différent de celui de prod au moment du merge).

⚠️ Ne pas mélanger cette modification avec des changements de code source (`src/`, `services/`) dans la même PR — le CI bloquera avec un message d'erreur explicite.
"""


def _commit_push_create_pr(algo: str, run_id: str, comp: dict) -> bool:
    branch = _current_branch()
    if branch == "main":
        logger.error("Sur 'main' — bascule sur ta branche de travail (DS) avant de continuer. Rien fait.")
        return False

    existing = subprocess.run(
        ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if existing.returncode == 0 and existing.stdout.strip() not in ("", "[]"):
        logger.error(
            "Une PR est déjà ouverte pour la branche '%s' — attends son merge avant d'en ouvrir une autre. Rien fait.",
            branch,
        )
        return False

    r = _run_git("add", str(CONFIG_PATH.relative_to(REPO_ROOT)))
    if r.returncode != 0:
        logger.error("git add a échoué : %s", r.stderr.strip())
        return False

    staged = _run_git("diff", "--cached", "--name-only").stdout.strip()
    if not staged:
        logger.info("config/model_params.yml déjà à jour dans git — rien à committer/pousser")
        return True

    title = f"blueprint: nouveaux hyperparamètres {algo} (run {run_id[:8]})"
    r = _run_git("commit", "-m", title)
    if r.returncode != 0:
        logger.error("git commit a échoué : %s", r.stderr.strip())
        return False

    r = _run_git("push", "origin", branch)
    if r.returncode != 0:
        logger.error("git push a échoué : %s", r.stderr.strip())
        return False
    logger.info("Poussé sur origin/%s", branch)

    body = _build_pr_body(algo, run_id, comp)
    r = subprocess.run(
        ["gh", "pr", "create", "--base", "main", "--head", branch, "--title", title, "--body", body],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.error("gh pr create a échoué : %s", r.stderr.strip())
        return False
    logger.info("PR créée : %s", r.stdout.strip())
    return True


def extract_blueprint(run_id: str | None = None, dry_run: bool = False, no_pr: bool = False) -> bool:
    """
    Extrait les hyperparamètres du run champion et met à jour config/model_params.yml.
    Si run_id est fourni : tague ce run export_to_prod=true puis l'utilise directement
    (une seule commande). Sinon : cherche le dernier run déjà tagué manuellement.
    Sauf --dry-run/--no-pr : commit + push + gh pr create jusqu'au bout.
    Retourne True si la mise à jour (et la PR, sauf --no-pr) a été effectuée.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    if run_id:
        try:
            client.set_tag(run_id, "export_to_prod", "true")
            run = client.get_run(run_id)
        except Exception as exc:
            logger.error("Run '%s' introuvable ou erreur MLflow : %s", run_id, exc)
            return False
    else:
        run = _find_export_run(client)

    if run is None:
        logger.info("Aucun blueprint à extraire — config/model_params.yml inchangé")
        return False

    algo = run.data.params.get("algorithm")
    if algo not in KNOWN_ALGOS:
        logger.warning("Algorithme '%s' inconnu — extraction ignorée", algo)
        return False

    extracted: dict[str, object] = {
        k: _coerce(v) for k, v in run.data.params.items() if k not in _META_PARAMS
    }

    logger.info(
        "Blueprint extrait — algo=%s run_id=%s params=%s",
        algo, run.info.run_id[:8], extracted,
    )

    if dry_run:
        logger.info("[dry-run] config/model_params.yml non modifié, pas de git/PR")
        return True

    # Charger le yaml existant et mettre à jour uniquement l'algo concerné
    current: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            current = yaml.safe_load(f) or {}

    current[algo] = extracted
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(current, f, default_flow_style=False, allow_unicode=True)

    logger.info("config/model_params.yml mis à jour pour algo=%s", algo)

    if no_pr:
        logger.info("--no-pr : pas de commit/push/PR — à faire toi-même")
        return True

    exp_metrics = {k: float(v) for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}
    kpi_gate_passed = run.data.tags.get("kpi_gate") == "PASSED"
    exp_model_name = run.data.tags.get("model_name", algo)
    comp = compute_comparison(exp_metrics, kpi_gate_passed, client, exp_model_name)

    return _commit_push_create_pr(algo, run.info.run_id, comp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DS champion blueprint to config/model_params.yml")
    parser.add_argument("run_id", nargs="?", default=None,
                        help="Run à promouvoir — tague export_to_prod=true et extrait directement. "
                             "Omis : cherche le dernier run déjà tagué manuellement dans l'UI.")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans écrire ni committer")
    parser.add_argument("--no-pr", action="store_true", help="Écrit le YAML mais s'arrête là (pas de git/PR)")
    args = parser.parse_args()
    success = extract_blueprint(run_id=args.run_id, dry_run=args.dry_run, no_pr=args.no_pr)
    exit(0 if success else 1)
