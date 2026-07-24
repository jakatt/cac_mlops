"""Revert d'un commit de merge blueprint (config/model_params.yml) sur main.

Mécanisme partagé entre deux appelants :
  - src.flows.deploy_vps_flow::revert_blueprint_task — échec test-api après
    promotion (rollback automatique, contexte Prefect).
  - services.gradio.app::cancel_run — STOP manuel au gate avant promotion
    (contexte Cockpit, pas de flow Prefect en cours à ce moment).

Même mécanisme jetable (clone git + PAT depuis S3) que
_dvc_push_and_git_commit dans etl_flow.py — réutilisé, pas réinventé.
"""
import os
import subprocess
import tempfile
from pathlib import Path

from src.utils.github import GITHUB_REPO, fetch_gh_pat


def revert_blueprint_on_main(sha_tag: str, reason: str, log) -> bool:
    """Revert -m 1 du commit de merge <sha_tag> sur main, push direct [skip ci].

    `log` : tout objet exposant .warning(msg, *args) — get_run_logger() (Prefect)
    ou un logging.Logger standard (Gradio) fonctionnent tous les deux.

    Retourne True si le revert a été poussé avec succès, False sinon (chaque
    échec est déjà loggué en warning — l'appelant n'a rien à re-logger).
    """
    if not sha_tag:
        log.warning("Pas de sha_tag — blueprint non reverté automatiquement sur main.")
        return False

    pat = fetch_gh_pat(log)
    if not pat:
        log.warning("GH_PAT indisponible — blueprint non reverté automatiquement sur main.")
        return False

    with tempfile.TemporaryDirectory(prefix="blueprint-revert-") as tmp:
        clone_dir = Path(tmp) / "repo"
        repo_url = f"https://oauth2:{pat}@github.com/{GITHUB_REPO}.git"

        # --depth 50 (pas 1) : il faut l'historique du commit à revert, pas
        # seulement le tip — cette étape peut tourner longtemps après le merge.
        r = subprocess.run(
            ["git", "clone", "--depth", "50", "--quiet", repo_url, str(clone_dir)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log.warning("git clone failed (revert blueprint) : %s", r.stderr.strip())
            return False

        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "prefect-worker",
            "GIT_AUTHOR_EMAIL": "ci@cac-mlops.fr",
            "GIT_COMMITTER_NAME": "prefect-worker",
            "GIT_COMMITTER_EMAIL": "ci@cac-mlops.fr",
        }

        # -m 1 : sha_tag est un commit de merge (mainline = main avant le merge).
        # --no-commit : on écrit nous-mêmes le message (avec [skip ci]) plutôt
        # que d'accepter celui auto-généré par --no-edit.
        r = subprocess.run(
            ["git", "revert", "-m", "1", "--no-commit", sha_tag],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning(
                "git revert failed (%s) — blueprint NON reverté automatiquement, "
                "intervention manuelle requise sur config/model_params.yml : %s",
                sha_tag, r.stderr.strip(),
            )
            return False

        # [skip ci] : sinon deploy.yml redétecterait ce changement de
        # config/model_params.yml et redéclencherait update-model-flow en
        # boucle pour rien — le modèle "reverté vers" est déjà celui qui tourne.
        commit_msg = f"revert(blueprint): {reason} — commit {sha_tag[:8]} reverté [skip ci]"
        r = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning("git commit failed (revert blueprint) : %s", r.stderr.strip())
            return False

        r = subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=clone_dir, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            log.warning("git push failed (revert blueprint) : %s", r.stderr.strip())
            return False

        log.warning(
            "event=rollback kind=blueprint_git sha=%s reason=%s — "
            "config/model_params.yml reverté sur main",
            sha_tag, reason,
        )
        return True
