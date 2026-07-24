"""Accès GitHub partagé — sans dépendance à Prefect.

Extrait de src/flows/etl_flow.py pour être importable depuis n'importe quel
service (Prefect ou non) : services/gradio/app.py a besoin du PAT pour le
revert blueprint sur STOP manuel (cancel_run), mais n'a pas et ne doit pas
avoir prefect en dépendance juste pour ça.
"""
import os

GITHUB_REPO = os.getenv("GITHUB_REPO", "jakatt/cac_mlops")


def fetch_gh_pat(log) -> str | None:
    """Récupère le PAT GitHub depuis S3 (secrets/gh_pat) plutôt qu'une variable
    d'environnement figée à la création du conteneur.

    Pourquoi : /app (image api, utilisée par prefect-worker) ne contient jamais
    .git — le Dockerfile ne fait que des COPY sélectifs — et prefect-worker ne
    peut pas se recréer lui-même pour appliquer un changement docker-compose.yml
    (la tâche qui le ferait tourne dans le conteneur qu'elle recréerait,
    tuant le flow en cours). Lire le PAT depuis S3 à chaque exécution rend le
    mécanisme indépendant du cycle de vie du conteneur et du pipeline de
    déploiement : rotation du PAT = un nouvel upload S3, jamais de redémarrage.

    `log` : tout objet exposant .warning(msg, *args) — get_run_logger() (Prefect)
    ou un logging.Logger standard (Gradio) fonctionnent tous les deux.
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
