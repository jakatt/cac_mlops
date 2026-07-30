"""Secrets tournables stockés sur S3 (cac-mlops-data/secrets/) plutôt que figés
dans l'environnement du conteneur prefect-worker à sa création.

Pourquoi : prefect-worker ne peut pas se recréer lui-même pour appliquer un
changement de .env/docker-compose.yml (la tâche qui le ferait tourne dans le
conteneur qu'elle recréerait, tuant le flow en cours) — cf. fetch_gh_pat
(src/utils/github.py, PR#185), premier cas traité de ce risque. Lire un secret
depuis S3 à chaque exécution rend sa rotation indépendante du cycle de vie du
conteneur : un nouvel upload S3 suffit, jamais de redémarrage.

Généralisé le 2026-07-30 pour TAILSCALE_AUTHKEY / CADDY_S3_ACCESS_KEY_ID /
CADDY_S3_SECRET_ACCESS_KEY (kapsule_up_flow.py), identifiés comme même risque
latent (cf. project_infra_secrets_todo).
"""
import os


def fetch_secret(key: str, log) -> str | None:
    """Récupère s3://cac-mlops-data/secrets/<key>. Ne lève jamais — retourne
    None si indisponible, à l'appelant de décider du fallback (os.getenv)."""
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url="https://s3.fr-par.scw.cloud",
            aws_access_key_id=os.environ["SCW_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["SCW_SECRET_ACCESS_KEY"],
        )
        obj = s3.get_object(Bucket="cac-mlops-data", Key=f"secrets/{key}")
        return obj["Body"].read().decode().strip()
    except Exception as exc:
        log.warning("Impossible de récupérer le secret '%s' depuis S3 : %s", key, exc)
        return None
