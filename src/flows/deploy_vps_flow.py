"""
Deploy VPS flow — smoke test + gate manuelle + promote MLflow + compose up + test-api + deploy Kapsule.

git pull / docker compose pull (téléchargement des images, sans interruption) sont
gérés par le script SSH de deploy.yml (côté HOST) avant que ce flow soit déclenché.
`docker compose up -d` / les redémarrages ciblés (impact VPS) sont en revanche
exécutés par ce flow, après la gate manuelle — pour que la gate protège bien
toute interruption de service, sur les 3 triggers :
  smoke test → gate → promote (T1/T3) + compose up (T2/T3 avec code) → test-api → Kapsule (si OK)

Rollback si test-api KO : alias MLflow (T1/T3) et/ou images Docker :rollback (T2/T3 avec code),
les deux étant appliqués indépendamment si le run cumule modèle + code.
Kapsule n'est déclenché que si test-api OK.

Modes d'appel :
  1. Depuis GitHub Actions (changement code seul) : sha_tag + needs_build/restart_services, pas de champion.
  2. Depuis check-new-data-flow (nouvelle data) : champion + métriques affichés à la gate.
  3. Depuis update-model-flow (nouveau blueprint) : champion + métriques + éventuellement
     sha_tag/needs_build/restart_services si le merge inclut aussi du code.

Concurrence : rien n'empêche 2 déclencheurs indépendants (ex: cron T1 lundi 8h
+ push code T2) d'avoir chacun leur run en pause à la gate en même temps. Un
verrou fichier (flock, DEPLOY_LOCK_PATH) sérialise la section post-gate qui
interrompt réellement le VPS, pour éviter 2 `docker compose up`/`restart`
concurrents sur les mêmes conteneurs — voir acquire_deploy_lock_task.
"""
import fcntl
import os
import time
from pathlib import Path

from prefect import flow, task, get_run_logger, pause_flow_run

from src.flows.deploy_kapsule_flow import deploy_kapsule_flow
from src.flows.test_api_flow import test_api_flow
from src.flows.train_flow import promote_task

NGINX_URL = os.getenv("NGINX_URL", "http://nginx:80")

# Verrou local (flock, pas de dépendance externe) sur la section qui interrompt
# le VPS (promote+restart / compose up). Rien n'empêchait jusqu'ici 2 runs de
# ce flow (ex: check-new-data-flow un lundi 8h + un push code) d'avoir chacun
# leur propre gate en pause simultanément puis d'être validés l'un après
# l'autre à quelques secondes d'intervalle — sans ce verrou, leurs `docker
# compose up -d` / `docker restart` respectifs s'exécuteraient en parallèle
# sur les mêmes conteneurs, sans aucune garantie d'atomicité côté Docker.
DEPLOY_LOCK_PATH = Path(os.getenv("DEPLOY_LOCK_PATH", "/app/state/deploy.lock"))


@task(name="acquire-deploy-lock")
def acquire_deploy_lock_task() -> int:
    """Verrou exclusif non bloquant — échoue vite et clairement plutôt que de
    laisser un 2e run attendre indéfiniment (jusqu'à 300s de timeout kubectl
    plus loin dans la chaîne rendrait l'attente confuse à diagnostiquer)."""
    log = get_run_logger()
    DEPLOY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(DEPLOY_LOCK_PATH, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(
            "Un autre déploiement VPS est déjà en cours (verrou deploy.lock actif).\n"
            "Cause probable : deux triggers (T1/T2/T3) ont franchi leur gate "
            "quasi simultanément.\n"
            "Actions requises :\n"
            "  1. Attendre la fin du run en cours (Prefect UI → Flow Runs)\n"
            "  2. Relancer ce run ensuite (Retry) — rien n'a encore été appliqué"
        )
    log.info("Verrou deploy.lock acquis")
    return fd


@task(name="release-deploy-lock")
def release_deploy_lock_task(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    get_run_logger().info("Verrou deploy.lock relâché")

# Seuls services dont l'image est reconstruite par deploy.yml. `docker compose
# up -d` doit TOUJOURS lister ces noms explicitement, jamais tourner sans
# argument sur tout le fichier — sinon un simple changement de config dans
# docker-compose.yml (env var, volume…) sur un AUTRE service (ex. prefect-worker)
# le fait recréer aussi. Si ce service est celui qui exécute la commande, il se
# tue en plein milieu : docker compose up -d s'interrompt, la stack entière
# reste à moitié recréée (incident vécu — PR #116, retrait d'env vars mortes
# de prefect-worker a suffi à déclencher ça).
MANAGED_SERVICES = ["api", "gradio", "gradio-public", "mlflow"]


def _trigger_label(champion: str | None, year: int | None, sha_tag: str) -> str:
    """T1 = nouvelles données (champion+year, pas de code) · T2 = code seul ·
    T3 = nouveau blueprint (champion+year+code) — même convention que le Cockpit."""
    if champion and year and sha_tag:
        return "T3"
    if champion and year:
        return "T1"
    return "T2"


def _format_gate_message(
    trigger: str,
    sha_tag: str,
    needs_build: bool,
    restart_services: str,
    champion: str | None,
    metrics: dict | None,
    year: int | None,
) -> str:
    """Message structuré affiché dans Prefect UI / Cockpit avant la gate manuelle."""
    CHK, DASH = "✓", "—"
    INTERRUPT = {
        "api": "~30 s", "gradio": "~5 s", "gradio-public": "~5 s",
        "mlflow": "~1 s", "nginx": "~1 s",
    }
    SVC_ORDER = ["api", "mlflow", "gradio", "gradio-public",
                 "nginx", "grafana", "prometheus", "loki", "promtail"]

    rebuilt   = set(MANAGED_SERVICES) if needs_build else set()
    restarted = rebuilt | {s for s in restart_services.split(",") if s}
    impacted  = [s for s in SVC_ORDER if s in restarted]

    if trigger == "T1":
        header = f"VALIDATION REQUISE — Nouvelles données (T1) · Données : {year}"
    elif trigger == "T2":
        header = f"VALIDATION REQUISE — Nouveau code (T2) · SHA : {sha_tag or 'N/A'}"
    else:
        header = f"VALIDATION REQUISE — Nouveau blueprint (T3) · SHA : {sha_tag or 'N/A'} · Données : {year}"

    W   = max(len(header) + 4, 56)
    sep = "═" * W
    lines = [f"╔{sep}╗", f"║  {header:<{W - 2}}║", f"╚{sep}╝", ""]

    if champion and metrics:
        m = (metrics or {}).get(champion, {})
        lines += [
            f"  Champion : {champion:<10}"
            f"  F1 : {m.get('f1', 0):.4f}"
            f"  Recall : {m.get('recall', 0):.4f}"
            f"  AUC : {m.get('auc', 0):.4f}",
            "",
        ]

    if champion:
        lines += ["  Promotion @Production → redémarrage API (~30 s) après GO", ""]

    if impacted:
        C = 16
        lines += [
            f"  {'Service':<{C}}│ Rebuild │ Restart │ Interruption",
            f"  {'─' * C}┼─────────┼─────────┼──────────────",
        ]
        for svc in impacted:
            rb = CHK if svc in rebuilt  else DASH
            rs = CHK if svc in restarted else DASH
            it = INTERRUPT.get(svc, "~1 s") if svc in restarted else DASH
            lines.append(f"  {svc:<{C}}│  {rb:<7}│  {rs:<7}│  {it}")
        lines.append("")
    elif not champion:
        lines += ["  Aucun service impacté.", ""]

    if trigger == "T1":
        lines += [
            "  → GO   : Promouvoir @Production → test-api → Kapsule",
            "  → STOP : run annulé, @Production inchangé, aucun impact",
        ]
    elif trigger == "T2":
        if impacted:
            lines += [
                "  → GO   : Rebuild/restart des services ci-dessus → test-api → Kapsule",
                "  → STOP : run annulé, aucun conteneur touché",
            ]
        else:
            lines += [
                "  → GO   : Valider (sources déjà actives via volumes) → test-api → Kapsule",
                "  → STOP : run annulé, aucun conteneur touché",
            ]
    else:
        if impacted:
            lines += [
                "  → GO   : Promouvoir @Production + rebuild/restart services → test-api → Kapsule",
                "  → STOP : run annulé, aucun impact",
            ]
        else:
            lines += [
                "  → GO   : Promouvoir @Production → test-api → Kapsule",
                "  → STOP : run annulé, aucun impact",
            ]

    return "\n".join(lines)


@task(name="smoke-test-health")
def smoke_test_task(max_wait_s: int = 90) -> bool:
    log = get_run_logger()
    import urllib.request
    for i in range(max_wait_s // 5):
        try:
            urllib.request.urlopen(f"{NGINX_URL}/health", timeout=5)
            log.info("Smoke test OK après %ds", (i + 1) * 5)
            return True
        except Exception:
            log.info("  attente smoke test… (%ds)", (i + 1) * 5)
            time.sleep(5)
    log.error("Smoke test échoué après %ds", max_wait_s)
    return False


@task(name="restart-api")
def restart_api_task() -> None:
    log = get_run_logger()
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "cac_mlops-api-1"})
        if not containers:
            log.warning("Conteneur API introuvable — restart ignoré")
            return
        containers[0].restart(timeout=30)
        log.info("API redémarrée — attente healthcheck...")
    except Exception as exc:
        log.warning("Docker restart API échoué (%s) — continuation", exc)
        return

    import requests as _req
    api_url = os.getenv("API_URL", "http://api:8000")
    for _ in range(12):
        try:
            if _req.get(f"{api_url}/health", timeout=3).status_code == 200:
                log.info("API prête")
                return
        except Exception:
            pass
        time.sleep(5)
    log.warning("API healthcheck timeout — continuation")


@task(name="get-current-production")
def get_current_production_task(champion: str) -> dict | None:
    """Sauvegarde @Production avant promote pour rollback si test-api KO."""
    import mlflow
    from src.models.train_model import MODEL_NAMES
    client = mlflow.tracking.MlflowClient()
    for model_name in MODEL_NAMES.values():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            return {"model_name": model_name, "version": mv.version}
        except Exception:
            continue
    return None


@task(name="compose-up")
def compose_up_task(needs_build: bool, restart_services: str) -> None:
    """Applique l'interruption VPS après la gate : up -d (si nouvelles images) + restarts ciblés.

    Les images ont déjà été pull-ées côté SSH (deploy.yml) avant la gate — cette étape
    ne fait qu'appliquer le changement (recréation des conteneurs), donc c'est bien ici
    que doit se situer la seule interruption de service du trigger 2/3.
    """
    import subprocess
    log = get_run_logger()
    compose_file = "/app/docker-compose.yml"
    project_dir = os.getenv("COMPOSE_PROJECT_DIR", "/home/deploy/cac_mlops")

    t0 = time.monotonic()
    log.info(
        "event=interruption_start kind=compose_up needs_build=%s services=%s",
        needs_build, restart_services or "-",
    )

    if needs_build:
        try:
            subprocess.run(
                ["docker", "compose", "-f", compose_file, "--project-directory", project_dir,
                 "up", "-d", "--remove-orphans", *MANAGED_SERVICES],
                check=True, capture_output=True, text=True,
            )
            log.info("docker compose up -d --remove-orphans OK (%s)", ", ".join(MANAGED_SERVICES))
        except subprocess.CalledProcessError as e:
            log.warning(
                "event=interruption_end kind=compose_up status=fail duration_s=%.1f",
                time.monotonic() - t0,
            )
            raise RuntimeError(f"docker compose up -d échoué:\n{e.stderr}")

    for service in filter(None, restart_services.split(",")):
        if service == "prefect-worker":
            # Se redémarrer soi-même tuerait ce processus avant qu'il ait pu
            # continuer (test-api, Kapsule, alerte finale) — run orphelin en
            # RUNNING pour toujours. Inutile de toute façon : le work pool
            # "process" lance chaque flow run dans un sous-processus neuf,
            # le code bind-monté est déjà rechargé sans restart.
            log.warning("Restart de prefect-worker ignoré (auto-référence dangereuse depuis ce flow)")
            continue
        try:
            subprocess.run(
                ["docker", "compose", "-f", compose_file, "--project-directory", project_dir,
                 "restart", service],
                check=True, capture_output=True, text=True,
            )
            log.info("Restart ciblé OK : %s", service)
        except subprocess.CalledProcessError as e:
            log.warning("Restart échoué pour %s — ignoré : %s", service, e.stderr.strip())

    log.info(
        "event=interruption_end kind=compose_up status=ok duration_s=%.1f",
        time.monotonic() - t0,
    )


@task(name="docker-rollback")
def docker_rollback_task(sha_tag: str = "") -> None:
    """Restaure les images :rollback + recrée les conteneurs (Trigger 2 — code seul)."""
    import subprocess
    log = get_run_logger()
    log.warning("event=rollback kind=docker_image sha=%s", sha_tag or "N/A")
    ghcr_user = os.getenv("GHCR_USER", "jakatt")
    registry = f"ghcr.io/{ghcr_user}"
    images = ["cac-mlops-api", "cac-mlops-mlflow", "cac-mlops-gradio"]
    compose_file = "/app/docker-compose.yml"
    project_dir = os.getenv("COMPOSE_PROJECT_DIR", "/home/deploy/cac_mlops")
    for img in images:
        try:
            subprocess.run(
                ["docker", "tag", f"{registry}/{img}:rollback", f"{registry}/{img}:latest"],
                check=True, capture_output=True, text=True,
            )
            log.info("Retaggé %s:rollback → :latest", img)
        except subprocess.CalledProcessError as e:
            log.warning("Tag échoué pour %s — ignoré : %s", img, e.stderr.strip())
    try:
        subprocess.run(
            ["docker", "compose", "-f", compose_file, "--project-directory", project_dir,
             "up", "-d", *MANAGED_SERVICES],
            check=True, capture_output=True, text=True,
        )
        log.info("Docker rollback OK — conteneurs recréés avec :rollback (SHA annulé : %s)", sha_tag or "N/A")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"docker compose up -d rollback échoué:\n{e.stderr}")


@task(name="rollback-promote")
def rollback_promote_task(previous: dict | None, champion: str) -> None:
    """Restaure @Production vers la version précédente après échec test-api."""
    import mlflow
    from src.models.train_model import MODEL_NAMES
    log = get_run_logger()
    log.warning("event=rollback kind=model_alias champion=%s", champion)
    client = mlflow.tracking.MlflowClient()
    new_model_name = MODEL_NAMES[champion]
    try:
        client.delete_registered_model_alias(new_model_name, "Production")
        log.info("@Production supprimé de %s", new_model_name)
    except Exception as exc:
        log.warning("Suppression alias échouée (%s)", exc)
    if previous:
        client.set_registered_model_alias(
            previous["model_name"], "Production", previous["version"]
        )
        log.info("@Production restauré → %s v%s", previous["model_name"], previous["version"])
    else:
        log.warning("Aucun @Production précédent à restaurer")


@flow(name="deploy-vps-flow", log_prints=True)
def deploy_vps_flow(
    champion: str | None = None,
    run_ids: dict | None = None,
    metrics: dict | None = None,
    year: int | None = None,
    sha_tag: str = "",
    needs_build: bool = False,
    restart_services: str = "",
) -> bool:
    """
    smoke test → gate manuelle → promote @Production + compose up (selon trigger)
    → test-api (validation finale) → Kapsule (seulement si test-api OK).

    Si test-api KO → rollback indépendant de chaque changement appliqué après la gate :
    alias MLflow restauré si un modèle a été promu, images :rollback si compose up a tourné
    (un run peut cumuler les deux — merge blueprint + code). Pas de Kapsule dans tous les cas.

    champion / run_ids / metrics / year : renseignés par check-new-data-flow / update-model-flow.
    sha_tag : SHA du commit buildé, passé par GitHub Actions.
    needs_build / restart_services : calculés côté SSH (deploy.yml) à partir du diff git,
    appliqués ici (après la gate) plutôt que dans le script SSH (avant la gate).
    """
    log = get_run_logger()

    # ── 1. Smoke test ─────────────────────────────────────────────────────────
    ok = smoke_test_task()
    if not ok:
        log.error(
            "event=alert severity=critical topic=deploy_failure reason=smoke_test_pre_gate sha=%s",
            sha_tag or "N/A",
        )
        raise RuntimeError(
            f"Smoke test ÉCHOUÉ — {NGINX_URL}/health ne répond pas après 90s.\n"
            f"SHA déployé : {sha_tag or 'N/A'}\n"
            "Actions requises :\n"
            "  1. docker compose logs api nginx --tail=100\n"
            "  2. Vérifier que l'API charge le modèle MLflow au démarrage\n"
            "  3. Si image corrompue : docker compose pull && docker compose up -d\n"
            "  4. Si MLflow inaccessible : vérifier healthcheck mlflow + minio"
        )

    # ── 2. Gate manuelle ──────────────────────────────────────────────────────
    trigger = _trigger_label(champion, year, sha_tag)
    log.info("\n%s", _format_gate_message(
        trigger, sha_tag, needs_build, restart_services, champion, metrics, year,
    ))
    log.info(
        "event=gate_open trigger=%s sha=%s champion=%s needs_build=%s restart_services=%s",
        trigger, sha_tag or "-", champion or "-", needs_build, restart_services or "-",
    )

    pause_flow_run(timeout=86400)

    # Le process ne reprend ICI qu'après un resume (GO) explicite — un cancel
    # (STOP, depuis le Cockpit) termine ce flow run avant d'atteindre cette ligne.
    # Le STOP est donc loggé côté Cockpit (service=gradio), pas ici.
    log.info("event=gate_resolved decision=GO trigger=%s sha=%s", trigger, sha_tag or "-")

    # Verrou acquis seulement APRÈS la gate : deux runs peuvent rester en pause
    # simultanément sans problème (aucune ressource touchée), seule la section
    # qui interrompt réellement le VPS doit être sérialisée.
    lock_fd = acquire_deploy_lock_task()
    try:
        # ── 3. Promote MLflow (Triggers 1 & 3 — nouveau modèle) ─────────────────
        previous_production: dict | None = None
        if champion and run_ids:
            previous_production = get_current_production_task(champion)
            log.info("Promotion @Production → %s", champion)
            promote_task(champion, run_ids)
            restart_api_task()
            log.info("API redémarrée avec le nouveau modèle @Production")

        # ── 3bis. Compose up (Triggers 2 & 3 — changement de code) ─────────────
        # Seule étape qui interrompt le VPS pour le code — désormais après la gate.
        if needs_build or restart_services:
            compose_up_task(needs_build, restart_services)
            ok = smoke_test_task()
            if not ok:
                log.error(
                    "event=alert severity=critical topic=deploy_failure reason=smoke_test_post_compose sha=%s",
                    sha_tag or "N/A",
                )
                docker_rollback_task(sha_tag)
                raise RuntimeError(
                    f"Smoke test ÉCHOUÉ après compose up — {NGINX_URL}/health ne répond pas après 90s.\n"
                    f"SHA déployé : {sha_tag or 'N/A'}\n"
                    "Rollback Docker :rollback effectué — conteneurs recréés."
                )

        # ── 4. Test-api — validation finale en production ───────────────────────
        try:
            test_api_flow(skip_rate_limit=True)
            log.info("test-api OK ✓")
        except Exception as exc:
            log.error("test-api ÉCHOUÉ : %s", exc)
            rolled_back_model = False
            rolled_back_code = False
            if champion and run_ids:
                log.info("Rollback promote @Production...")
                rollback_promote_task(previous_production, champion)
                restart_api_task()
                rolled_back_model = True
            if needs_build or restart_services:
                log.info("Rollback Docker images :rollback (code)...")
                docker_rollback_task(sha_tag)
                rolled_back_code = True
            log.error(
                "event=alert severity=critical topic=deploy_failure reason=test_api sha=%s "
                "rolled_back_model=%s rolled_back_code=%s",
                sha_tag or "N/A", rolled_back_model, rolled_back_code,
            )
            raise RuntimeError(
                f"Test-api ÉCHOUÉ — les tests fonctionnels sont KO après le deploy.\n"
                f"SHA déployé : {sha_tag or 'N/A'}\n"
                f"Erreur : {exc}\n"
                + ("@Production rollback effectué — l'ancienne version est restaurée.\n" if rolled_back_model else "")
                + ("Rollback Docker effectué — images :rollback recréées.\n" if rolled_back_code else "")
                + "Actions requises :\n"
                "  1. docker compose logs api --tail=100\n"
                "  2. Vérifier que le modèle @Production est accessible dans MLflow\n"
                "  3. Tester manuellement : curl -X POST http://VPS:8080/predict\n"
                "  4. Si rollback insuffisant : reset-flow puis full-retrain"
            )

        # ── 5. Deploy Kapsule (seulement si test-api OK) ─────────────────────────
        deploy_kapsule_flow()
    finally:
        release_deploy_lock_task(lock_fd)

    # Confirmation de succès : visible dans Loki/Grafana (Cockpit, dashboard
    # Résilience), mais pas d'email — un succès n'est pas une alerte. Seuls les
    # topics ci-dessus (severity=critical) déclenchent une notification.
    log.info(
        "event=alert severity=info topic=deploy_success sha=%s champion=%s",
        sha_tag or "N/A", champion or "-",
    )
    return True
