"""
Deploy Kapsule flow — rolling update conditionnel sur le cluster K8s.

Vérifie d'abord si Kapsule est actif (state/kapsule_ips non vide).
Si inactif : skip silencieux.

Paramètres de contexte passés par deploy_vps_flow :
  new_model  : nouveau @Production promu → upload_model_s3 + restart api
  new_data   : nouveau dataset → upload_data_s3 (+ restart api si new_model aussi)
  new_images : nouvelles images buildées (CI rebuild) → restart api + gradio-public

Déploiements redémarrés :
  api           si new_images ou new_model  (initContainer refetch model depuis S3)
  gradio-public si new_images
  nginx/caddy   jamais explicitement — apply_manifests gère les changements de spec

Si aucun restart nécessaire (docs/config VPS uniquement) : apply_manifests +
cleanup défensif puis return sans aucun rollout restart.

Rollback auto si rollout restart échoue
(event=alert severity=critical topic=kapsule_failure — alerte Grafana).
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.flows.kapsule_up_flow import apply_manifests, upload_data_s3, upload_model_s3

CLUSTER_ID    = os.getenv("KAPSULE_CLUSTER_ID", "")
KAPSULE_STATE = Path(os.getenv("KAPSULE_STATE", "/app/state/kapsule_ips"))
K8S_NAMESPACE = "cac-mlops"

_ALL_KAPSULE_DEPLOYMENTS = ["api", "gradio-public", "nginx", "caddy"]


def _scw(args: list[str], timeout: int = 60) -> str:
    env = os.environ.copy()
    env["SCW_ACCESS_KEY"] = env.get("SCW_ACCESS_KEY_ID", "")
    env["SCW_SECRET_KEY"] = env.get("SCW_SECRET_ACCESS_KEY", "")
    r = subprocess.run(["scw"] + args, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scw {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def _kubectl(kubeconfig: str, args: list[str], check: bool = True) -> str:
    r = subprocess.run(
        ["kubectl", f"--kubeconfig={kubeconfig}"] + args,
        capture_output=True, text=True, timeout=300,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout + r.stderr


@task(name="check-kapsule-active")
def check_kapsule_task() -> bool:
    """Return True if Kapsule cluster is running (state file non-empty)."""
    log = get_run_logger()
    if not KAPSULE_STATE.exists():
        log.info("state/kapsule_ips absent — Kapsule inactif, skip")
        return False
    content = KAPSULE_STATE.read_text().strip()
    if not content:
        log.info("state/kapsule_ips vide — Kapsule inactif, skip")
        return False
    log.info("Kapsule actif — IPs: %s", content[:100])
    return True


@task(name="get-kubeconfig-deploy")
def get_kubeconfig_task() -> str:
    """Fetch kubeconfig for the Kapsule cluster via scw CLI."""
    log = get_run_logger()
    if not CLUSTER_ID:
        raise RuntimeError("KAPSULE_CLUSTER_ID non configuré")
    log.info("Récupération kubeconfig cluster %s", CLUSTER_ID)
    content = _scw(["k8s", "kubeconfig", "get", CLUSTER_ID])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
    return f.name


@task(name="pre-deploy-cleanup")
def cleanup_before_deploy_task(kubeconfig: str) -> None:
    """Ménage léger avant le rollout — purement défensif, n'échoue jamais
    (check=False partout) pour ne jamais bloquer le déploiement sur un
    problème de nettoyage :

    1. Supprime les pods Completed/Failed qui traînent dans le namespace
       (observé en direct : un vieux pod grafana resté en 0/1 Completed
       après un rollout précédent — ne consomme pas de CPU/RAM mais
       encombre `kubectl get pods` et peut laisser des logs/fs de conteneur
       sur le disque du nœud tant que le GC kubelet ne passe pas).
    2. Log les conditions DiskPressure/MemoryPressure/PIDPressure de chaque
       nœud — informatif seulement (pas de blocage), pour avoir un signal
       clair dans Grafana/Loki AVANT un éventuel échec de rollout plutôt
       que de le découvrir seulement après un timeout de 300s (incident
       vécu : DiskPressure sur les 2 nœuds, 2026-07-10).
    """
    import json
    log = get_run_logger()

    for phase in ("Succeeded", "Failed"):
        out = _kubectl(kubeconfig, [
            "delete", "pods", "-n", K8S_NAMESPACE,
            f"--field-selector=status.phase={phase}",
            "--ignore-not-found",
        ], check=False)
        if out.strip() and "No resources found" not in out:
            log.info("Nettoyage pods %s : %s", phase, out.strip())

    raw = _kubectl(kubeconfig, ["get", "nodes", "-o", "json"], check=False)
    try:
        nodes = json.loads(raw).get("items", [])
    except Exception:
        nodes = []
    for node in nodes:
        name = node.get("metadata", {}).get("name", "?")
        conds = {c["type"]: c["status"] for c in node.get("status", {}).get("conditions", [])}
        pressure = {
            k: v for k, v in conds.items()
            if k in ("DiskPressure", "MemoryPressure", "PIDPressure") and v == "True"
        }
        if pressure:
            log.warning(
                "event=alert severity=warning topic=kapsule_node_pressure node=%s pressure=%s",
                name, pressure,
            )
        else:
            log.info("Noeud %s OK (pas de pression ressources)", name)


def _pod_node_map(kubeconfig: str, deploy_name: str) -> dict[str, str]:
    out = _kubectl(kubeconfig, [
        "get", "pods", "-n", K8S_NAMESPACE,
        "-l", f"app={deploy_name}",
        "-o", "jsonpath={range .items[*]}{.metadata.name} {.spec.nodeName}\n{end}",
    ], check=False)
    mapping: dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    return mapping


@task(name="rebalance-topology")
def rebalance_topology_task(kubeconfig: str, deploy_name: str, max_attempts: int = 3) -> None:
    """topologySpreadConstraints (ScheduleAnyway, maxSkew=1 — cf. k8s/*/deployment.yaml)
    n'est qu'une préférence de scoring, pas une garantie : un `rollout restart`
    séquentiel (remplace un pod à la fois) peut quand même aboutir aux 2
    répliques sur le même nœud, le scheduler évaluant un état transitoire
    (constaté en direct, 2026-07-14 — reproduit dès le tout premier vrai
    déploiement après l'ajout de la contrainte, sur api/caddy/nginx). Purement
    défensif (check=False partout, n'échoue jamais) : supprime UNE réplique
    co-localisée pour forcer un reschedule qui, lui, respecte correctement la
    contrainte (validé en direct : un scale 0→N la respecte systématiquement,
    contrairement au rollout restart).

    Boucle de retry (max_attempts) : sur `api` (seul des 4 avec un HPA actif),
    un rééquilibrage réussi ici a été défait 2 fois de suite un peu plus tard
    dans le même cycle de déploiement (constaté en direct, 2026-07-14) — le
    controller de scale-down HPA choisit quel pod supprimer selon son propre
    algorithme (âge, etc.), pas topology-spread-aware, et peut donc redéfaire
    un équilibre déjà obtenu. Cette boucle seule ne suffit pas contre un
    scale-down qui arrive après le dernier appel — voir le passage de
    vérification final ajouté dans rolling_update_task."""
    log = get_run_logger()

    nodes_out = _kubectl(kubeconfig, ["get", "nodes", "-o", "name"], check=False)
    node_count = len([l for l in nodes_out.strip().splitlines() if l.strip()])
    if node_count < 2:
        return  # rien à répartir

    for attempt in range(1, max_attempts + 1):
        pod_nodes = _pod_node_map(kubeconfig, deploy_name)
        distinct_nodes = set(pod_nodes.values())
        if len(pod_nodes) < 2 or len(distinct_nodes) >= len(pod_nodes):
            return  # déjà réparti (ou pas assez de répliques pour que ça compte)

        pod_to_delete = next(iter(pod_nodes))
        log.warning(
            "event=alert severity=warning topic=kapsule_topology_skew deploy=%s pod=%s "
            "attempt=%s/%s — répliques co-localisées sur le même nœud, suppression "
            "pour forcer un reschedule réparti",
            deploy_name, pod_to_delete, attempt, max_attempts,
        )
        _kubectl(kubeconfig, ["delete", "pod", pod_to_delete, "-n", K8S_NAMESPACE, "--wait=false"], check=False)
        _kubectl(kubeconfig, [
            "wait", f"deployment/{deploy_name}", "-n", K8S_NAMESPACE,
            "--for=condition=available", "--timeout=60s",
        ], check=False)
        time.sleep(5)  # laisse le scheduler/HPA se stabiliser avant de revérifier

    log.warning(
        "event=alert severity=warning topic=kapsule_topology_skew_persistent deploy=%s "
        "— toujours co-localisé après %s tentatives, abandon (non bloquant)",
        deploy_name, max_attempts,
    )


@task(name="kubectl-rolling-update", retries=1, retry_delay_seconds=30)
def rolling_update_task(kubeconfig: str, deployments: list[str]) -> tuple[bool, list[str]]:
    """
    Rollout restart SÉQUENTIEL (un deployment à la fois : restart puis
    attente avant de passer au suivant). Un rollout parallèle fait
    temporairement coexister l'ancien ET le nouveau pod pour tous les
    deployments en même temps, ce qui double quasiment la charge
    instantanée sur des nœuds déjà petits (BASIC3-X2C-8G ×2) et a
    fait timeout le rollout `api` à 300s alors qu'un retry séquentiel
    juste après a réussi en 3m26s sans aucun changement de code
    (incident vécu, 2026-07-11). Le séquentiel réduit le pic de charge
    et échoue/alerte plus vite (sur le premier deployment bloqué).

    Returns (ok, touched) — touched liste les deployments dont le `rollout
    restart` a réellement démarré (donc les seuls à rollback en cas
    d'échec). Avec le séquentiel, un échec sur le 1er deployment signifie
    que les suivants n'ont jamais été touchés — les inclure dans le rollback
    ferait échouer `kubectl rollout undo` avec "no rollout history found"
    (aucune révision précédente puisque jamais redémarrés, bug vécu
    2026-07-12).

    `kubectl set image` avec la même chaîne (":latest") ne produit AUCUN
    diff de spec pour Kubernetes donc AUCUN rollout même si le contenu de
    l'image a changé sur le registre (bug vécu, confirmé par `rollout
    history` inchangé, 2026-07-10). `rollout restart` force toujours une
    nouvelle ReplicaSet (patch annotation), donc un vrai repull ET un
    re-run de l'initContainer fetch-model.

    Rééquilibrage topologie : max_attempts=1 pour les deployments sans HPA
    (nginx, caddy, gradio-public — un rééquilibrage réussi ne sera pas
    défait), max_attempts=3 pour api (HPA actif, peut redéfaire l'équilibre
    après un scale-down, bug vécu 2× en direct 2026-07-14). Passe finale
    uniquement sur api pour la même raison.
    """
    log = get_run_logger()
    touched: list[str] = []
    try:
        for deploy_name in deployments:
            log.info("Rolling restart %s", deploy_name)
            _kubectl(kubeconfig, [
                "rollout", "restart",
                f"deployment/{deploy_name}",
                "-n", K8S_NAMESPACE,
            ])
            touched.append(deploy_name)
            log.info("Attente rollout %s…", deploy_name)
            _kubectl(kubeconfig, [
                "rollout", "status",
                f"deployment/{deploy_name}",
                "-n", K8S_NAMESPACE,
                "--timeout=300s",
            ])
            rebalance_topology_task(
                kubeconfig, deploy_name,
                max_attempts=3 if deploy_name == "api" else 1,
            )

        # Passe finale uniquement sur api : seul deployment avec HPA actif.
        # Un scale-down HPA déclenché par le pic de charge du déploiement
        # complet peut défaire un rééquilibrage pourtant réussi en cours de
        # boucle — le temps écoulé à traiter les autres deployments laisse
        # au HPA l'occasion de se stabiliser avant cette vérification finale
        # (bug vécu 2× de suite, 2026-07-14).
        if "api" in deployments:
            rebalance_topology_task(kubeconfig, "api", max_attempts=3)

        log.info("Rolling update Kapsule OK")
        return True, touched

    except Exception as exc:
        # Exception large et pas seulement RuntimeError : _kubectl utilise
        # subprocess.run(timeout=300), qui lève subprocess.TimeoutExpired
        # (pas une RuntimeError) si le rollout ne converge pas dans les
        # temps. Avant ce fix, ce cas précis crashait le flow sans jamais
        # appeler rollback_kapsule_task (incident vécu, 2026-07-10 :
        # DiskPressure sur les 2 nœuds, le rollback annoncé n'a jamais eu lieu).
        log.error("Rolling update échoué : %s", exc)
        return False, touched


@task(name="kubectl-rollback")
def rollback_kapsule_task(kubeconfig: str, deployments: list[str]) -> None:
    log = get_run_logger()
    log.warning("event=rollback kind=kapsule targets=%s", ",".join(deployments) or "aucun")
    for deploy_name in deployments:
        log.info("Rollback %s", deploy_name)
        try:
            _kubectl(kubeconfig, [
                "rollout", "undo",
                f"deployment/{deploy_name}",
                "-n", K8S_NAMESPACE,
            ])
        except RuntimeError as exc:
            log.error("Rollback %s échoué : %s", deploy_name, exc)


@flow(name="deploy-kapsule-flow", log_prints=True)
def deploy_kapsule_flow(
    new_model: bool = False,
    new_data: bool = False,
    new_images: bool = True,
) -> bool:
    """
    Rolling update conditionnel sur Kapsule si le cluster est actif.
    Entièrement automatique — pas de gate manuelle.

    new_model  : nouveau @Production promu — upload_model_s3 + restart api
    new_data   : nouveau dataset — upload_data_s3
    new_images : nouvelles images buildées par CI — restart api + gradio-public

    Si aucun restart n'est nécessaire (new_model=False, new_images=False) :
    apply_manifests + cleanup défensif, aucun rollout restart.
    """
    log = get_run_logger()

    active = check_kapsule_task()
    if not active:
        log.info("Kapsule inactif — deploy Kapsule ignoré")
        return True

    kubeconfig = get_kubeconfig_task()

    # Uploads S3 conditionnels — n'uploader que ce qui a réellement changé.
    # Contexte : ces tasks n'étaient pas appelées au kapsule-up initial (bug
    # silencieux) — K8s continuait de servir l'ANCIEN modèle/dataset jusqu'au
    # prochain kapsule-down/up. Réexporter systématiquement quand pertinent est
    # inoffensif (upload S3 idempotent) mais uploader un modèle/dataset inchangé
    # sur un déploiement docs-only est inutile (bug vécu : PR #172, docs HTML
    # uniquement, upload_model_s3 et upload_data_s3 tournaient quand même).
    if new_model:
        upload_model_s3()
    if new_data:
        upload_data_s3()

    # Resynchronise les manifests k8s/ — idempotent, fast, toujours utile :
    # `rollout restart` seul redémarre les pods avec le spec DÉJÀ enregistré,
    # tout changement de manifest (env var, ressources, configmap...) depuis le
    # dernier kapsule-up restait invisible (bug vécu : COCKPIT_ENV ajouté à
    # k8s/gradio/deployment.yaml, jamais propagé sur plusieurs déploiements
    # malgré des rollout restart réussis, 2026-07-11). Pour nginx et caddy
    # (images standard), apply_manifests est le seul vecteur de propagation
    # d'un changement de spec — pas de rollout restart explicite nécessaire.
    apply_manifests(kubeconfig)
    cleanup_before_deploy_task(kubeconfig)

    # Calcul du périmètre de restart :
    #   api           si new_images (image CI reconstruite) OU new_model
    #                 (initContainer doit refetch le modèle depuis S3)
    #   gradio-public si new_images (image gradio CI reconstruite)
    #   nginx/caddy   jamais explicitement — leurs changements de spec sont
    #                 propagés par apply_manifests (qui déclenche un rolling
    #                 update automatique si la spec du Deployment change).
    to_restart: list[str] = []
    if new_images:
        to_restart.extend(["api", "gradio-public"])
    if new_model and "api" not in to_restart:
        to_restart.append("api")

    if not to_restart:
        log.info(
            "Aucun rollout restart nécessaire (new_model=%s new_images=%s) "
            "— manifests resynchronisés, skip rolling update",
            new_model, new_images,
        )
        log.info("event=alert severity=info topic=kapsule_success")
        return True

    log.info("Rollout restart : %s", ", ".join(to_restart))
    ok, touched = rolling_update_task(kubeconfig, to_restart)

    if not ok:
        rollback_kapsule_task(kubeconfig, touched)
        log.error("event=alert severity=critical topic=kapsule_failure")
        raise RuntimeError(
            "Deploy Kapsule ÉCHOUÉ — rolling update impossible sur le cluster K8s.\n"
            "Le rollback vers la version précédente a été effectué automatiquement.\n"
            "Actions requises :\n"
            "  1. kubectl get pods -n cac-mlops  (pods en erreur ?)\n"
            "  2. kubectl describe deployment api -n cac-mlops  (events, image pull error ?)\n"
            "  3. Vérifier que l'image GHCR est pullable depuis le cluster\n"
            "  4. Si cluster instable : kapsule-down puis kapsule-up"
        )

    log.info("event=alert severity=info topic=kapsule_success")
    return True
