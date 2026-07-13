"""
Deploy Kapsule flow — rolling update de api/gradio-public/nginx/caddy sur le cluster K8s.

Vérifie d'abord si Kapsule est actif (state/kapsule_ips non vide).
Si inactif : skip silencieux.
Si actif : kubectl apply -f k8s/ (resynchro manifests) → ménage léger
(pods Completed/Failed + log des conditions de pression nœuds) → rollout
restart SÉQUENTIEL (un deployment à la fois, pas les 3 en parallèle — cf.
rolling_update_task) + rollout status. Rollback auto si échec
(event=alert severity=critical topic=kapsule_failure — alerte Grafana).
"""
import os
import subprocess
import tempfile
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.flows.kapsule_up_flow import apply_manifests

CLUSTER_ID    = os.getenv("KAPSULE_CLUSTER_ID", "")
KAPSULE_STATE = Path(os.getenv("KAPSULE_STATE", "/app/state/kapsule_ips"))
K8S_NAMESPACE = "cac-mlops"

DEPLOYMENTS = ["api", "gradio-public", "nginx", "caddy"]


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


@task(name="kubectl-rolling-update", retries=1, retry_delay_seconds=30)
def rolling_update_task(kubeconfig: str) -> tuple[bool, list[str]]:
    """
    Rollout restart SÉQUENTIEL (un deployment à la fois : restart puis
    attente avant de passer au suivant) — pas les 3 en parallèle. Un rollout
    parallèle fait temporairement coexister l'ancien ET le nouveau pod pour
    api + gradio + gradio-public en même temps, ce qui double quasiment la
    charge instantanée sur des nœuds déjà petits (BASIC3-X2C-8G ×2) et a
    fait timeout le rollout `api` à 300s alors qu'un retry séquentiel juste
    après a réussi en 3m26s sans aucun changement de code (incident vécu,
    2026-07-11). Le séquentiel réduit le pic de charge et échoue/alerte plus
    vite (sur le premier deployment bloqué, pas après avoir attendu les 3).

    Returns (ok, touched) — touched liste les deployments dont le `rollout
    restart` a réellement démarré (donc les seuls à rollback en cas
    d'échec). Avec le séquentiel, un échec sur `api` (1er de la liste)
    signifie que gradio/gradio-public n'ont jamais été touchés — les
    inclure quand même dans le rollback fait échouer `kubectl rollout undo`
    avec "no rollout history found" (aucune révision précédente puisque
    jamais redémarrés), une erreur trompeuse mais sans conséquence
    (bug vécu, 2026-07-12).

    `kubectl set image` avec la même chaîne (toujours ":latest", jamais de
    tag par SHA) ne produit AUCUN diff de spec pour Kubernetes — donc
    AUCUN rollout, même si le contenu réel de l'image a changé sur le
    registre (bug vécu, jamais détecté avant : confirmé par
    `rollout history` inchangé après un `set image` réel, 2026-07-10).
    `rollout restart` force toujours une nouvelle ReplicaSet (patch d'une
    annotation de redémarrage), donc un vrai repull de l'image ET un
    re-run de l'initContainer fetch-model — nécessaire aussi bien pour
    Trigger 1 (nouveau modèle promu, jamais rechargé par un pod déjà
    tournant) que Trigger 2 (nouveau code).
    """
    log = get_run_logger()
    touched: list[str] = []
    try:
        for deploy_name in DEPLOYMENTS:
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

        log.info("Rolling update Kapsule OK")
        return True, touched

    except Exception as exc:
        # Exception large et pas seulement RuntimeError : _kubectl utilise
        # subprocess.run(timeout=300), qui lève subprocess.TimeoutExpired
        # (pas une RuntimeError) si le rollout ne converge pas dans les
        # temps. Avant ce fix, ce cas précis échappait entièrement au except
        # ci-dessus, crashait le flow sans jamais appeler rollback_kapsule_task
        # — donc AUCUN rollback n'était tenté malgré ce que dit le message
        # d'erreur de deploy_kapsule_flow (incident vécu, 2026-07-10 :
        # DiskPressure sur les 2 nœuds a fait dépasser les 300s deux fois de
        # suite, le rollback annoncé n'a jamais eu lieu).
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
def deploy_kapsule_flow() -> bool:
    """
    Rolling update sur Kapsule si le cluster est actif.
    Entièrement automatique — pas de gate manuelle.
    """
    log = get_run_logger()

    active = check_kapsule_task()
    if not active:
        log.info("Kapsule inactif — deploy Kapsule ignoré")
        return True

    kubeconfig = get_kubeconfig_task()

    # Resynchronise les manifests k8s/ avant le rollout : `rollout restart` seul
    # ne fait que redémarrer les pods avec le spec DÉJÀ enregistré sur le cluster —
    # tout changement de manifest (env var, ressources...) fait depuis le dernier
    # kapsule-up restait invisible tant qu'on ne relançait pas kapsule-down/up ou
    # qu'on n'appliquait pas manuellement (bug vécu : COCKPIT_ENV ajouté à
    # k8s/gradio/deployment.yaml après le dernier kapsule-up, jamais propagé sur
    # plusieurs déploiements malgré des rollout restart réussis, 2026-07-11).
    apply_manifests(kubeconfig)
    cleanup_before_deploy_task(kubeconfig)

    ok, touched = rolling_update_task(kubeconfig)

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

    # Confirmation de succès : visible dans Loki/Grafana, pas d'email (cf. deploy_vps_flow.py).
    log.info("event=alert severity=info topic=kapsule_success")
    return True
