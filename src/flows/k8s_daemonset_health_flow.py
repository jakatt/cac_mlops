"""
K8s DaemonSet health flow — vérifie les 4 composants K8s sans Service stable
ni serveur HTTP (node-exporter, promtail : vrais DaemonSets, 1 pod par nœud ;
loki-forwarder — proxy SOCKS5 — et tailscale-subnet-router — pas de serveur
HTTP — sont en réalité des Deployments 1 replica, cf. k8s/loki-forwarder/
deployment.yaml et k8s/tailscale/deployment.yaml). Inatteignables
directement depuis le Cockpit Gradio (pas de kubeconfig sur ce process —
seul prefect-worker en dispose), d'où ce flow dédié déclenché à la demande
depuis le Healthcheck.

Bug corrigé le 2026-07-29 : les 4 étaient interrogés via `kubectl get
daemonset`, ce qui renvoie NotFound (donc NOK) pour loki-forwarder/
tailscale-subnet-router puisque ce ne sont pas des DaemonSets — ils
apparaissaient NOK en permanence, peu importe leur état réel.

Imprime une ligne "DAEMONSET_STATUS <name>=OK|NOK" par composant, parsée
depuis les logs du run par le Cockpit (services/gradio/app.py::
_check_k8s_daemonsets) — plus simple à récupérer via l'API REST Prefect
que le résultat sérialisé d'un flow.
"""
from prefect import flow, get_run_logger, task

from src.flows.deploy_kapsule_flow import K8S_NAMESPACE, _kubectl, check_kapsule_task, get_kubeconfig_task

# (name, kind) — kind détermine la ressource kubectl et le jsonpath à interroger.
COMPONENTS = [
    ("node-exporter", "daemonset"),
    ("promtail", "daemonset"),
    ("loki-forwarder", "deployment"),
    ("tailscale-subnet-router", "deployment"),
]


@task(name="check-daemonset-ready")
def check_daemonset_task(kubeconfig: str, name: str, kind: str) -> bool:
    if kind == "daemonset":
        jsonpath = "{.status.numberReady}/{.status.desiredNumberScheduled}"
    else:
        jsonpath = "{.status.readyReplicas}/{.spec.replicas}"
    out = _kubectl(kubeconfig, [
        "get", kind, name, "-n", K8S_NAMESPACE,
        "-o", f"jsonpath={jsonpath}",
    ], check=False)
    ready, _, desired = out.strip().partition("/")
    return bool(ready) and ready == desired and ready != "0"


@flow(name="k8s-daemonset-health", log_prints=True)
def k8s_daemonset_health_flow() -> dict[str, bool]:
    log = get_run_logger()
    if not check_kapsule_task():
        log.info("Kapsule inactif — tous les composants considérés NOK")
        for name, _ in COMPONENTS:
            print(f"DAEMONSET_STATUS {name}=NOK")
        return {name: False for name, _ in COMPONENTS}

    kubeconfig = get_kubeconfig_task()
    results: dict[str, bool] = {}
    for name, kind in COMPONENTS:
        ok = check_daemonset_task(kubeconfig, name, kind)
        results[name] = ok
        print(f"DAEMONSET_STATUS {name}={'OK' if ok else 'NOK'}")
    return results


if __name__ == "__main__":
    k8s_daemonset_health_flow()
