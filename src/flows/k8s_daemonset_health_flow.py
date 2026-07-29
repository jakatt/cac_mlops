"""
K8s DaemonSet health flow — vérifie les 4 composants K8s sans Service stable
ni serveur HTTP (node-exporter, promtail : DaemonSets, 1 IP par nœud, pas de
nom DNS unique à interroger ; loki-forwarder : proxy SOCKS5 ; tailscale-
subnet-router : pas de serveur HTTP). Inatteignables directement depuis le
Cockpit Gradio (pas de kubeconfig sur ce process — seul prefect-worker en
dispose), d'où ce flow dédié déclenché à la demande depuis le Healthcheck.

Imprime une ligne "DAEMONSET_STATUS <name>=OK|NOK" par composant, parsée
depuis les logs du run par le Cockpit (services/gradio/app.py::
_check_k8s_daemonsets) — plus simple à récupérer via l'API REST Prefect
que le résultat sérialisé d'un flow.
"""
from prefect import flow, get_run_logger, task

from src.flows.deploy_kapsule_flow import K8S_NAMESPACE, _kubectl, check_kapsule_task, get_kubeconfig_task

DAEMONSETS = ["node-exporter", "promtail", "loki-forwarder", "tailscale-subnet-router"]


@task(name="check-daemonset-ready")
def check_daemonset_task(kubeconfig: str, name: str) -> bool:
    out = _kubectl(kubeconfig, [
        "get", "daemonset", name, "-n", K8S_NAMESPACE,
        "-o", "jsonpath={.status.numberReady}/{.status.desiredNumberScheduled}",
    ], check=False)
    ready, _, desired = out.strip().partition("/")
    return bool(ready) and ready == desired and ready != "0"


@flow(name="k8s-daemonset-health", log_prints=True)
def k8s_daemonset_health_flow() -> dict[str, bool]:
    log = get_run_logger()
    if not check_kapsule_task():
        log.info("Kapsule inactif — tous les DaemonSets considérés NOK")
        for name in DAEMONSETS:
            print(f"DAEMONSET_STATUS {name}=NOK")
        return {name: False for name in DAEMONSETS}

    kubeconfig = get_kubeconfig_task()
    results: dict[str, bool] = {}
    for name in DAEMONSETS:
        ok = check_daemonset_task(kubeconfig, name)
        results[name] = ok
        print(f"DAEMONSET_STATUS {name}={'OK' if ok else 'NOK'}")
    return results


if __name__ == "__main__":
    k8s_daemonset_health_flow()
