"""
Kapsule Up — crée le node pool, déploie la stack K8s complète.
Remplace .github/workflows/kapsule-up.yml
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import boto3
import joblib
import mlflow
from botocore.config import Config
from prefect import flow, task, get_run_logger

CLUSTER_ID    = os.getenv("KAPSULE_CLUSTER_ID", "")
KAPSULE_STATE = Path(os.getenv("KAPSULE_STATE", "/app/state/kapsule_ips"))
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
APP_DIR       = Path(os.getenv("WORKING_DIR", "/app"))
K8S_DIR       = APP_DIR / "k8s"
INFRA_DIR     = APP_DIR / "infrastructure"
K8S_NAMESPACE = "cac-mlops"
SCW_S3_URL    = "https://s3.fr-par.scw.cloud"
SCW_REGION    = "fr-par"
SCW_BUCKET    = "cac-mlops-data"

ALL_MODEL_NAMES = ["lgbm_accidents", "rf_accidents", "xgb_accidents"]


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


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=SCW_S3_URL,
        region_name=SCW_REGION,
        aws_access_key_id=os.getenv("SCW_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SCW_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


@task(name="create-node-pool")
def create_node_pool(node_type: str, node_count: int) -> str:
    logger = get_run_logger()
    if not CLUSTER_ID:
        raise ValueError("KAPSULE_CLUSTER_ID non configuré")
    existing = json.loads(_scw(["k8s", "pool", "list", f"cluster-id={CLUSTER_ID}", "-o", "json"]))
    if any(p["name"] == "main" for p in existing):
        logger.info("Pool 'main' déjà existant sur %s — création ignorée (retry après échec en aval)", CLUSTER_ID)
        return "already-exists"
    logger.info("Création pool %s×%d sur cluster %s", node_type, node_count, CLUSTER_ID)
    _scw([
        "k8s", "pool", "create",
        f"cluster-id={CLUSTER_ID}",
        "name=main",
        f"size={node_count}",
        f"node-type={node_type}",
        "zone=fr-par-1",
    ])
    logger.info("✓ Pool demandé")
    return "created"


@task(name="wait-pool-ready")
def wait_pool_ready(max_minutes: int = 15) -> str:
    logger = get_run_logger()
    max_iter = max_minutes * 6
    for i in range(1, max_iter + 1):
        raw = _scw(["k8s", "pool", "list", f"cluster-id={CLUSTER_ID}", "-o", "json"])
        pools = json.loads(raw)
        status = pools[-1]["status"] if pools else "none"
        logger.info("[%ds] pool status=%s", i * 10, status)
        if status == "ready":
            logger.info("✓ Pool ready")
            return status
        if status == "warning" and i >= 30:
            logger.warning("Pool warning après %ds — on continue", i * 10)
            return status
        if i == max_iter:
            raise TimeoutError(f"Pool not ready après {max_minutes} min (status={status})")
        time.sleep(10)
    return "unknown"


@task(name="get-kubeconfig-up")
def get_kubeconfig() -> str:
    logger = get_run_logger()
    logger.info("Récupération kubeconfig cluster %s", CLUSTER_ID)
    content = _scw(["k8s", "kubeconfig", "get", CLUSTER_ID])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
    out = _kubectl(f.name, ["cluster-info"])
    logger.info(out)
    return f.name


@task(name="upload-model-s3")
def upload_model_s3() -> str:
    """Export du modèle @Production dans un fichier temporaire — jamais sous
    src/ : ce dossier est monté :ro dans prefect-worker (override du code des
    flows), toute écriture y échoue avec "Read-only file system" (incident
    vécu, 2026-07-10, premier run réel de kapsule-up-flow)."""
    logger = get_run_logger()
    logger.info("Export modele @Production depuis MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    exported = False
    model_path = ""
    for name in ALL_MODEL_NAMES:
        try:
            model = mlflow.sklearn.load_model(f"models:/{name}@Production")
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                model_path = f.name
            joblib.dump(model, model_path)
            logger.info("✓ %s@Production exporté → %s", name, model_path)
            exported = True
            break
        except Exception as e:
            logger.warning("skip %s: %s", name, e)
    if not exported:
        raise RuntimeError("Aucun modele @Production trouvé dans le registry MLflow")

    s3 = _s3_client()
    s3.upload_file(model_path, SCW_BUCKET, "k8s-model/trained_model.joblib")
    logger.info("✓ Modele uploadé → s3://%s/k8s-model/trained_model.joblib", SCW_BUCKET)
    return model_path


@task(name="upload-data-s3")
def upload_data_s3() -> str:
    logger = get_run_logger()
    data_root = APP_DIR / "data" / "preprocessed"
    candidates = [
        data_root / "cumul_2021_2022_2023",
        data_root / "cumul_2021_2022",
        data_root / "2023",
        data_root / "2022",
    ]
    s3 = _s3_client()
    for ppath in candidates:
        if (ppath / "X_test.csv").exists() and (ppath / "y_test.csv").exists():
            for fname in ("X_test.csv", "y_test.csv"):
                s3.upload_file(str(ppath / fname), SCW_BUCKET, f"k8s-gradio-data/{fname}")
                logger.info("✓ %s uploadé depuis %s", fname, ppath)
            return str(ppath)
    raise FileNotFoundError(f"Aucune donnée preprocessée trouvée dans {data_root}")


@task(name="setup-namespace-secrets")
def setup_namespace_secrets(kubeconfig: str) -> str:
    logger = get_run_logger()
    # Namespace idempotent
    _kubectl(kubeconfig, [
        "create", "namespace", K8S_NAMESPACE,
        "--dry-run=client", "-o", "yaml",
    ])
    _kubectl(kubeconfig, ["apply", "-f", str(K8S_DIR / "namespace.yaml")])
    logger.info("✓ Namespace %s", K8S_NAMESPACE)

    scw_key    = os.getenv("SCW_ACCESS_KEY_ID", "")
    scw_secret = os.getenv("SCW_SECRET_ACCESS_KEY", "")
    jwt_secret = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
    api_user   = os.getenv("API_USERNAME", "admin")
    api_pass   = os.getenv("API_PASSWORD", "changeme")
    pg_pass    = os.getenv("POSTGRES_PASSWORD", "mlops")

    for sec_name, literals in [
        ("s3-creds", [
            f"AWS_ACCESS_KEY_ID={scw_key}",
            f"AWS_SECRET_ACCESS_KEY={scw_secret}",
        ]),
        ("app-creds", [
            f"JWT_SECRET_KEY={jwt_secret}",
            f"API_USERNAME={api_user}",
            f"API_PASSWORD={api_pass}",
            f"POSTGRES_PASSWORD={pg_pass}",
        ]),
    ]:
        args = [
            "create", "secret", "generic", sec_name,
            "-n", K8S_NAMESPACE, "--dry-run=client", "-o", "yaml",
        ]
        for lit in literals:
            args += ["--from-literal", lit]
        yaml_out = _kubectl(kubeconfig, args)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_out)
        _kubectl(kubeconfig, ["apply", "-f", f.name])
        logger.info("✓ Secret %s", sec_name)

    return "OK"


@task(name="setup-tailscale-secret")
def setup_tailscale_secret(kubeconfig: str) -> str:
    """Secret pour le subnet-router (k8s/tailscale/) qui remplace les
    LoadBalancer publics de gradio/prefect-server/grafana par un accès
    Tailscale — parité avec le VPS (${VPS_TAILSCALE_IP}, aucune de ces
    interfaces n'a d'authentification applicative propre à gradio/prefect).
    Clé reusable + ephemeral, taguée tag:k8s-cac-mlops — cf. .env.example
    pour la configuration Tailscale ACL (tagOwners/autoApprovers) requise
    une seule fois côté console, sans quoi la route doit être ré-approuvée
    manuellement à chaque cycle kapsule-up/down."""
    logger = get_run_logger()
    authkey = os.getenv("TAILSCALE_AUTHKEY", "")
    if not authkey:
        logger.warning(
            "TAILSCALE_AUTHKEY absent — subnet-router non fonctionnel : "
            "gradio/prefect-server/grafana (ClusterIP) resteront inaccessibles "
            "tant que la clé n'est pas configurée (voir .env.example)"
        )
        return "skipped"
    args = [
        "create", "secret", "generic", "tailscale-auth",
        "-n", K8S_NAMESPACE, "--dry-run=client", "-o", "yaml",
        "--from-literal", f"TS_AUTHKEY={authkey}",
    ]
    yaml_out = _kubectl(kubeconfig, args)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_out)
    _kubectl(kubeconfig, ["apply", "-f", f.name])
    logger.info("✓ Secret tailscale-auth")
    return "OK"


@task(name="setup-grafana-configmaps")
def setup_grafana_configmaps(kubeconfig: str) -> str:
    logger = get_run_logger()
    for cm_name, cm_path in [
        ("grafana-provisioning-datasources", INFRA_DIR / "grafana" / "provisioning" / "datasources"),
        ("grafana-provisioning-dashboards",  INFRA_DIR / "grafana" / "provisioning" / "dashboards"),
        ("grafana-dashboards",               INFRA_DIR / "grafana" / "dashboards"),
    ]:
        args = [
            "create", "configmap", cm_name,
            f"--from-file={cm_path}",
            "-n", K8S_NAMESPACE, "--dry-run=client", "-o", "yaml",
        ]
        yaml_out = _kubectl(kubeconfig, args)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_out)
        _kubectl(kubeconfig, ["apply", "-f", f.name])
        logger.info("✓ ConfigMap %s", cm_name)
    return "OK"


@task(name="apply-k8s-manifests")
def apply_manifests(kubeconfig: str) -> str:
    logger = get_run_logger()
    _kubectl(kubeconfig, ["apply", "-f", str(K8S_DIR / "namespace.yaml")])
    _kubectl(kubeconfig, ["apply", "-f", str(K8S_DIR / "configmap.yaml")])
    for subdir in ["api", "mlflow", "nginx", "prefect", "prometheus", "grafana", "gradio", "tailscale"]:
        d = K8S_DIR / subdir
        if d.exists():
            _kubectl(kubeconfig, ["apply", "-f", f"{d}/"])
            logger.info("✓ kubectl apply k8s/%s/", subdir)
        else:
            logger.warning("Répertoire k8s/%s/ absent — ignoré", subdir)
    return "OK"


@task(name="patch-prefect-url")
def patch_prefect_url(kubeconfig: str) -> str:
    """prefect-server est en ClusterIP (pas de LoadBalancer — voir sécurité,
    Prefect OSS n'a aucune authentification native). L'URL DNS interne au
    cluster est stable dès la création du Service (contrairement à une IP
    de LoadBalancer, qui change à chaque recréation kapsule-up/down) —
    reachable depuis l'extérieur uniquement via le subnet-router Tailscale
    + le nameserver split-DNS cluster.local configuré côté admin Tailscale."""
    logger = get_run_logger()
    url = f"http://prefect-server.{K8S_NAMESPACE}.svc.cluster.local:4200/api"
    _kubectl(kubeconfig, [
        "set", "env", "deployment/prefect-server",
        "-n", K8S_NAMESPACE,
        f"PREFECT_UI_API_URL={url}",
    ])
    logger.info("✓ PREFECT_UI_API_URL=%s", url)
    return url


@task(name="wait-api-ready-k8s")
def wait_api_ready(kubeconfig: str) -> str:
    logger = get_run_logger()
    out = _kubectl(kubeconfig, [
        "wait", "deployment", "api",
        "-n", K8S_NAMESPACE,
        "--for=condition=available",
        "--timeout=300s",
    ])
    logger.info(out)
    return "ready"


@task(name="write-kapsule-state")
def write_kapsule_state(kubeconfig: str) -> dict[str, str]:
    """nginx reste seul exposé publiquement (LoadBalancer) — équivalent du
    chemin public Caddy→nginx→gradio-public sur le VPS. grafana/prefect-
    server/gradio sont en ClusterIP (parité VPS — Tailscale-only, aucune
    authentification propre à gradio/prefect) : reachable uniquement via le
    subnet-router Tailscale (k8s/tailscale/) + split-DNS cluster.local."""
    logger = get_run_logger()
    ips: dict[str, str] = {}

    out = _kubectl(kubeconfig, [
        "get", "svc", "nginx", "-n", K8S_NAMESPACE,
        "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}",
    ], check=False)
    ips["NGINX_LB"] = out.strip() or "pending"

    for svc_name, key in [
        ("grafana",        "GRAFANA_DNS"),
        ("prefect-server", "PREFECT_DNS"),
        ("gradio",         "GRADIO_DNS"),
    ]:
        ips[key] = f"{svc_name}.{K8S_NAMESPACE}.svc.cluster.local"

    KAPSULE_STATE.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(f"{k}={v}" for k, v in ips.items())
    KAPSULE_STATE.write_text(content + "\n")
    logger.info("✓ state/kapsule_ips écrit:\n%s", content)

    logger.info("=== Services ===")
    logger.info(_kubectl(kubeconfig, ["get", "svc", "-n", K8S_NAMESPACE], check=False))
    logger.info("=== Pods ===")
    logger.info(_kubectl(kubeconfig, ["get", "pods", "-n", K8S_NAMESPACE], check=False))
    return ips


@flow(name="kapsule-up", log_prints=True)
def kapsule_up_flow(
    node_type:  str = "BASIC3-X2C-8G",
    node_count: int = 2,
) -> dict[str, str]:
    """
    Provisionne Kapsule K8s :
      1. Crée le node pool (node_type × node_count)
      2. Attend que les nœuds soient ready
      3. Récupère le kubeconfig
      4. Upload modele @Production → S3 (s3://cac-mlops-data/k8s-model/)
      5. Upload X_test/y_test → S3 (s3://cac-mlops-data/k8s-gradio-data/)
      6. Namespace + Secrets K8s (app + tailscale-auth)
      7. ConfigMaps Grafana
      8. kubectl apply de tous les manifests k8s/ (dont le subnet-router Tailscale)
      9. Patch PREFECT_UI_API_URL (DNS interne cluster, ClusterIP)
      10. Attend que le deployment api soit available
      11. Écrit les adresses dans state/kapsule_ips (IP publique pour nginx,
          DNS interne pour grafana/prefect/gradio — reachable via Tailscale)
    """
    create_node_pool(node_type, node_count)
    wait_pool_ready()
    kubeconfig = get_kubeconfig()
    upload_model_s3()
    upload_data_s3()
    setup_namespace_secrets(kubeconfig)
    setup_tailscale_secret(kubeconfig)
    setup_grafana_configmaps(kubeconfig)
    apply_manifests(kubeconfig)
    patch_prefect_url(kubeconfig)
    wait_api_ready(kubeconfig)
    ips = write_kapsule_state(kubeconfig)
    return ips
