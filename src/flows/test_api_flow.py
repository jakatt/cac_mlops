"""
Test API flow — 5 tests de l'API (santé, JWT, 401, 200, 429 rate-limit).
Remplace .github/workflows/test-api.yml
"""
from __future__ import annotations

import os

import requests as http
from prefect import flow, task, get_run_logger

# Dans le réseau Docker interne : nginx écoute sur nginx:80
NGINX_URL    = os.getenv("NGINX_URL",    "http://nginx:80")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")

_SAMPLE_PAYLOAD = {
    "place": 1, "catu": 1, "sexe": 1, "secu1": 0.0,
    "year_acc": 2023, "victim_age": 35.0, "catv": 7.0, "obsm": 0.0,
    "motor": 0.0, "catr": 3.0, "circ": 2.0, "surf": 1.0, "situ": 1.0,
    "vma": 80.0, "jour": 3, "mois": 6, "lum": 1, "dep": 75, "com": 75056,
    "agg_": 1, "intersection_type": 0, "atm": 1.0, "col": 3.0,
    "lat": 48.866667, "long": 2.333333, "hour": 14,
    "nb_victim": 2, "nb_vehicules": 1,
}


@task(name="test-health", retries=2)
def test_health() -> str:
    logger = get_run_logger()
    r = http.get(f"{NGINX_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health check: HTTP {r.status_code}"
    logger.info("✓ /health → %s", r.json())
    return "OK"


@task(name="test-token")
def test_token() -> str:
    logger = get_run_logger()
    r = http.post(
        f"{NGINX_URL}/token",
        data={"username": API_USERNAME, "password": API_PASSWORD},
        timeout=10,
    )
    assert r.status_code == 200, f"/token: HTTP {r.status_code} — {r.text}"
    token = r.json()["access_token"]
    logger.info("✓ JWT token obtenu")
    return token


@task(name="test-401-sans-token")
def test_no_auth() -> str:
    logger = get_run_logger()
    r = http.post(f"{NGINX_URL}/predict", json=_SAMPLE_PAYLOAD, timeout=10)
    assert r.status_code == 401, f"Attendu 401, reçu {r.status_code}"
    logger.info("✓ 401 sans token: OK")
    return "OK"


@task(name="test-200-avec-token")
def test_with_auth(token: str) -> str:
    logger = get_run_logger()
    r = http.post(
        f"{NGINX_URL}/predict",
        json=_SAMPLE_PAYLOAD,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    assert r.status_code == 200, f"Predict: HTTP {r.status_code} — {r.text}"
    logger.info("✓ /predict avec token → %s", r.json())
    return "OK"


@task(name="test-whatif-vitesse-130-vs-110")
def test_whatif_speed(token: str) -> str:
    """Vérifie la cohérence métier : vma=130 → proba gravité > vma=110 (autoroute)."""
    logger = get_run_logger()
    headers = {"Authorization": f"Bearer {token}"}
    autoroute = {
        **_SAMPLE_PAYLOAD,
        "catr": 1.0,   # autoroute
        "agg_": 0,     # hors agglomération
        "lum": 5,      # nuit sans éclairage public
        "hour": 22,    # 22h
        "col": 6,      # collision frontale
    }
    r130 = http.post(f"{NGINX_URL}/predict", json={**autoroute, "vma": 130.0},
                     headers=headers, timeout=10)
    r110 = http.post(f"{NGINX_URL}/predict", json={**autoroute, "vma": 110.0},
                     headers=headers, timeout=10)
    assert r130.status_code == 200, f"Predict vma=130: HTTP {r130.status_code}"
    assert r110.status_code == 200, f"Predict vma=110: HTTP {r110.status_code}"
    p130 = r130.json()["probability"]
    p110 = r110.json()["probability"]
    logger.info("✓ What-If vitesse — proba(vma=130)=%.3f  proba(vma=110)=%.3f  Δ=%.3f",
                p130, p110, p130 - p110)
    assert p130 > p110, (
        f"Cohérence métier KO : proba(vma=130)={p130:.3f} ≤ proba(vma=110)={p110:.3f}"
    )
    return f"OK (Δ={p130 - p110:+.3f})"


@task(name="test-429-rate-limit")
def test_rate_limit(token: str) -> str:
    logger = get_run_logger()
    hit_429 = False
    for i in range(22):
        r = http.post(
            f"{NGINX_URL}/predict",
            json=_SAMPLE_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 429:
            logger.info("✓ Rate-limit 429 déclenché à la requête %d", i + 1)
            hit_429 = True
            break
    assert hit_429, "Rate-limit 429 non déclenché en 22 requêtes (vérifier nginx.conf)"
    return "OK"


@flow(name="test-api", log_prints=True)
def test_api_flow(skip_rate_limit: bool = False) -> dict[str, str]:
    """
    Tests fonctionnels de l'API via nginx.

    skip_rate_limit=True en CD pour éviter de saturer nginx sur la prod.
    Laisser False pour un test manuel complet (6 tests).

    Toujours exécutés (1-5) :
      1. health check
      2. obtention d'un token JWT
      3. 401 sans token
      4. 200 avec token
      5. What-If vitesse : proba(vma=130) > proba(vma=110) (cohérence métier)

    Optionnel (6) :
      6. 429 rate-limit après 22 requêtes (skip si skip_rate_limit=True)
    """
    health    = test_health()
    token     = test_token()
    no_auth   = test_no_auth()
    with_auth = test_with_auth(token)
    whatif    = test_whatif_speed(token)

    results = {
        "health":       health,
        "token":        "OK",
        "no_auth":      no_auth,
        "with_auth":    with_auth,
        "whatif_speed": whatif,
        "rate_limit":   "skipped (CD mode)",
    }

    if not skip_rate_limit:
        results["rate_limit"] = test_rate_limit(token)

    return results
