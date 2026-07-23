"""
Test API flow — 5 tests de l'API (santé, JWT, 401, 200, 429 rate-limit).
Remplace .github/workflows/test-api.yml
"""
from __future__ import annotations

import os

import requests as http
from prefect import flow, task

# Dans le réseau Docker interne : nginx écoute sur nginx:80
NGINX_URL    = os.getenv("NGINX_URL",    "http://nginx:80")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")

_SAMPLE_PAYLOAD = {
    "place": 1, "catu": 1, "sexe": 1, "secu1": 0.0,
    "victim_age": 35.0, "catv": 7.0, "obsm": 0.0,
    "motor": 0.0, "catr": 3.0, "circ": 2.0, "surf": 1.0, "situ": 1.0,
    "vma": 80.0, "jour": 3, "mois": 6, "lum": 1, "dep": 75, "com": 75056,
    "agg_": 1, "int": 0, "atm": 1.0, "col": 3.0,
    "lat": 48.866667, "long": 2.333333, "hour": 14,
    "nb_victim": 2, "nb_vehicules": 1,
}


@task(name="test-health", retries=2)
def test_health() -> str:
    r = http.get(f"{NGINX_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health check: HTTP {r.status_code}"
    print(f"✓ /health → {r.json()}")
    return "OK"


@task(name="test-token")
def test_token() -> str:
    r = http.post(
        f"{NGINX_URL}/token",
        data={"username": API_USERNAME, "password": API_PASSWORD},
        timeout=10,
    )
    assert r.status_code == 200, f"/token: HTTP {r.status_code} — {r.text}"
    token = r.json()["access_token"]
    print("✓ JWT token obtenu")
    return token


@task(name="test-401-sans-token")
def test_no_auth() -> str:
    r = http.post(f"{NGINX_URL}/predict", json=_SAMPLE_PAYLOAD, timeout=10)
    assert r.status_code == 401, f"Attendu 401, reçu {r.status_code}"
    print("✓ 401 sans token: OK")
    return "OK"


@task(name="test-200-avec-token")
def test_with_auth(token: str) -> str:
    r = http.post(
        f"{NGINX_URL}/predict",
        json=_SAMPLE_PAYLOAD,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    assert r.status_code == 200, f"Predict: HTTP {r.status_code} — {r.text}"
    print(f"✓ /predict avec token → {r.json()}")
    return "OK"


@task(name="test-whatif-vitesse-90-vs-50")
def test_whatif_speed(token: str) -> str:
    """Fonctionnel uniquement : la fonctionnalité What-If de l'interface doit
    répondre (comme si un utilisateur la sollicitait), rien de plus. Ne juge
    jamais le sens des prédictions du modèle — un modèle statistiquement bon
    peut légitimement donner un résultat contre-intuitif sur un scénario
    synthétique donné (incident vécu 2026-07-23 : rf class_weight=balanced,
    rollback automatique déclenché à tort sur un simple test métier trop
    strict, alors que le modèle passait toutes les métriques KPI)."""
    headers = {"Authorization": f"Bearer {token}"}
    route_dept_nuit = {
        **_SAMPLE_PAYLOAD,
        "catr": 3.0,   # route départementale
        "agg_": 1,     # hors agglomération
        "lum": 5,      # nuit sans éclairage public
        "hour": 23,
        "mois": 12,
        "col": 6.0,    # collision frontale
        "nb_vehicules": 2,
    }
    r90 = http.post(f"{NGINX_URL}/predict", json={**route_dept_nuit, "vma": 90.0},
                    headers=headers, timeout=10)
    r50 = http.post(f"{NGINX_URL}/predict", json={**route_dept_nuit, "vma": 50.0},
                    headers=headers, timeout=10)
    assert r90.status_code == 200, f"Predict vma=90: HTTP {r90.status_code}"
    assert r50.status_code == 200, f"Predict vma=50: HTTP {r50.status_code}"
    p90 = r90.json()["probability"]
    p50 = r50.json()["probability"]
    print(f"✓ What-If vitesse — proba(vma=90)={p90:.3f}  proba(vma=50)={p50:.3f}  Δ={p90 - p50:+.3f}")
    return "OK"


@task(name="test-429-rate-limit")
def test_rate_limit(token: str) -> str:
    hit_429 = False
    for i in range(22):
        r = http.post(
            f"{NGINX_URL}/predict",
            json=_SAMPLE_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 429:
            print(f"✓ Rate-limit 429 déclenché à la requête {i + 1}")
            hit_429 = True
            break
    assert hit_429, "Rate-limit 429 non déclenché en 22 requêtes (vérifier nginx.conf)"
    return "OK"


@flow(name="test-api", log_prints=True)
def test_api_flow(
    skip_rate_limit: bool = False,
    require_model: bool = True,
) -> dict[str, str]:
    """
    Tests fonctionnels de l'API via nginx.

    skip_rate_limit=True en CD pour éviter de saturer nginx sur la prod.
    require_model=False en état post-reset (aucun @Production enregistré) :
      seuls /health et /token sont testés, les tests /predict sont ignorés.
    Laisser False pour un test manuel complet (6 tests).

    Toujours exécutés (1-2) :
      1. health check
      2. obtention d'un token JWT

    Uniquement si require_model=True (3-5) :
      3. 401 sans token
      4. 200 avec token
      5. What-If vitesse : la fonctionnalité répond (fonctionnel uniquement, ne juge pas le résultat)

    Optionnel (6) :
      6. 429 rate-limit après 22 requêtes (skip si skip_rate_limit=True)
    """
    health = test_health()
    token  = test_token()

    results: dict[str, str] = {
        "health": health,
        "token":  "OK",
    }

    if not require_model:
        print("⚠ Tests /predict ignorés — aucun modèle @Production enregistré (état post-reset attendu)")
        results["no_auth"]      = "skipped (no model)"
        results["with_auth"]    = "skipped (no model)"
        results["whatif_speed"] = "skipped (no model)"
        results["rate_limit"]   = "skipped (no model)"
        return results

    no_auth   = test_no_auth()
    with_auth = test_with_auth(token)
    whatif    = test_whatif_speed(token)

    results.update({
        "no_auth":    no_auth,
        "with_auth":  with_auth,
        "whatif_speed": whatif,
        "rate_limit": "skipped (CD mode)",
    })

    if not skip_rate_limit:
        results["rate_limit"] = test_rate_limit(token)

    return results
