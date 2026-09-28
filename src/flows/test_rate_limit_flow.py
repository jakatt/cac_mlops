"""
Test rate-limit flow — vérifie les 2 niveaux de protection anti-abus de nginx
sur POST /predict (cf. services/nginx/nginx.conf) :

  - limite PAR CLIENT (IP réelle, 20/min, rafale 10) : un client abusif est
    bloqué seul, les autres clients ne sont pas pénalisés ;
  - limite GLOBALE (60/min, rafale 30) : plafond de charge du serveur, même
    quand chaque client reste sous sa propre limite.

Plusieurs clients sont simulés depuis le réseau Docker interne via l'en-tête
X-Forwarded-For (IP fictives des plages de documentation RFC 5737). nginx ne
croit cet en-tête que depuis les réseaux privés Docker ; côté Internet, Caddy
l'écrase avec l'IP réelle — impossible à falsifier de l'extérieur.

Déclenché manuellement : Cockpit → Orchestration → « Tester la protection
anti-abus sur l'API ». Jamais en CD : la dernière étape épuise volontairement le quota
global (~15 s de 429 possibles pour un vrai client /predict).
"""
from __future__ import annotations

import random

from prefect import flow, task

from src.flows.test_api_flow import API_PASSWORD, API_USERNAME, NGINX_URL, SAMPLE_PAYLOAD, http

# Doivent rester alignés sur services/nginx/nginx.conf (affichage uniquement).
PER_CLIENT_RATE, PER_CLIENT_BURST = "20/min", 10
GLOBAL_RATE, GLOBAL_BURST = "60/min", 30

# Garde-fous : nombre max de requêtes envoyées par étape.
MAX_SINGLE_CLIENT_REQUESTS = 20
REQUESTS_PER_GLOBAL_CLIENT = 3   # < PER_CLIENT_BURST + 1 : aucun client ne dépasse sa limite
MAX_GLOBAL_CLIENTS = 15


def _fake_client_ips(n: int) -> list[str]:
    """IP fictives uniques par run — un run relancé juste après ne réutilise pas
    des quotas « par client » encore entamés par le précédent."""
    pool = [f"{net}.{i}" for net in ("198.51.100", "203.0.113") for i in range(1, 255)]
    return random.sample(pool, n)


def _predict_as(client_ip: str, token: str, base_url: str) -> int:
    r = http.post(
        f"{base_url}/predict",
        json=SAMPLE_PAYLOAD,
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": client_ip},
        timeout=10,
    )
    assert r.status_code in (200, 429), f"Réponse inattendue : HTTP {r.status_code} — {r.text[:200]}"
    return r.status_code


@task(name="1 · API en ligne (GET health)", retries=2)
def check_api_up(base_url: str) -> None:
    r = http.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, f"/health : HTTP {r.status_code}"
    print("✓ L'API répond (HTTP 200) — on peut tester la protection anti-abus.")


@task(name="2 · Obtenir un jeton JWT")
def get_token(base_url: str) -> str:
    r = http.post(
        f"{base_url}/token",
        data={"username": API_USERNAME, "password": API_PASSWORD},
        timeout=10,
    )
    assert r.status_code == 200, f"/token : HTTP {r.status_code} — {r.text[:200]}"
    print("✓ Jeton JWT obtenu — les prédictions suivantes sont authentifiées.")
    return r.json()["access_token"]


@task(name="3 · Limite par client — rafale d'un seul client")
def single_client_burst(token: str, client_ip: str, base_url: str) -> dict:
    accepted = 0
    for i in range(1, MAX_SINGLE_CLIENT_REQUESTS + 1):
        if _predict_as(client_ip, token, base_url) == 429:
            print(
                f"✓ Client A ({client_ip}) : {accepted} prédictions acceptées, "
                f"la n°{i} refusée par nginx (HTTP 429)."
            )
            return {"accepted": accepted, "blocked_at": i}
    raise AssertionError(
        f"Client A : {MAX_SINGLE_CLIENT_REQUESTS} prédictions acceptées d'affilée, "
        f"aucun refus — la limite par client ({PER_CLIENT_RATE}, rafale {PER_CLIENT_BURST}) "
        "ne fonctionne pas (vérifier nginx.conf et real_ip)."
    )


@task(name="4 · Limite par client — un autre client n'est pas pénalisé")
def other_client_not_blocked(token: str, client_ip: str, base_url: str) -> None:
    status = _predict_as(client_ip, token, base_url)
    assert status == 200, (
        f"Client B ({client_ip}) refusé (HTTP {status}) alors qu'il n'a rien envoyé — "
        "les clients partagent un même quota (vérifier real_ip dans nginx.conf)."
    )
    print(f"✓ Client B ({client_ip}) : accepté (HTTP 200) juste après le blocage de A.")


@task(name="5 · Limite globale — plusieurs clients sous leur quota")
def many_clients_global_cap(token: str, client_ips: list[str], base_url: str) -> dict:
    accepted = 0
    for c, ip in enumerate(client_ips, start=1):
        for _ in range(REQUESTS_PER_GLOBAL_CLIENT):
            if _predict_as(ip, token, base_url) == 429:
                n = accepted + 1
                print(
                    f"✓ Limite globale : {accepted} prédictions acceptées puis la n°{n} refusée "
                    f"(HTTP 429), réparties sur {c} clients qui envoyaient chacun "
                    f"≤ {REQUESTS_PER_GLOBAL_CLIENT} requêtes — sous leur limite individuelle."
                )
                return {"accepted": accepted, "blocked_at": n, "clients": c}
            accepted += 1
    raise AssertionError(
        f"{accepted} prédictions acceptées sur {len(client_ips)} clients, aucun refus — "
        f"la limite globale ({GLOBAL_RATE}, rafale {GLOBAL_BURST}) ne fonctionne pas."
    )


@flow(name="test-rate-limit", log_prints=True)
def test_rate_limit_flow(base_url: str = NGINX_URL) -> dict:
    ips = _fake_client_ips(2 + MAX_GLOBAL_CLIENTS)
    ip_a, ip_b, global_ips = ips[0], ips[1], ips[2:]

    check_api_up(base_url)
    token = get_token(base_url)
    per_client = single_client_burst(token, ip_a, base_url)
    other_client_not_blocked(token, ip_b, base_url)
    glob = many_clients_global_cap(token, global_ips, base_url)

    # Places du quota global déjà prises par les étapes 3 et 4 (A accepté + B),
    # d'où un blocage global avant la rafale complète de GLOBAL_BURST + 1.
    consumed_before = per_client["accepted"] + 1
    result = {
        "per_client_rate":       PER_CLIENT_RATE,
        "per_client_burst":      PER_CLIENT_BURST,
        "global_rate":           GLOBAL_RATE,
        "global_burst":          GLOBAL_BURST,
        "per_client_accepted":   per_client["accepted"],
        "per_client_blocked_at": per_client["blocked_at"],
        "global_consumed_before": consumed_before,
        "global_accepted":       glob["accepted"],
        "global_blocked_at":     glob["blocked_at"],
        "global_clients":        glob["clients"],
    }
    # Ligne structurée lue par le Cockpit pour afficher un résultat simple.
    print("event=rate_limit_result status=ok " + " ".join(f"{k}={v}" for k, v in result.items()))
    return result


if __name__ == "__main__":
    test_rate_limit_flow()
