"""Test fonctionnel gradio-public — Phase 2 observabilité par accès (2026-07-28).

Complète test_api_flow.py : aujourd'hui, FastAPI n'est appelé en pratique que
par des tests automatisés et la simulation de drift (full_retrain_flow) —
jamais par un usage réel. gradio-public est le seul vrai point d'accès
utilisateur du système, et jusqu'ici aucun test ne validait fonctionnellement
qu'il répond correctement après un déploiement (seul un ping /health existait,
cf. smoke_test_task). Ce test appelle réellement la fonction "Predict" exposée
par le cockpit public via gradio_client — pas un ping, une vraie inférence de
bout en bout côté UI.

Un test par onglet du Cockpit public (Predict, What-if, Points Noirs) : chacun
appelle la même fonction que le bouton de l'onglet et vérifie que le résultat
s'affiche, sans message d'erreur. Aucun ne juge le sens du résultat (rôle de
l'évaluation du modèle) — seulement que la fonctionnalité marche.
"""
from __future__ import annotations

import os

from prefect import flow, get_run_logger, task

from src.flows.test_api_flow import NGINX_URL, SAMPLE_PAYLOAD, SYNTHETIC_HEADERS


@task(name="test-gradio-public-predict", task_run_name="1 · Onglet Predict : prédire un accident", retries=1)
def test_predict(base_url: str = NGINX_URL) -> str:
    from gradio_client import Client

    client = Client(base_url, headers=SYNTHETIC_HEADERS)  # trafic de TEST côté Cockpit
    # Positionnel — l'ordre de SAMPLE_PAYLOAD correspond exactement à celui des
    # inputs du bouton Predict (services/gradio/app_public.py, _pred_inputs).
    result = client.predict(*SAMPLE_PAYLOAD.values(), api_name="/predict")

    assert isinstance(result, str) and result.strip(), "Réponse vide du cockpit public"
    # run_predict() attrape ses propres exceptions et renvoie une chaîne
    # "Erreur de prédiction : ..." plutôt que de lever — un appel HTTP réussi
    # ne suffit donc pas à prouver que l'inférence a fonctionné, il faut
    # inspecter le contenu de la réponse.
    assert "Erreur" not in result, f"Prédiction en erreur côté cockpit public : {result}"
    assert "prioritaire" in result.lower(), f"Réponse inattendue du cockpit public : {result}"

    print(f"✓ gradio-public /predict → {result.splitlines()[0]}")
    return result


def _client(base_url: str):
    from gradio_client import Client
    return Client(base_url, headers=SYNTHETIC_HEADERS)  # trafic de TEST côté Cockpit


@task(name="test-gradio-public-whatif", task_run_name="2 · Onglet What-if : simuler les giratoires", retries=1)
def test_whatif(base_url: str = NGINX_URL) -> str:
    # Même appel que le bouton « Lancer l'analyse » : scénario, taille d'échantillon, multiplicateur.
    chart, stats = _client(base_url).predict("giratoire", 5000, 2.0, api_name="/whatif")

    assert isinstance(stats, str) and stats.strip(), "Réponse vide du What-if"
    # run_whatif() renvoie ses erreurs en texte (« Erreur scenario / de prediction : … »).
    assert "Erreur" not in stats, f"What-if en erreur côté cockpit public : {stats}"
    assert "Gravité prédite (scénario)" in stats, f"Tableau de résultats absent : {stats}"
    assert chart, "Graphique What-if absent"

    delta = next((line for line in stats.splitlines() if line.startswith("| Delta")), "")
    print(f"✓ gradio-public /whatif (giratoires) → {delta.strip('| ')}")
    return stats


@task(name="test-gradio-public-points-noirs", task_run_name="3 · Onglet Points Noirs : générer la carte", retries=1)
def test_points_noirs(base_url: str = NGINX_URL) -> str:
    # Même appel que le bouton « Generer la carte », avec les réglages par défaut de l'onglet.
    fig, top10, stats = _client(base_url).predict(40, 3, [], 20000, api_name="/points_noirs")

    assert isinstance(stats, str) and "zones" in stats, f"Résumé de la carte absent : {stats!r}"
    assert fig, "Carte Points Noirs absente"
    rows = (top10 or {}).get("data", []) if isinstance(top10, dict) else []
    assert rows, "Top 10 des zones vide"

    print(f"✓ gradio-public /points_noirs → {stats} · top 10 : {len(rows)} zones")
    return stats


@flow(name="test-gradio-public", flow_run_name="test-gradio-public-{base_url}")
def test_gradio_public_flow(base_url: str = NGINX_URL) -> None:
    log = get_run_logger()
    test_predict(base_url=base_url)
    test_whatif(base_url=base_url)
    test_points_noirs(base_url=base_url)
    log.info("test-gradio-public OK ✓ (Predict, What-if, Points Noirs)")


if __name__ == "__main__":
    test_gradio_public_flow(base_url=os.getenv("NGINX_URL", NGINX_URL))
