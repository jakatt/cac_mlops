"""Test fonctionnel gradio-public — Phase 2 observabilité par accès (2026-07-28).

Complète test_api_flow.py : aujourd'hui, FastAPI n'est appelé en pratique que
par des tests automatisés et la simulation de drift (full_retrain_flow) —
jamais par un usage réel. gradio-public est le seul vrai point d'accès
utilisateur du système, et jusqu'ici aucun test ne validait fonctionnellement
qu'il répond correctement après un déploiement (seul un ping /health existait,
cf. smoke_test_task). Ce test appelle réellement la fonction "Predict" exposée
par le cockpit public via gradio_client — pas un ping, une vraie inférence de
bout en bout côté UI.
"""
from __future__ import annotations

import os

from prefect import flow, get_run_logger, task

from src.flows.test_api_flow import NGINX_URL, SAMPLE_PAYLOAD


@task(name="test-gradio-public-predict", retries=1)
def test_predict(base_url: str = NGINX_URL) -> str:
    from gradio_client import Client

    client = Client(base_url)
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


@flow(name="test-gradio-public", flow_run_name="test-gradio-public-{base_url}")
def test_gradio_public_flow(base_url: str = NGINX_URL) -> None:
    log = get_run_logger()
    test_predict(base_url=base_url)
    log.info("test-gradio-public OK ✓")


if __name__ == "__main__":
    test_gradio_public_flow(base_url=os.getenv("NGINX_URL", NGINX_URL))
