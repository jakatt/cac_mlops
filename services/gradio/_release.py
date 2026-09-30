"""Pastille « Mise à jour du … » en bas de la page d'accueil des 3 Cockpits.

Lit docs/release.json sur l'adresse publique des docs : ce fichier n'est mis
en ligne qu'au GO (sync-static-assets, deploy_vps_flow.py), sans rebuild ni
redémarrage. Il sert à la démo live Trigger 2 (scripts/demo_release.sh) : une
PR d'une ligne → CI → merge → gate → GO → la nouvelle date apparaît.
"""
from __future__ import annotations

import html
import os
import time

import requests

RELEASE_URL = os.getenv("RELEASE_URL", "https://mlops.jakat-inc.fr/ci-docs/release.json")
_CACHE_S = 10
_cache: tuple[float, str] = (0.0, "")


def _label() -> str:
    global _cache
    if time.time() - _cache[0] < _CACHE_S:
        return _cache[1]
    try:
        r = requests.get(RELEASE_URL, timeout=2)
        label = str(r.json().get("display", "")) if r.ok else ""
    except Exception:
        label = ""
    _cache = (time.time(), label)
    return label


def release_badge_html() -> str:
    label = _label()
    if not label:
        return ""
    return (
        '<div style="display:flex;justify-content:flex-end;margin-top:6px;">'
        '<span style="font-family:Inter,Segoe UI,sans-serif;font-size:.78rem;color:#156082 !important;'
        'background:#eaf4f9;border:1px solid #c2dbe4;border-radius:999px;padding:5px 12px;">'
        f'🕒 Mise à jour du {html.escape(label)}</span></div>'
    )
