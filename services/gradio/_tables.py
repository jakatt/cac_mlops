"""Tableaux HTML au format des tableaux des Cockpits (Healthcheck, Modèles, Liens).

En-tête bleu clair en capitales, police à chasse fixe, lignes alternées,
bordures fines. Styles en ligne + !important : le CSS par défaut de Gradio
écrase sinon les couleurs des tableaux insérés en HTML.
Utilisé par les résultats What-if (Cockpits admin et public) et le tableau
« Qualité des données brutes » de l'accordéon Drift.
"""
from __future__ import annotations

NAVY = "#156082"
SLATE = "#374151"
MUTED = "#6B7280"

_FONT = "font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
_CELL = "padding:12px 14px;border:1px solid #E5E7EB;text-align:left;vertical-align:middle;"
_TH = (f"{_CELL}{_FONT}background:#c2dbe4 !important;color:{NAVY} !important;font-size:.8rem !important;"
       "letter-spacing:.5px;text-transform:uppercase;font-weight:700 !important;white-space:nowrap;")
_TD = f"{_CELL}{_FONT}font-size:.85rem !important;color:{SLATE} !important;"
_ROW_BG = ("#FFFFFF", "#F9FAFB")
# Même rendu que les titres « ### … » des onglets (CSS h3 des Cockpits)
_TITLE = (f"color:{NAVY} !important;font-size:.82rem;font-weight:600;text-transform:uppercase;"
          "letter-spacing:.6px;margin:0 0 14px 0;font-family:Inter,'Segoe UI',sans-serif;")


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """`rows` : cellules déjà en HTML (badges, liens, gras autorisés)."""
    head = "".join(f'<th style="{_TH}">{h}</th>' for h in headers)
    body = "".join(
        f'<tr style="background:{_ROW_BG[i % 2]} !important;">'
        + "".join(f'<td style="{_TD}">{c}</td>' for c in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    return (f'<div style="width:100%;overflow-x:auto;"><table style="border-collapse:collapse;width:100%;'
            f'border:1px solid #E5E7EB;"><tr>{head}</tr>{body}</table></div>')


def section_title(text: str) -> str:
    return f'<div style="{_TITLE}">{text}</div>'


def whatif_results(title: str, rows: list[tuple[str, str]], note: str) -> str:
    """Bloc résultats What-if : titre aligné sur celui de la colonne de gauche, tableau, note."""
    return (section_title(f"Résultats — {title}")
            + html_table(["Indicateur", "Valeur"], [[label, value] for label, value in rows])
            + f'<p style="color:{MUTED} !important;font-size:.8rem;font-style:italic;margin:10px 0 0;">{note}</p>')


def whatif_placeholder() -> str:
    return (section_title("Résultats")
            + f'<p style="color:{MUTED} !important;font-size:.85rem;">Les résultats s\'afficheront ici après l\'analyse.</p>')
