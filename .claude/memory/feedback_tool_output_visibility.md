---
name: feedback-tool-output-visibility
description: "Le user ne voit JAMAIS le stdout des tool calls (Bash, etc.) — tout contenu destiné au user (schémas, tableaux, résultats) doit être écrit directement dans le texte de réponse"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Un schéma ASCII généré via `cat << 'EOF' ... EOF` dans un appel Bash n'est **pas visible** par le user — seul le texte écrit en dehors des tool calls l'est. Corrigé explicitement le 2026-07-22 : "tu n'affiche pas de tableau ou schema !" après avoir produit un diagramme ETL/DVC entier via Bash au lieu de l'écrire dans la réponse.

**Why:** Réflexe erroné d'utiliser Bash/heredoc pour "afficher" quelque chose de propre/aligné (monospace), en oubliant que le résultat d'un tool call est invisible pour le user — seul le texte de la réponse compte.

**How to apply:** Tout schéma ASCII, diagramme, tableau récapitulatif ou contenu destiné à être *lu par le user* doit être écrit directement dans le corps de la réponse (texte markdown), jamais généré/imprimé via un tool call (Bash `echo`/`cat`/`printf`, etc.) dans l'espoir que ça s'affiche. Réserver Bash à l'exécution réelle (tests, vérifications, actions) — jamais à la présentation. Si un doute existe sur si un contenu doit être visible, l'écrire en texte direct par défaut. Ce piège n'est pas spécifique à ce projet — s'applique à toute conversation.
