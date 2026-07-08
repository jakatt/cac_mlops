---
name: feedback-dataframe-none-check
description: "Ne jamais utiliser `df or []` pour tester un DataFrame — ValueError ambiguous"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

Ne jamais écrire `len(some_function() or [])` quand `some_function()` peut retourner un DataFrame.

**Why:** `df or []` tente un cast booléen du DataFrame → `ValueError: The truth value of a DataFrame is ambiguous`. C'est ce qui a crashé `cac_mlops-gradio-1` en boucle (PR #84, 2026-07-05).

**How to apply:** Toujours tester `is not None` explicitement :
```python
# FAUX
_n_base = len(_get_data() or [])

# CORRECT
_df = _get_data()
_n_base = len(_df) if _df is not None else 0
```
