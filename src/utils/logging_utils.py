"""Configuration logging partagée — un seul point d'init pour tout le projet.

Certains modules (train_model.py, make_dataset.py, import_raw_data.py) n'appelaient
`logging.basicConfig()` que dans leur `main()` — jamais quand ils sont importés comme
librairies par les flows Prefect, qui ne passent jamais par ce `main()`. Résultat : la
visibilité des logs INFO dépendait de l'ordre d'import dans le process, pas d'une
configuration garantie.

`init_logging()` doit être appelée au niveau module (pas dans un `main()`) par tout
fichier qui définit un logger utilisé en dehors d'un contexte Prefect (`get_run_logger()`
n'est valide que dans une tâche/flow active). `force=True` réinitialise le root logger
même si un autre module l'a déjà configuré différemment — peu importe qui importe quoi
en premier, tous convergent vers le même format.
"""
import logging

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATEFMT = "%H:%M:%S"


def init_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT, force=True)
