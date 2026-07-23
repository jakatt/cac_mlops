from dotenv import load_dotenv

# Charge .env (SCW_*, AWS_*/MLFLOW_S3_ENDPOINT_URL, MLFLOW_TRACKING_URI...) pour
# toute commande locale `python -m src.xxx` — sans ça, chaque script qui a besoin
# d'une variable d'environnement (ex: credentials S3 MinIO pour les artefacts
# MLflow) échoue silencieusement selon que le shell courant l'a exportée ou non.
# No-op dans les conteneurs : aucun .env n'y est copié (variables déjà injectées
# via docker-compose `environment:`), et override=False par défaut ne les écrase
# jamais si un .env y traînait quand même.
load_dotenv()
