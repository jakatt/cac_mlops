# Bucket policy `cac-mlops-data` — clé S3 restreinte pour Caddy

`caddy-bucket-policy.json` restreint l'application IAM Scaleway
`caddy-cert-storage` (utilisée par Caddy K8s pour stocker ses certificats
Let's Encrypt) au seul préfixe `caddy-certs/` du bucket `cac-mlops-data` —
elle ne peut rien lire/écrire/lister ailleurs dans le bucket (DVC, artefacts
MLflow, `k8s-model/`...).

## Setup one-time (déjà fait, gardé ici pour reproductibilité)

```bash
scw iam application create name=caddy-cert-storage
scw iam policy create name=caddy-cert-storage-policy application-id=<APP_ID> \
  rules.0.project-ids.0=<PROJECT_ID> \
  rules.0.permission-set-names.0=ObjectStorageReadOnly \
  rules.0.permission-set-names.1=ObjectStorageObjectsWrite \
  rules.0.permission-set-names.2=ObjectStorageObjectsDelete
scw iam api-key create application-id=<APP_ID> \
  description="Caddy cert storage - K8s" expires-at=<+365j max>
# → AccessKey/SecretKey à mettre dans .env (CADDY_S3_ACCESS_KEY_ID/CADDY_S3_SECRET_ACCESS_KEY)

# Appliquer la bucket policy (remplacer les IDs par les vrais avant) :
python3 -c "
import boto3, json
s3 = boto3.client('s3', endpoint_url='https://s3.fr-par.scw.cloud', region_name='fr-par',
    aws_access_key_id='<SCW_ACCESS_KEY_ID admin>', aws_secret_access_key='<SCW_SECRET_ACCESS_KEY admin>')
s3.put_bucket_policy(Bucket='cac-mlops-data', Policy=open('caddy-bucket-policy.json').read())
"
```

## ⚠️ Piège rencontré en production (2026-07-14)

Scaleway Object Storage : dès qu'une bucket policy existe sur un bucket,
**tous** les principals (y compris le owner de l'organisation) doivent y
être **explicitement autorisés** pour continuer à accéder au bucket — une
IAM policy large (même `ObjectStorageFullAccess`/owner) ne suffit plus si
une bucket policy est posée sans mentionner ce principal.

Première version de cette policy : seulement les 2 statements Caddy, sans
statement pour le owner. Résultat en prod : `s3-creds` (la clé admin
réutilisée par `fetch-model` de l'API, DVC, MLflow, gradio-public...) s'est
retrouvée en `AccessDenied` partout, y compris sur des objets qu'elle
lisait sans problème avant. Symptôme observé : l'initContainer
`fetch-model` du pod `api` en CrashLoopBackOff, rollout `api` timeout à
300s, déploiement Kapsule échoué + rollback auto.

**Toujours inclure un statement `AllowOwnerFullAccess` explicite** (voir
`caddy-bucket-policy.json`) dans toute bucket policy Scaleway, même quand
l'intention est juste d'ajouter un accès restreint pour un nouveau
principal — sinon la simple présence de la policy bascule tout le bucket
en mode restrictif pour tout le monde.

## Vérification

```bash
# Doit réussir (owner, accès complet) :
aws s3 ls s3://cac-mlops-data/ --endpoint-url https://s3.fr-par.scw.cloud

# Doit réussir (clé Caddy, uniquement sous caddy-certs/) :
aws s3 ls s3://cac-mlops-data/caddy-certs/ --endpoint-url https://s3.fr-par.scw.cloud \
  --profile caddy-scoped

# Doit échouer en AccessDenied (clé Caddy, hors préfixe) :
aws s3 ls s3://cac-mlops-data/k8s-model/ --endpoint-url https://s3.fr-par.scw.cloud \
  --profile caddy-scoped
```
