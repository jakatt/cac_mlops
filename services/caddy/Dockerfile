# Image Caddy custom — ajoute le plugin de stockage S3 pour partager l'état
# ACME (certificat Let's Encrypt) entre plusieurs répliques K8s. L'image
# caddy:2-alpine stock ne l'inclut pas (plugin tiers, à compiler via xcaddy).
# Réutilise le bucket S3 déjà en place pour le projet (cac-mlops-data),
# préfixe caddy-certs/ — voir k8s/caddy/configmap.yaml pour la config Caddyfile.
FROM caddy:2-builder-alpine AS build

RUN xcaddy build \
    --with github.com/techknowlogick/certmagic-s3

FROM caddy:2-alpine

COPY --from=build /usr/bin/caddy /usr/bin/caddy
