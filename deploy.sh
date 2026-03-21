#!/usr/bin/env bash
# deploy.sh — Build the Docker image and deploy it to Azure App Service.
#
# Prerequisites:
#   1. Docker installed and running          (docker info)
#   2. Logged in to Docker Hub              (docker login)
#   3. Azure CLI installed and logged in    (az login)
#   4. Terraform applied                    (cd terraform && terraform apply)
#   5. info.pdf present in project root
#   6. pages_data.json + embeddings.npy generated (python preprocess.py)
#
# Usage:
#   bash deploy.sh
#   bash deploy.sh --app-name my-custom-name   # override the app name

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DOCKER_IMAGE="winterzone2/psych101-ai"
APP_NAME="psych101-ai"
RESOURCE_GROUP="${APP_NAME}-rg"

# ── Parse optional flags ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-name) APP_NAME="$2"; RESOURCE_GROUP="${APP_NAME}-rg"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "==> Checking required files..."
missing=0
for f in pages_data.json embeddings.npy; do
  if [[ ! -f "$f" ]]; then
    echo "    MISSING: $f"
    missing=1
  fi
done
if [[ $missing -eq 1 ]]; then
  echo ""
  echo "ERROR: One or more required files are missing."
  echo "  Run:  python preprocess.py   (needs info.pdf in the project root)"
  exit 1
fi
echo "    All required files present."

# ── Build Docker image ────────────────────────────────────────────────────────
echo ""
echo "==> Building Docker image: $DOCKER_IMAGE:latest ..."
echo "    (first build takes ~5-10 min — subsequent builds are fast due to layer caching)"
docker build -t "$DOCKER_IMAGE:latest" .

# ── Push to Docker Hub ────────────────────────────────────────────────────────
echo ""
echo "==> Pushing to Docker Hub..."
docker push "$DOCKER_IMAGE:latest"

# ── Restart Azure Web App to pull the new image ───────────────────────────────
echo ""
echo "==> Restarting Azure Web App: $APP_NAME ..."
az webapp restart --resource-group "$RESOURCE_GROUP" --name "$APP_NAME"

echo ""
echo "==> Deployment complete!"
echo "    Your app: https://${APP_NAME}.azurewebsites.net"
echo ""
echo "    The app will be ready in ~30 seconds after the container starts."
