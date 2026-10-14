# FloodSight web app for any container host (Azure Container Apps: deploy/AZURE.md; Hugging Face: deploy/README.md).
# Secrets (SH_CLIENT_ID, ...) are set in the host's settings, never baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    HOST=0.0.0.0 PORT=7860 FLOODSIGHT_PUBLIC=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU PyTorch (the free tier has no GPU); then the rest
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# libexpat: Python's XML reader needs it (flow path and GRD metadata); the slim image may not include it.
# Installed here, after the Python packages, so a rebuild reuses the large PyTorch layers.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

COPY --chown=user . .
# the AI model is skipped silently if its weights are missing, so refuse to build without them
RUN test -f models/siamese_unet.pt || (echo "ERROR: models/siamese_unet.pt is missing" && exit 1)
# fail the build, not the live site, if a system library is missing
RUN python -c "import xml.parsers.expat, rasterio, pyproj, shapely, torch; import floodsight.pipeline, floodsight.analysis.flowtrace, server.api; print('imports ok')"
# showcase runs (finished analyses) are served from runs/; the folder must be writable for new runs
RUN mkdir -p runs && cp -r showcase/. runs/ 2>/dev/null || true; chown -R user:user /app
USER user

EXPOSE 7860
CMD ["python", "serve.py", "--no-browser"]
