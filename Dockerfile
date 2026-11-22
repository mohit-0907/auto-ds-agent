# syntax=docker/dockerfile:1
#
# The web app in one image: a Node stage builds the React frontend, then a slim Python
# stage runs `autods serve` (API + built frontend on one port).
#
# The API key is NOT in the image. Pass it when the container starts:
#   docker build -t autods .
#   docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v autods-runs:/app/runs autods
# (README "Docker" explains each flag.)

# ---- Stage 1: build the frontend ---------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
# Lockfile first: `npm ci` is only re-run when the dependencies change, not on every
# edit to a component.
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: the app ---------------------------------------------------------------
# Nothing from stage 1 reaches this image except web/dist: no Node, no node_modules.
FROM python:3.12-slim

# LightGBM needs the GNU OpenMP runtime, which slim images leave out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies before code, so editing src/ reuses this (slow) layer.
# --locked: exactly uv.lock, or fail. --no-dev: no pytest/ruff in the image.
# XGBoost pulls in NVIDIA's NCCL on Linux (~345 MB) for multi-GPU training; the arena
# trains on CPU only, and XGBoost loads NCCL only when it is asked to use GPUs.
ARG UV_SKIP="--no-install-package nvidia-nccl-cu12 --no-install-package nvidia-nccl-cu13"
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project $UV_SKIP

# The project itself, installed editable on purpose: config.py finds datasets/ and
# web/dist relative to src/ (PROJECT_ROOT = /app), which a site-packages install breaks.
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev $UV_SKIP

COPY datasets/ datasets/
COPY --from=web /web/dist web/dist

# Run as an ordinary user. Everything above stays owned by root (read-only to the app);
# runs/ is the only folder it may write to. A new named volume mounted there copies
# this folder's ownership, so the volume is writable too.
RUN useradd --create-home --uid 1000 autods \
    && mkdir runs \
    && chown autods runs
USER autods

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

# 0.0.0.0: inside a container, 127.0.0.1 is unreachable from the host even with -p.
CMD ["autods", "serve", "--host", "0.0.0.0", "--port", "8000"]
