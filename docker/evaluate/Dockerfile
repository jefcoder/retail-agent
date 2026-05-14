FROM python:3.10-slim

# Install Docker CLI (for spawning sandbox containers via host socket)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://download.docker.com/linux/static/stable/$(uname -m)/docker-27.5.1.tgz \
       | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app

COPY docker/evaluate/pyproject.toml docker/evaluate/uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Pre-cache SentenceTransformer model (avoids runtime download in scoring)
ARG HF_TOKEN=""
RUN HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN=${HF_TOKEN} .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

COPY retailbench/ /app/retailbench/
COPY src/agent/ /app/src/agent/
COPY src/__init__.py /app/src/

COPY data/ /app/data/

RUN mkdir -p /app/logs

ENV PYTHONPATH=/app

ENTRYPOINT [".venv/bin/python", "-m", "retailbench.test_runner"]
