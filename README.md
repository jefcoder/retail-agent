# RetailBench (local)

This tree is a **local shopping-agent benchmark** using Docker Compose. It runs agents in an isolated sandbox against [ShoppingBench](https://arxiv.org/abs/2508.04266)-style problems, scores outcomes against ground truth, and optionally judges reasoning quality via the same inference proxy the agent uses.

## Prerequisites

- Docker with Compose v2
- A non-empty Lucene index at `./indexes` (mounted read-only into `search-server` at `/app/indexes`)
- For agent runs: an **OpenRouter** API key (see [`.env.example`](.env.example))

## Quick start

From this directory (`retail-agent/`):

```bash
cp .env.example .env   # edit: set OPENROUTER_API_KEY
mkdir -p logs
docker compose up -d search-server proxy
docker compose --profile test build test
docker compose --profile test run --no-deps test --agent-file src/agent/agent_test.py
```

If `search-server` and `proxy` are already up from another clone on the same Docker networks (`retailbench-main`, `sandbox-network`), you can use `--no-deps` on `run test` so Compose does not recreate those services.

### Common flags (after the service name `test`)

- `--problem-file data/suites/problem_suite_v3.json` (default)
- `--skip-reasoning` — skip the extra LLM judge pass on trajectories
- `--max-workers`, `--timeout` — passed through to the sandbox

## Services

| Service        | Role |
|----------------|------|
| `search-server`| Product search API (needs `./indexes`) |
| `proxy`      | Routes `/search/*` to the search server and `/inference/*` to **OpenRouter** only |
| `test`       | Profile `test`: builds/runs the **evaluate** image; spawns sandbox containers via the host Docker socket |
| `sandbox`    | Profile `tools`: optional standalone sandbox (used by CI integration tests with `docker-compose.test.yml`) |

**Note:** Proxy images from this tree onward validate `/inference/*` against an OpenRouter model allowlist and require an OpenRouter-style API key (`Bearer sk-or-...`). Chutes is not supported here.

## Development

- **Python layout**: agent and scoring code under `src/agent/`; local runner and Docker helpers under `retailbench/`.
- **Unit tests** (host): from `retail-agent/`, sync the evaluate lockfile then run pytest:

  ```bash
  cd docker/evaluate && uv sync --frozen --no-install-project --python 3.10
  uv pip install --python .venv/bin/python pytest pytest-cov
  cd ../.. && docker/evaluate/.venv/bin/python -m pytest tests/ -v --ignore=tests/integration
  ```

- **Evaluate image**: [`docker/evaluate/Dockerfile`](docker/evaluate/Dockerfile) — rebuild after changing `retailbench/` or scoring code that is copied into the image (not bind-mounted in the default `test` service).

## Publishing images

Maintainers: see [`.github/workflows/publish-images.yml`](.github/workflows/publish-images.yml) for building and pushing Docker Hub images under `erenhex/retailbench-*` (`search-server`, `proxy`, `sandbox`, `evaluate`). Configure the `DOCKERHUB_TOKEN` repository secret for pushes.
