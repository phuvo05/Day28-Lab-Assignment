# Lab 28 Local-Runnable Submission Design

## Goal

Prepare the Lab 28 platform artifacts so the submission can be demonstrated and graded locally, while still supporting the guide's real Kaggle/ngrok integrations when credentials and URLs are provided.

## Architecture

The implementation keeps the existing Lab 28 structure:

- Docker Compose runs Kafka, Prefect, Redis-backed Feast simulation, Qdrant, Prometheus, Grafana, and the FastAPI API Gateway.
- Kaggle-hosted vLLM and embedding services remain optional external integrations.
- Local deterministic fallback behavior is added only at external boundaries so the platform can run without secrets or live tunnels.

## Components

- `api-gateway`: exposes `/health`, `/api/v1/chat`, and Prometheus metrics. It calls vLLM when `VLLM_NGROK_URL` is configured, otherwise returns a deterministic demo answer.
- `scripts/05_embed_to_qdrant.py`: calls the embedding service when `EMBED_NGROK_URL` is configured, otherwise generates deterministic 384-dimensional embeddings locally.
- `scripts/09_verify_observability.py`: verifies Prometheus and treats LangSmith as optional unless a key is configured.
- `scripts/production_readiness_check.py`: reports a readiness score from local services without failing on optional external services.
- `smoke-tests/test_e2e.py`: validates local happy path, health, Qdrant, Prometheus/Grafana, failure behavior, and Redis feature data.

## Data Flow

1. `scripts/01_ingest_to_kafka.py` sends sample documents to Kafka topic `data.raw`.
2. Prefect flow consumes records and writes parquet files under `delta-lake/raw`.
3. `scripts/03_delta_to_feast.py` loads parquet records and writes feature entries to Redis.
4. `scripts/05_embed_to_qdrant.py` embeds sample records and upserts points into Qdrant.
5. API Gateway receives chat requests, searches Qdrant, builds context, and either calls real vLLM or returns a local demo response.
6. Prometheus scrapes API Gateway metrics for Grafana dashboards and readiness checks.

## Error Handling

- Missing request fields return client errors instead of server crashes.
- Qdrant search failures do not prevent a demo chat response; the response includes best-effort context when available.
- vLLM and embedding service failures fall back to local deterministic behavior only when external URLs are absent or unavailable.
- Optional LangSmith verification is skipped when no key is configured.

## Testing and Submission Review

The implementation should support these local verification commands:

- `docker compose up -d --build`
- `python scripts/01_ingest_to_kafka.py`
- `python scripts/05_embed_to_qdrant.py`
- `python scripts/03_delta_to_feast.py`
- `pytest smoke-tests/ -v`
- `python scripts/production_readiness_check.py`

The expected submission outcome is a runnable `lab28`-style repository with source code, smoke test evidence, production readiness output above 80%, and documentation matching `SUBMISSION.md`.
