# VisionGuard deployment and verification

Hayyan Haider · 23L-0727

## Local containers

1. Copy `.env.example` to `.env`. Set fresh `POSTGRES_PASSWORD` and `INFERENCE_API_KEY` values; use the same password in `DATABASE_URL` if running without Compose. Compose supplies its own service URL.
2. Copy `models/champion.keras` and `models/champion.json` from the submission, or run `python -m scripts.provision_model` after installing `requirements-serve.txt`.
3. Leave `INFERENCE_API_URL` empty for resident inference. Run `docker compose up --build -d`. The tested Mac installation uses the equivalent standalone `docker-compose` command.
4. Open http://localhost:8000. Check `/health`, upload an image, then inspect `/history`.
5. `docker compose down` preserves the named database volume. `down -v` deletes it; avoid that when preserving records.

The demonstrated stack uses PostgreSQL 17 and a read-only model mount. The database has no host port. The web process runs as UID 10001. The verification host used `WEB_PORT=8002` because a standalone service occupied port 8000.

## Cloud API

- Codespace: `shiny-memory-g4467vr6gq7pc97pw`
- Workspace: https://shiny-memory-g4467vr6gq7pc97pw.github.dev/
- API base: https://shiny-memory-g4467vr6gq7pc97pw-8001.app.github.dev
- Health: `GET /health`
- Prediction: `POST /api/predict`, multipart field `image`, header `X-API-Key`

Use the repository's Python 3.11 dev container. Install `requirements-serve.txt`, then run `python -m scripts.provision_model`. Configure `INFERENCE_API_KEY` through a Codespaces secret. Run `python -m scripts.start_cloud`; set port 8001's backend protocol to HTTP and visibility to Public in the Ports panel. HTTPS is supplied by the external forwarded address. All routes except health require the key in API-only mode.

Codespaces sleeps after inactivity. Resume it before demonstration/grading, restart the API process if needed, and set port visibility to Public again. This is an on-demand host; no permanent uptime is claimed. The two-core machine was chosen within the account's included allowance. Stop it when it is no longer needed to conserve usage.

For local-to-cloud mode set `INFERENCE_API_URL` to the API base above and `INFERENCE_API_KEY` to the matching private key, then restart Gunicorn or recreate the web container. The local process does not load a model in this mode. Never commit or submit the real environment file. A grading key should be shared privately through the instructor's permitted mechanism.

## Postman / external network

Import `postman/VisionGuard.postman_collection.json` and provide `api_url` and `api_key` variables privately. Alternatively install Newman, set `API_URL` and `INFERENCE_API_KEY`, and run `node scripts/run_postman.cjs`. The Public API verification GitHub workflow runs this same collection on a separate GitHub-hosted network. Its sanitized artifact contains statuses, predictions, assertion counts and timing; no authentication header is exported.

## Tracking provenance

The original Colab champion and threshold were selected before the original test pass. Historical imports are tagged as imports. Six later live manual/autolog training runs use training and validation only and never replace that champion. The original release and registry version 2 have exactly equal weights (262 arrays); reserialization can change their file SHA-256. Deployment verifies the release file SHA-256.

The DVC dataset remote is a permitted local folder on the verification host. Dataset bytes are not included in the submission ZIP or claimed as DagsHub remote storage. Download the referenced dataset, rebuild using the saved audit/splits, and configure a local remote path for a fresh machine. The tiny DVC lab includes both metadata commits and data objects for recovery.
