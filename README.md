# 23L-0727_A2_Ops - VisionGuard

[![VisionGuard CI](https://github.com/HayyanHaider/23L-0727_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/HayyanHaider/23L-0727_A2_Ops/actions/workflows/ci.yml)

AI-versus-real image classification, reproducible training, MLflow/DagsHub tracking, a Flask UI and model API, PostgreSQL history, Docker Compose, CI, and cloud deployment.

Name: Hayyan Haider. Roll number: 23L-0727.

## Quick start

Copy `.env.example` to `.env` and set fresh passwords and an API key. Provision the champion model into `models/` from the submission archive or model registry. Run `docker compose up --build -d` and open http://localhost:8000. The PostgreSQL database stays on the internal Compose network and stores records in a named volume.

For local Flask connected to the cloud API, set `INFERENCE_API_URL` and the matching `INFERENCE_API_KEY`, then run `gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 2 wsgi:app`. `MODEL_PATH`, threshold, database URL, and API settings are environment variables. Debug mode is disabled.

## Repository and evidence

- GitHub: https://github.com/HayyanHaider/23L-0727_A2_Ops
- DagsHub: https://dagshub.com/HayyanHaider/23L-0727_A2_Ops
- Training: `python -m scripts.run_training`
- Tracking tool labs: `python -m scripts.track_experiments --help`
- Registry provisioning: `python -m scripts.predict_by_alias IMAGE`
- Tests: `pytest -q`; lint: see `.github/workflows/ci.yml`.

The original T4 model comparison and threshold were locked before the original held-out test evaluation. Later tracking demonstrations use validation data and do not change that champion. Historical MLflow imports are tagged explicitly and are distinguished from live manual-callback and autologged runs.

Image bytes are decoded and processed in memory. Only sanitized filenames, UTC timestamps, decisions, scores, model identities, and latency are stored. The detector can make mistakes; scores are not calibrated proof of authenticity.

The PDF report records the completed verification and deployment evidence. Credentials, runtime databases, dataset bytes, and caches are excluded from Git. Sources are cited in the report.

## Submission and live demonstration

Submit the PDF and `23L-0727_A2_Submission.zip` from `submission/`. The ZIP includes source, model metadata and champion, executed notebook, and supporting evidence. See [deployment instructions](docs/DEPLOYMENT.md) for fresh local setup and cloud restart.

The verified cloud API base is https://shiny-memory-g4467vr6gq7pc97pw-8001.app.github.dev. Predictions require the private assignment key; health is public. Codespaces can sleep, so resume it and restore public port 8001 before grading. The external Postman collection passed in [GitHub Actions](https://github.com/HayyanHaider/23L-0727_A2_Ops/actions/runs/42303357317). Both valid demonstration inputs reached the model; the AI-labelled example was classified as Real, which is a retained model error rather than a transport failure.
