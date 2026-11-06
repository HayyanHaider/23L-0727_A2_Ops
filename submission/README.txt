VISIONGUARD - MLOPS ASSIGNMENT 2
Hayyan Haider | 23L-0727

SUBMIT:
1. 23L-0727_VisionGuard.pdf (complete Tasks 1-8 report)
2. 23L-0727_A2_Submission.zip (source, model, notebook and evidence)

PROJECT LINKS:
GitHub: https://github.com/HayyanHaider/23L-0727_A2_Ops
DagsHub: https://dagshub.com/HayyanHaider/23L-0727_A2_Ops
CI toy: https://github.com/HayyanHaider/23L-0727_A2_Ops-ci-toy
API: https://shiny-memory-g4467vr6gq7pc97pw-8001.app.github.dev

The API uses a private key. No real credentials are in this archive.
Codespaces can sleep. Resume the Codespace, start scripts.start_cloud if
needed, and restore public HTTP port 8001 before grading. See docs/DEPLOYMENT.md.

LOCAL QUICK START:
Copy .env.example to .env and set fresh passwords/API key.
Keep INFERENCE_API_URL empty for local model inference.
The ZIP includes models/champion.keras and models/champion.json.
Run docker compose up --build -d and open http://localhost:8000.
Do not use down -v if retaining database history.

DATA AND EVIDENCE:
Dataset bytes are obtained from the assignment's Kaggle source; they are not
bundled. Audits, splits and DVC pointers are supplied. The demonstrated full
dataset DVC remote was a permitted local folder on the verification host.
A tiny DVC history bundle and data objects support offline version recovery.
The original executed Colab notebook and selected checkpoints accompany the
saved metrics. Later live tracking demonstrations do not change the champion.
The original AI-labelled API sample was classified as Real; this real model
error is retained and explained. API success is not a new accuracy estimate.
Screenshots are genuine captures, cropped for readability without retouching.
