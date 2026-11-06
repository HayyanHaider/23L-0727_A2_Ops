"""Start the model API with a local persistent database (or managed PostgreSQL).

For Codespaces, export the required environment variables first, then forward 8001.
"""
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
if not os.getenv("INFERENCE_API_KEY"):
    raise SystemExit("Set INFERENCE_API_KEY before starting the cloud service")
os.environ["API_ONLY"] = "true"
os.environ["INFERENCE_API_URL"] = ""
os.environ.setdefault("DATABASE_URL", "sqlite:////workspaces/23L-0727_A2_Ops/cloud-history.sqlite3")
if not Path(os.getenv("MODEL_PATH", "models/champion.keras")).is_file():
    raise SystemExit("Provision the champion model before startup (registry/release asset).")
subprocess.run(["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "1", "--threads", "2",
                "--timeout", "120", "--access-logfile", "-", "wsgi:app"], check=True)
