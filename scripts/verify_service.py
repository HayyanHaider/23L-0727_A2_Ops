"""Exercise actual HTTP uploads and preserve responses without recording API keys."""
import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--out", default="outputs/full_assignment/evidence/http-local")
    p.add_argument("--api", action="store_true")
    args = p.parse_args()
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    samples = Path("samples"); samples.mkdir(exist_ok=True)
    frame = pd.read_csv("outputs/23L-0727_A2_Ops/data_evidence/train.csv")
    provenance = []
    for label, name in [(0, "known_real.png"), (1, "known_ai.png")]:
        row = frame[frame.label == label].iloc[0]
        path = Path("data/processed") / row.path
        # Small PNG copies for demonstrations; never part of final metric calculation.
        with Image.open(path) as image:
            image.thumbnail((1000, 1000))
            image.save(samples / name)
        provenance.append(dict(filename=name, source_path=row.path, folder_label=label,
                               sample_sha256=hashlib.sha256((samples / name).read_bytes()).hexdigest(), split="train"))
    (samples / "provenance.json").write_text(json.dumps(provenance, indent=2))
    key = dotenv_values(".env")["INFERENCE_API_KEY"]
    headers = {"X-API-Key": key} if args.api else {}
    result = dict(project="23L-0727_A2_Ops", timestamp=datetime.now(timezone.utc).isoformat(),
                  base_url=args.base_url, health=requests.get(args.base_url + "/health", timeout=15).json(), cases=[])
    endpoint = "/api/predict" if args.api else "/predict"
    for name, data, expected in [("known_real.png", (samples / "known_real.png").read_bytes(), 200),
                                 ("known_ai.png", (samples / "known_ai.png").read_bytes(), 200),
                                 ("wrong.exe", b"invalid", 400),
                                 ("corrupt.png", b"not an image", 400),
                                 ("oversize.png", b"x" * 5_242_881, 413)]:
        started = time.perf_counter()
        response = requests.post(args.base_url + endpoint,
                                 files={"image": (name, io.BytesIO(data))}, headers=headers, timeout=120)
        assert response.status_code == expected, (name, response.status_code, response.text[:200])
        suffix = "json" if args.api else "html"
        (dest / f"{name}.{suffix}").write_text(response.text)
        item = dict(filename=name, status=response.status_code, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        if args.api:
            item["response"] = response.json()
        result["cases"].append(item)
    history = requests.get(args.base_url + "/history", headers=headers, timeout=15)
    if not args.api:
        assert "known_ai.png" in history.text and "known_real.png" in history.text
        (dest / "history.html").write_text(history.text)
    (dest / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
