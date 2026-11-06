"""Download a fixed release artifact once, verify it, then use local inference."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


def main():
    base = os.getenv("MODEL_RELEASE_URL", "https://github.com/HayyanHaider/23L-0727_A2_Ops/releases/download/champion-v1")
    out = Path("models"); out.mkdir(exist_ok=True)
    with urlopen(base + "/champion.json", timeout=60) as response:
        metadata = json.load(response)
    with urlopen(base + "/champion.keras", timeout=120) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Downloaded model does not match the release SHA-256")
    temp = out / "champion.keras.download"
    temp.write_bytes(content)
    temp.replace(out / "champion.keras")
    (out / "champion.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(project=metadata["project"], model_version=metadata["version"], sha256=digest,
                          bytes=len(content), status="verified"), indent=2))


if __name__ == "__main__":
    main()
