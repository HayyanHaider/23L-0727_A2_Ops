"""Resolve champion once, export a verified local Keras file, classify one image."""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from PIL import Image, ImageOps


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--export", default="models")
    args = parser.parse_args()
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias("23L-0727_A2_Ops", "champion")
    model = mlflow.tensorflow.load_model("models:/23L-0727_A2_Ops@champion")
    dest = Path(args.export); dest.mkdir(exist_ok=True)
    model.save(dest / "champion.keras")
    digest = hashlib.sha256((dest / "champion.keras").read_bytes()).hexdigest()
    threshold = 0.6806868314743042
    metadata = dict(project="23L-0727_A2_Ops", version=version.version, threshold=threshold,
                    sha256=digest, registry_uri="models:/23L-0727_A2_Ops@champion", run_id=version.run_id)
    (dest / "champion.json").write_text(json.dumps(metadata, indent=2))
    import numpy as np
    from visionguard.inference import Predictor
    with Image.open(args.image) as im:
        pixels = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    print(json.dumps(Predictor(dest / "champion.keras", dest / "champion.json").predict(pixels), indent=2))


if __name__ == "__main__":
    main()
