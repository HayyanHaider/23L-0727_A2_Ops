"""One resident model; the same EXIF, RGB and TensorFlow letterboxing as training."""
import hashlib
import io
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


class InvalidImage(ValueError):
    pass


def decode_upload(data, filename):
    if not filename or "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        raise InvalidImage("Choose a JPG, PNG, or WebP image.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImage("The file content does not match a supported image format.")
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                raise InvalidImage("This image has too many pixels. Use at most 25 megapixels.")
            return np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("This file cannot be decoded as a valid image.") from exc


class Predictor:
    def __init__(self, model_path, metadata_path, threshold=None):
        import tensorflow as tf
        self.tf = tf
        self.metadata = json.loads(Path(metadata_path).read_text())
        self.threshold = float(threshold if threshold is not None else self.metadata["threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("MODEL_THRESHOLD must be in [0, 1]")
        self.version = str(self.metadata["version"])
        self.sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if self.metadata.get("sha256") and self.sha256 != self.metadata["sha256"]:
            raise ValueError("Model checksum does not match champion metadata")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.size = int(self.model.input_shape[1])
        self.lock = threading.Lock()
        self.model(tf.zeros((1, self.size, self.size, 3)), training=False)

    def predict(self, pixels):
        image = self.tf.image.resize_with_pad(pixels, self.size, self.size,
                                             method="bilinear", antialias=True)
        with self.lock:
            p_ai = float(self.model(self.tf.expand_dims(image, 0), training=False).numpy().ravel()[0])
        label = "AI-generated" if p_ai >= self.threshold else "Real"
        return {"label": label, "ai_score": p_ai,
                "confidence": p_ai if label == "AI-generated" else 1 - p_ai,
                "threshold": self.threshold, "model_version": self.version,
                "model_sha256": self.sha256,
                "caveat": "A model estimate, not proof of authenticity. Scores are not calibrated probabilities."}
