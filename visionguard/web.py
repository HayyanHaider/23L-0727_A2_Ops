"""Flask UI and authenticated model API; failures never silently fall back."""
import hmac
import io
import logging
import os
import time
import uuid

import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .database import Store
from .inference import InvalidImage, Predictor, decode_upload

LOG = logging.getLogger(__name__)


def create_app(config=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(PROJECT_NAME=os.getenv("PROJECT_NAME", "23L-0727_A2_Ops"),
                      MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_BYTES", "5242880")),
                      DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///visionguard.sqlite3"),
                      MODEL_PATH=os.getenv("MODEL_PATH", "models/champion.keras"),
                      MODEL_METADATA_PATH=os.getenv("MODEL_METADATA_PATH", "models/champion.json"),
                      MODEL_THRESHOLD=os.getenv("MODEL_THRESHOLD"),
                      INFERENCE_API_URL=os.getenv("INFERENCE_API_URL", "").rstrip("/"),
                      INFERENCE_API_KEY=os.getenv("INFERENCE_API_KEY", ""),
                      API_ONLY=os.getenv("API_ONLY", "false").lower() == "true",
                      DEBUG=False)
    if config:
        app.config.update(config)
    store = Store(app.config["DATABASE_URL"])
    predictor = app.config.get("PREDICTOR")
    if not app.config["INFERENCE_API_URL"] and predictor is None:
        predictor = Predictor(app.config["MODEL_PATH"], app.config["MODEL_METADATA_PATH"],
                              app.config["MODEL_THRESHOLD"])
    app.extensions.update(store=store, predictor=predictor)

    @app.before_request
    def protect_api_deployment():
        if app.config["API_ONLY"] and request.path != "/health":
            expected = app.config["INFERENCE_API_KEY"]
            if not expected or not hmac.compare_digest(request.headers.get("X-API-Key", ""), expected):
                return jsonify(error="A valid API key is required."), 401

    def error(message, status):
        if request.path.startswith("/api/") or request.path == "/health":
            return jsonify(error=message), status
        return render_template("error.html", message=message, status=status), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error("The upload is too large. Choose a file smaller than 5 MB.", 413)

    @app.errorhandler(InvalidImage)
    def invalid_image(exc):
        return error(str(exc), 400)

    @app.errorhandler(404)
    def missing(_):
        return error("That page does not exist.", 404)

    @app.errorhandler(Exception)
    def unexpected(exc):
        LOG.exception("request_failed: %s", type(exc).__name__)
        return error("The service is temporarily unavailable. Please try again later.", 503)

    @app.get("/health")
    def health():
        store.check()
        return jsonify(status="ok", project=app.config["PROJECT_NAME"],
                       mode="remote_api" if app.config["INFERENCE_API_URL"] else "resident_model",
                       model_version=predictor.version if predictor else "remote",
                       debug=bool(app.debug), database="ready")

    @app.get("/")
    def index():
        return render_template("upload.html")

    @app.get("/history")
    def history():
        return render_template("history.html", rows=store.recent())

    @app.post("/predict")
    @app.post("/api/predict")
    def predict():
        if request.path.startswith("/api/"):
            expected = app.config["INFERENCE_API_KEY"]
            if not expected or not hmac.compare_digest(request.headers.get("X-API-Key", ""), expected):
                return error("A valid API key is required.", 401)
        started = time.perf_counter()
        upload = request.files.get("image")
        if not upload:
            return error("Choose an image to upload.", 400)
        filename = secure_filename(upload.filename or "")[:255]
        data = upload.read()
        pixels = decode_upload(data, filename)
        if app.config["INFERENCE_API_URL"]:
            response = requests.post(app.config["INFERENCE_API_URL"] + "/api/predict",
                                     headers={"X-API-Key": app.config["INFERENCE_API_KEY"]},
                                     files={"image": (filename, io.BytesIO(data), upload.mimetype)}, timeout=(10, 90))
            response.raise_for_status()
            result = response.json()
            for key in ("label", "ai_score", "confidence", "threshold", "model_version", "model_sha256"):
                if key not in result:
                    raise ValueError("Incomplete model API response")
        else:
            result = predictor.predict(pixels)
        request_id = str(uuid.uuid4())
        latency = round((time.perf_counter() - started) * 1000, 2)
        store.add(request_id=request_id, filename=filename,
                  **{key: result[key] for key in ("label", "ai_score", "confidence", "threshold", "model_version", "model_sha256")},
                  latency_ms=latency)
        result.update(request_id=request_id, filename=filename, latency_ms=latency)
        if request.path.startswith("/api/"):
            return jsonify(result)
        return render_template("result.html", result=result)

    return app
