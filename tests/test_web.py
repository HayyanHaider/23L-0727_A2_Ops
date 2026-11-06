import io

import numpy as np
import pytest
from PIL import Image

from visionguard.web import create_app


class FakePredictor:
    version = "test-2"

    def predict(self, pixels):
        assert pixels.shape == (20, 40, 3)
        return dict(label="AI-generated", ai_score=0.8, confidence=0.8, threshold=0.68,
                    model_version=self.version, model_sha256="a" * 64, caveat="Test estimate")


@pytest.fixture
def app(tmp_path):
    return create_app(dict(TESTING=True, PREDICTOR=FakePredictor(),
                           DATABASE_URL="sqlite:///" + str(tmp_path / "history.sqlite3"),
                           INFERENCE_API_URL="", INFERENCE_API_KEY="test-key"))


def image_bytes():
    buff = io.BytesIO()
    Image.fromarray(np.zeros((20, 40, 3), dtype="uint8")).save(buff, format="PNG")
    buff.seek(0)
    return buff


def test_prediction_and_persistent_history(app):
    c = app.test_client()
    result = c.post("/api/predict", headers={"X-API-Key": "test-key"},
                    data={"image": (image_bytes(), "../../face.png")})
    assert result.status_code == 200
    assert result.json["filename"] == "face.png"
    assert result.json["model_version"] == "test-2"
    assert b"face.png" in c.get("/history").data
    assert len(app.extensions["store"].recent()) == 1


@pytest.mark.parametrize("name,content", [("image.exe", b"x"), ("fake.png", b"not an image")])
def test_invalid_upload_does_not_create_prediction(app, name, content):
    response = app.test_client().post("/predict", data={"image": (io.BytesIO(content), name)})
    assert response.status_code == 400
    assert not app.extensions["store"].recent()


def test_size_limit_and_missing_file(app):
    c = app.test_client()
    assert c.post("/predict").status_code == 400
    assert c.post("/predict", data={"image": (io.BytesIO(b"x" * 5_242_881), "big.png")}).status_code == 413


def test_api_auth_and_health(app):
    c = app.test_client()
    assert c.post("/api/predict").status_code == 401
    assert c.get("/health").json["debug"] is False
    assert c.get("/health").json["database"] == "ready"


def test_remote_api_failure_has_no_local_fallback(app, monkeypatch):
    import requests
    app.config["INFERENCE_API_URL"] = "https://inference.example"
    def fail(*args, **kwargs):
        raise requests.Timeout("timeout")
    monkeypatch.setattr(requests, "post", fail)
    response = app.test_client().post("/predict", data={"image": (image_bytes(), "ok.png")})
    assert response.status_code == 503
    assert not app.extensions["store"].recent()
