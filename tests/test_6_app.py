"""手順6のテスト（app.py）。サーバーを立てずに POST /predict を叩く。

リポジトリ直下の model.pt / tokenizer.json（手順4の `python train.py` で作ったもの）を使う。
実行: pytest tests/test_6_app.py
"""

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import ROOT  # noqa: E402


@pytest.fixture(scope="module")
def app_module():
    if not (ROOT / "model.pt").exists():
        pytest.skip("model.pt が無い（先に python train.py）")
    here = os.getcwd()
    os.chdir(ROOT)
    try:
        import app
        yield app
    finally:
        os.chdir(here)


@pytest.fixture(scope="module")
def client(app_module):
    return TestClient(app_module.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_shape(client):
    r = client.post("/predict", json={"text": "a wonderful , moving and beautifully acted film"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"label", "confidence"}
    assert body["label"] in ["negative", "positive"]
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_is_the_trained_model(client):
    """学習済みの重みが読まれていること。はっきりした2文なら SST-2 で学習したモデルは外さない。"""
    assert client.post("/predict", json={"text": "a wonderful , moving and beautifully acted film"}).json()["label"] == "positive"
    assert client.post("/predict", json={"text": "a dull , boring and painfully bad movie"}).json()["label"] == "negative"


def test_the_served_model_is_int8(app_module):
    """app が配っているのは量子化したモデル（周2の新要素が本番に載っていること）。"""
    assert type(app_module.model.head.weight).__name__ not in ("Tensor", "Parameter")


def test_bad_requests(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": 123}).status_code == 422
    assert client.get("/predict").status_code == 405
