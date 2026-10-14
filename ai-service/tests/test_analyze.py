import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIError
from PIL import Image

import main


@pytest.fixture()
def client():
    return TestClient(main.app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_missing_image_is_rejected(client: TestClient):
    assert client.post("/analyze").status_code == 422


def test_forged_mime_type_is_rejected(client: TestClient):
    """Regression: a non-image uploaded with an image Content-Type must 400."""
    resp = client.post(
        "/analyze",
        files={"image": ("evil.jpg", b"this is definitely not an image", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_valid_image_with_wrong_declared_mime_succeeds(client: TestClient, monkeypatch):
    """Validation is content-based: a real PNG declared as octet-stream passes."""

    async def fake_create(*args, **kwargs):
        url = kwargs["messages"][1]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;") 
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A red square."))]
        )

    _patch_client(monkeypatch, fake_create)

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "application/octet-stream")},
        data={"detected_class": "person"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "A red square."}


def test_provider_error_details_are_not_leaked(client: TestClient, monkeypatch):
    """Regression: internal provider details must never appear in the response."""

    async def failing_create(*args, **kwargs):
        raise APIError(
            "INTERNAL: upstream secret-proxy.internal failed",
            request=httpx.Request("POST", "https://provider/v1/chat/completions"),
            body={"error": {"message": "internal-secret-detail"}},
        )

    _patch_client(monkeypatch, failing_create)

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 502
    assert "internal-secret-detail" not in resp.text
    assert "secret-proxy" not in resp.text


def _patch_client(monkeypatch, create_impl):
    """Replace the module-level OpenAI client with a stub whose
    chat.completions.create delegates to create_impl."""

    class _Completions:
        async def create(self, *args, **kwargs):
            return await create_impl(*args, **kwargs)

    class _Chat:
        completions = _Completions()

    class _StubClient:
        chat = _Chat()

    monkeypatch.setattr(main, "client", _StubClient())