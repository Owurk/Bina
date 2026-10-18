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

def test_provider_no_choices_returns_502(client: TestClient, monkeypatch):
    """Regression: empty choices must be a controlled 502, not IndexError/500."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(choices=[])

    _patch_client(monkeypatch, fake_create)

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 502


def test_provider_none_content_returns_502(client: TestClient, monkeypatch):
    """Regression: None content must not become HTTP 200 with empty description."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

    _patch_client(monkeypatch, fake_create)

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 502


def test_provider_whitespace_content_returns_502(client: TestClient, monkeypatch):
    """Regression: whitespace-only content must not report success."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   \n\t"))]
        )

    _patch_client(monkeypatch, fake_create)

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 502


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




def _truncated_jpeg_bytes() -> bytes:
    """
    A JPEG whose header is intact but whose scan data is cut off at the
    tail. It survives Image.open() + verify() (the old, weaker check) but
    FAILS a full pixel decode (img.load()) — exactly the gap between
    structural verification and real decoding.
    """
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (30, 90, 200)).save(buf, format="JPEG")
    full = buf.getvalue()
    return full[:-16]


def test_truncated_image_is_rejected(client: TestClient):
    """
    Regression: a structurally-valid but truncated JPEG (passes verify(),
    fails a full pixel decode) must be rejected with 400 before it is ever
    forwarded to the AI provider.
    """
    truncated = _truncated_jpeg_bytes()


    with Image.open(io.BytesIO(truncated)) as img:
        assert img.format == "JPEG"
        img.verify()


    resp = client.post(
        "/analyze",
        files={"image": ("snap.jpg", truncated, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "decoded" in resp.json()["detail"]


def test_oversized_upload_returns_413(client: TestClient):
    """
    Regression: an upload above MAX_IMAGE_BYTES must be rejected with 413.
    The endpoint reads at most MAX_IMAGE_BYTES + 1 bytes, so an oversized
    payload is never fully materialized in memory.
    """
    oversized = b"\x00" * (main.MAX_IMAGE_BYTES + 1024)  # 5 MB + 1 KB
    resp = client.post(
        "/analyze",
        files={"image": ("snap.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_request_body_limit_rejects_before_parsing(client: TestClient):
    """
    Regression: bodies declaring a Content-Length above MAX_BODY_BYTES are
    rejected with 413 by the middleware before multipart parsing starts
    (chunked bodies without Content-Length are capped by the bounded read).
    """
    oversized = b"\x00" * (main.MAX_BODY_BYTES + 1024)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Request body too large."