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


def _json_content(title: str, description: str, severity: str) -> str:
    import json

    return json.dumps({"title": title, "description": description, "severity": severity})





def test_structured_json_is_returned_as_three_fields(client, monkeypatch):
    """Happy path: a well-formed JSON model reply becomes the three fields."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content(
                            "Person at the front door",
                            "A person is standing at the front door, facing the entrance.",
                            "medium",
                        )
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "title": "Person at the front door",
        "description": "A person is standing at the front door, facing the entrance.",
        "severity": "medium",
    }


def test_severity_is_always_lowercase(client, monkeypatch):
    """The model must not be able to smuggle 'HIGH'/'High' into the response."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content("T", "Someone is climbing the fence.", "HIGH")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["severity"] == "high"


def test_json_wrapped_in_fences_or_prose_still_parses(client, monkeypatch):
    """Markdown fences and chatty lead-ins must not break parsing (no fallback)."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "```json\n"
                            '{"title": "Cat on the porch", '
                            '"description": "A cat is sitting on the porch railing.", '
                            '"severity": "low"}\n'
                            "```"
                        )
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "title": "Cat on the porch",
        "description": "A cat is sitting on the porch railing.",
        "severity": "low",
    }


def test_invalid_severity_falls_back_to_medium(client, monkeypatch):
    """'critical' is not an accepted level — the event must stay usable."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content("Broken window", "A window is shattered.", "critical")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity"] == "medium"
    assert body["title"] == "Broken window"
    assert body["description"] == "A window is shattered."


def test_missing_title_is_derived_from_description(client, monkeypatch):
    """A JSON reply without a title must still produce all three fields."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content("", "A person is walking across the yard.", "low")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "A person is walking across the yard"
    assert body["severity"] == "low"


def test_overlong_title_is_clamped_to_ten_words(client, monkeypatch):
    """The 10-word title cap is enforced in code, not just in the prompt."""

    long_title = "one two three four five six seven eight nine ten eleven twelve"

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content(long_title, "A person is visible.", "low")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "one two three four five six seven eight nine ten"





def test_non_json_output_falls_back_to_plain_text(client, monkeypatch):
    """Fallback: free text (old BIN-008 behaviour) must still return three fields."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "A person is walking across the backyard carrying a "
                            "ladder towards the shed."
                        )
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"title", "description", "severity"}
    assert (
        body["description"]
        == "A person is walking across the backyard carrying a ladder towards the shed."
    )
    # Title = first MAX_TITLE_WORDS words of the description.
    assert body["title"] == "A person is walking across the backyard carrying a ladder"
    assert body["severity"] == "medium"


def test_truncated_json_falls_back_to_plain_text(client, monkeypatch):
    """Regression: JSON cut off by the token cap must fall back, not 500."""

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"title": "Person at door", "description": "A person is standing at'
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"title", "description", "severity"}
    assert body["description"].startswith('{"title": "Person at door"')
    assert body["severity"] == "medium"


def test_prompt_asks_for_json_and_includes_detected_class(client, monkeypatch):
    """BIN-009 step 1 + BIN-008 acceptance: JSON instructions and class hint."""

    captured = {}

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content("t", "A person is visible.", "low")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "image/png")},
        data={"detected_class": "person"},
    )
    assert resp.status_code == 200
    system_prompt = captured["messages"][0]["content"]
    user_text = captured["messages"][1]["content"][0]["text"]
    assert "JSON" in system_prompt
    assert '"title"' in system_prompt
    assert '"description"' in system_prompt
    assert '"severity"' in system_prompt
    assert "low" in system_prompt and "medium" in system_prompt and "high" in system_prompt
    assert "person" in user_text
    assert captured["max_tokens"] == main.ANALYSIS_MAX_TOKENS





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
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_json_content("Red square", "A solid red square.", "low")
                    )
                )
            ]
        )

    _patch_client(monkeypatch, fake_create)
    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", _png_bytes(), "application/octet-stream")},
        data={"detected_class": "person"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "title": "Red square",
        "description": "A solid red square.",
        "severity": "low",
    }


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


def test_oversized_chunked_upload_without_content_length_returns_413(client: TestClient):
    """
    Regression: an oversized multipart body streamed WITHOUT a Content-Length
    header (chunked transfer) must be cut off by the ASGI streaming body
    limit while the stream is consumed — 413 before multipart parsing
    completes, independent of the (absent) Content-Length header.
    """
    boundary = "----bina-test-boundary"
    part = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="big.jpg"\r\n'
        "Content-Type: image/jpeg\r\n"
        "\r\n"
    ).encode() + b"\x00" * (main.MAX_BODY_BYTES + 1024) + (
        f"\r\n--{boundary}--\r\n"
    ).encode()
    resp = client.post(
        "/analyze",
        content=iter([part]),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert "content-length" not in resp.request.headers
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Request body too large."


def test_oversized_image_dimensions_rejected(client: TestClient):
    """
    Regression: decompression-bomb guard — an image whose header declares
    more than MAX_IMAGE_PIXELS is rejected with 413 straight from the
    header, before any pixel is decoded (the file itself is tiny on disk).
    """
    buf = io.BytesIO()
    Image.new("L", (6000, 4000)).save(buf, format="PNG")
    big_dims = buf.getvalue()
    assert len(big_dims) < main.MAX_IMAGE_BYTES

    resp = client.post(
        "/analyze",
        files={"image": ("snap.png", big_dims, "image/png")},
    )
    assert resp.status_code == 413
    assert "16 MP" in resp.json()["detail"]
