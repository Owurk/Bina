import base64
import io
import json
import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 16_000_000
REQUEST_TIMEOUT_SECONDS = 30.0


SEVERITY_LEVELS = ("low", "medium", "high")

FALLBACK_SEVERITY = "medium"
MAX_TITLE_WORDS = 10

ANALYSIS_MAX_TOKENS = 300

ANALYSIS_SYSTEM_PROMPT = (
    'You are the vision-analysis component of Bina, a home security camera '
    'system. You receive a single webcam snapshot. Respond with ONLY a JSON '
    'object - no prose, no markdown fences - with exactly these keys:\n'
    '- "title": a short factual label for the event, at most 10 words\n'
    '- "description": what is happening in the image, in 1-3 short factual '
    'sentences\n'
    '- "severity": how concerning the scene is, one of "low", "medium" or '
    '"high"\n'
    'Use "low" for an empty or calm scene, "medium" for a person or animal '
    'doing something ordinary near the property, and "high" for suspicious '
    'or threatening behaviour such as someone forcing a door or window.\n'
    'Example: {"title": "Person at the front door", "description": "A person '
    'is standing at the front door, facing the entrance.", "severity": '
    '"medium"}'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bina.ai")

if not OPENAI_API_KEY:
    print(
        "ERROR: OPENAI_API_KEY is not set.\n",
        file=sys.stderr,
    )
    sys.exit(1)

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=REQUEST_TIMEOUT_SECONDS,
)

print(
    f"AI endpoint: {OPENAI_BASE_URL or 'https://api.openai.com/v1 (official)'}"
    f" | model: {OPENAI_MODEL}"
)


class BodyLimitMiddleware:
    """
    Pure-ASGI middleware enforcing MAX_BODY_BYTES on the request body while
    the ASGI stream is consumed — independent of the Content-Length header,
    which may be absent (chunked uploads) or simply lie.

    Three layers:
    1. Fast-fail: a declared Content-Length above the cap is rejected
       with 413 before a single byte is read.
    2. Streaming cap: body bytes are counted as they flow through the
       ASGI receive channel; once the cap is exceeded, the stream is cut
       off, so the multipart parser never spools the rest of the body.
    3. Clean rejection: the error response the app produces for the
       cut-off stream (a 400 from the aborted multipart parse) is
       suppressed and replaced with a 413 — the endpoint never runs.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False
        eof = {"type": "http.request", "body": b"", "more_body": False}

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return eof
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    return eof
            return message

        async def guarded_send(message: Message) -> None:
            if exceeded:
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body too large."},
        )
        await response(scope, receive, send)


app = FastAPI(title="Bina AI Service")
app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _validated_content_type(data: bytes) -> str:
    """
    The client-provided Content-Type is never trusted (it can be forged),
    so this is the authoritative image check.

    Decompression-bomb protection — two independent caps:
    - MAX_IMAGE_BYTES bounds the compressed wire payload (enforced by the
      caller before this function runs);
    - MAX_IMAGE_PIXELS bounds the decoded raster. It is checked straight
      from the image headers, before img.load() runs, so an image with
      absurd dimensions never gets its pixels allocated at all.

    Returns the canonical content type used for the provider request.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            if fmt not in ("JPEG", "PNG"):
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG or PNG images are accepted.",
                )
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large (max {MAX_IMAGE_PIXELS // 1_000_000} MP).",
                )
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.warning("Image decode failed (size=%d bytes).", len(data))
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as an image.",
        ) from None

    return f"image/{fmt.lower()}"


def _extract_json_object(text: str) -> dict | None:
    """
    Best-effort extraction of a JSON object from raw model output.

    Models regularly wrap JSON in markdown fences or lead in with a short
    sentence ("Here is the analysis: ..."), so the raw text is never
    trusted as-is: the substring from the first '{' to the last '}' is
    what gets parsed. Anything else — no braces, truncated JSON, a JSON
    array or scalar — yields None, and the caller applies the fallback.
    """
    if not text:
        return None

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _clamped_title(title: str, max_words: int = MAX_TITLE_WORDS) -> str:
    """
    Whitespace-normalize a title and clamp it to MAX_TITLE_WORDS, so the
    notification UI never receives an over-long headline even when the
    model ignores the "at most 10 words" instruction.
    """
    words = " ".join(title.split()).split()
    return " ".join(words[:max_words])


def _parse_ai_analysis(raw: str) -> dict | None:
    """
    Parse the model response into the structured event fields (BIN-009):
    {"title": str, "description": str, "severity": "low"|"medium"|"high"}.

    Returns None only when the raw output contains no usable description
    at all — the endpoint then reports a provider failure (502) instead of
    inventing an event.

    Fallback contract ("keep a fallback if parsing fails"):
    - Valid JSON object: fields are used directly. An unknown, missing or
      non-string severity becomes FALLBACK_SEVERITY; a missing or empty
      title is derived from the description; the title is clamped to
      MAX_TITLE_WORDS either way.
    - Non-JSON output (old free-text behaviour, truncated JSON, chatty
      prose): the whole text becomes the description, the title is
      derived from its first words and the severity stays FALLBACK_SEVERITY.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    description = raw
    severity = FALLBACK_SEVERITY
    title: str | None = None

    parsed = _extract_json_object(raw)
    if parsed is not None:
        json_title = parsed.get("title")
        if isinstance(json_title, str) and json_title.strip():
            title = json_title.strip()

        json_description = parsed.get("description")
        if isinstance(json_description, str) and json_description.strip():
            description = json_description.strip()

        json_severity = parsed.get("severity")
        if isinstance(json_severity, str):
            candidate = json_severity.strip().lower()
            if candidate in SEVERITY_LEVELS:
                severity = candidate

    if title is None:

        title = description.rstrip()
        if title.endswith("."):
            title = title[:-1]

    return {
        "title": _clamped_title(title),
        "description": description,
        "severity": severity,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    detected_class: str | None = Form(None),
) -> dict:
    """
    Receives a single snapshot image (never video/clips — data minimization
    is an architectural decision) and optionally the class detected by the
    C++ pipeline, and returns a structured analysis:

        {"title": str, "description": str, "severity": "low"|"medium"|"high"}

    If the model's output is not valid JSON, the plain text becomes the
    description and title/severity are derived/defaulted (BIN-009 fallback).
    """
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB).")

    content_type = _validated_content_type(data)
    b64_image = base64.b64encode(data).decode("ascii")

    class_hint = ""
    if detected_class and detected_class.strip():
        class_hint = (
            f"\nA lightweight motion-based detector classified this frame as "
            f"'{detected_class.strip()}'. Take that into account, but describe "
            f"what you actually see."
        )

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=ANALYSIS_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": ANALYSIS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this snapshot and reply with the JSON object."
                            + class_hint,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
        )
    except APITimeoutError as e:
        logger.warning("AI provider timed out after %.0fs.", REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="AI provider timed out.") from e
    except APIConnectionError as e:
        logger.warning("Could not reach AI provider: %s", e)
        raise HTTPException(status_code=502, detail="Could not reach the AI provider.") from e
    except AuthenticationError as e:
        logger.error("AI provider rejected credentials: %s", getattr(e, "body", e))
        raise HTTPException(status_code=502, detail="AI provider rejected the credentials.") from e
    except RateLimitError as e:
        logger.warning("AI provider rate limit hit: %s", getattr(e, "body", e))
        raise HTTPException(status_code=429, detail="AI provider rate limit hit.") from e
    except APIError as e:
        logger.error("AI provider error: %s | body=%s", e, getattr(e, "body", None))
        raise HTTPException(status_code=502, detail="AI provider error.") from e

    if not response.choices or response.choices[0].message is None:
        logger.error("AI provider returned no usable choices.")
        raise HTTPException(
            status_code=502,
            detail="AI provider returned no usable result.",
        )

    description = (response.choices[0].message.content or "").strip()
    if not description:
        logger.error(
            "AI provider returned an empty description (finish_reason=%s).",
            getattr(response.choices[0], "finish_reason", "unknown"),
        )
        raise HTTPException(
            status_code=502,
            detail="AI provider returned no usable result.",
        )

    analysis = _parse_ai_analysis(description)
    if analysis is None:
        logger.error(
            "AI analysis parsing produced nothing usable (finish_reason=%s).",
            getattr(response.choices[0], "finish_reason", "unknown"),
        )
        raise HTTPException(
            status_code=502,
            detail="AI provider returned no usable result.",
        )

    return analysis
