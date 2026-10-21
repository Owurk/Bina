import base64
import io
import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from PIL import Image, UnidentifiedImageError

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 16_000_000
REQUEST_TIMEOUT_SECONDS = 30.0

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


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    detected_class: str | None = Form(None),
) -> dict:
    """
    Receives a single snapshot image (never video/clips — data minimization
    is an architectural decision) and optionally the class detected by the
    C++ pipeline, and returns a short text description.
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
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the vision-analysis component of Bina, a home "
                        "security camera system. You receive a single webcam "
                        "snapshot. Describe in 2-3 short, factual sentences what "
                        "is happening in the image."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this snapshot." + class_hint},
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

    return {"description": description}