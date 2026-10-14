import base64
import io
import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

app = FastAPI(title="Bina AI Service")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _validated_content_type(data: bytes) -> str:
    """
    Decode the upload with Pillow and verify it is a real JPEG or PNG.

    The client-provided Content-Type is never trusted (it can be forged),
    so this is the authoritative image check. Returns the canonical
    content type used for the provider request.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            if fmt not in ("JPEG", "PNG"):
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG or PNG images are accepted.",
                )
            img.verify()
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
    data = await image.read()
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

    description = (response.choices[0].message.content or "").strip()
    return {"description": description}