import base64
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

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}
REQUEST_TIMEOUT_SECONDS = 30.0

if not OPENAI_API_KEY:
    print(
        "ERROR: OPENAI_API_KEY is not set.\n",
        file=sys.stderr,
    )
    sys.exit(1)

client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)

app = FastAPI(title="Bina AI Service")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    detected_class: str | None = Form(None)
    ) -> dict:
    """
    Receives a single snapshot image (never video/clips — data minimization
    is an architectural decision) and optionally the class detected by the
    C++ pipeline, and returns a short text description.
    """
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG or PNG images are accepted.",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB).")

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
                                "url": f"data:{image.content_type};base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="AI provider timed out.")
    except APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the AI provider.")
    except AuthenticationError:
        raise HTTPException(status_code=502, detail="AI provider rejected the credentials.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI provider rate limit hit.")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"AI provider error: {e}")

    description = (response.choices[0].message.content or "").strip()
    return {"description": description}