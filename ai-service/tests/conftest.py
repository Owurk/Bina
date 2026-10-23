import os

# The service exits at import time if OPENAI_API_KEY is missing; provide a
# dummy value so tests can import the module without a real .env present.
os.environ.setdefault("OPENAI_API_KEY", "test-key")