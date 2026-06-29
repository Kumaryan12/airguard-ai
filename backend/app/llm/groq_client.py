import os

from dotenv import load_dotenv
from groq import Groq

from ml.config import PROJECT_ROOT


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to .env as GROQ_API_KEY=..."
        )

    return Groq(api_key=api_key)


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")