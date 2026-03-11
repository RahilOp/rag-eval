import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )
    return _client

def get_embeddings(input_text: str) -> List[float]:
    """Fetch embeddings for the input text using the OpenAI client."""
    client = _get_client()
    emb = client.embeddings.create(
        model="BAAI/bge-m3",
        input=input_text,
    )
    return emb.data[0].embedding  # type: ignore
