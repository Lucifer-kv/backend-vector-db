import cohere
import os
from ..core.config import settings

cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)

def generate_embedding(text: str) -> list[float]:
    try:
        response = cohere_client.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
        embedding = response.embeddings[0]
        if not isinstance(embedding, list):
            raise ValueError("Invalid embedding format returned by Cohere API")
        return embedding
    except Exception as e:
        raise ValueError(f"Error generating embedding: {str(e)}")