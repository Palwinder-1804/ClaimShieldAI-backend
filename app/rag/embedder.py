import logging
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class MockEmbeddings(Embeddings):
    """
    Mock embeddings generator that generates deterministic 1536-dimension vectors.
    Used for local execution when no OpenAI API Key is provided.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # text-embedding-3-small has 1536 dimensions
        dimension = 1536
        results = []
        for text in texts:
            # Generate deterministic values based on text hash/length to help test queries
            seed = sum(ord(c) for c in text[:100]) / 10000.0
            vector = [seed + float(i) / 10000.0 for i in range(dimension)]
            # Normalize vector (optional but good practice)
            norm = sum(x**2 for x in vector)**0.5
            normalized_vector = [x / norm for x in vector]
            results.append(normalized_vector)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class EmbeddingsProvider:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key and "your-openai-api-key" not in self.api_key:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=self.api_key
                )
                logger.info("OpenAIEmbeddings (text-embedding-3-small) successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAIEmbeddings: {e}. Using MockEmbeddings fallback.")
                self.embeddings = MockEmbeddings()
        else:
            logger.warning("OPENAI_API_KEY not configured. Using MockEmbeddings fallback.")
            self.embeddings = MockEmbeddings()

    def get_embeddings(self) -> Embeddings:
        """
        Returns the active langchain Embeddings provider.
        """
        return self.embeddings

embeddings_provider = EmbeddingsProvider()
