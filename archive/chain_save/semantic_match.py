from typing import List
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def embed_text(text: str, embed_model: Embeddings) -> np.ndarray:
    return np.array(embed_model.embed_query(text))


def compute_similarity_score(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity([a], [b])[0][0]


# This module previously implemented custom semantic matching logic.
# Now, use Chroma's built-in similarity search for vector-based retrieval.
# You can keep utility functions here if needed for other purposes.

def embed_text(text: str, embed_model) -> list:
    """Utility: Get embedding for a text using the provided embedding model."""
    return embed_model.embed_query(text)

# If you need to do any custom filtering or scoring on top of Chroma results,
# you can add those functions here.

# For document retrieval, use:
# results = vectorstore.similarity_search_by_vector(query_embedding, k=5)
