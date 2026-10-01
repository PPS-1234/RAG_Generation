"""
Pluggable embedding backends behind one interface. This is the key design
decision for "works with different document sets/models without code changes":
swapping the embedding model is an environment variable, never a code edit.

  EMBEDDER=tfidf                 (default) - scikit-learn TF-IDF, zero downloads,
                                  works fully offline, good enough for keyword-
                                  heavy technical docs.
  EMBEDDER=sentence-transformers - BAAI/all-MiniLM-L6-v2 via `sentence-transformers`,
                                  genuine semantic embeddings, needs one-time
                                  model download (~80MB) and internet access.

Both implement the same three methods, so pipeline.py never knows or cares
which one is active.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    dim: int

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Called once per collection, on the full corpus, before first embed."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> bool:
        """Returns True if a saved state was found and loaded."""
        return False


class TfidfEmbedder(BaseEmbedder):
    """
    Default backend. TF-IDF vectors are sparse and don't capture deep semantics
    the way a transformer embedding does, but they're deterministic, need zero
    downloads, and are genuinely effective for technical documents where the
    right keywords (function names, error codes, config keys) matter a lot.

    Honest limitation to state in an interview: it won't match a paraphrased
    query to a passage that uses completely different words for the same idea -
    that's exactly why EMBEDDER=sentence-transformers exists as a drop-in upgrade.
    """

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.dim = max_features
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True
        self.dim = len(self.vectorizer.get_feature_names_out())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embedding")
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> bool:
        fpath = os.path.join(path, "tfidf_vectorizer.pkl")
        if not os.path.exists(fpath):
            return False
        with open(fpath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True
        self.dim = len(self.vectorizer.get_feature_names_out())
        return True


class SentenceTransformerEmbedder(BaseEmbedder):
    """Optional upgrade path - real semantic embeddings, needs internet on first run."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> None:
        pass  # pretrained, nothing to fit

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def get_embedder() -> BaseEmbedder:
    """Factory: reads EMBEDDER env var, defaults to tfidf. This is the ONE place
    that knows about backend names - everything else uses BaseEmbedder."""
    backend = os.getenv("EMBEDDER", "tfidf").lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBEDDER backend: {backend}")
