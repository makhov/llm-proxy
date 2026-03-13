from __future__ import annotations

from app.config import Settings
from app.rag import embedder, vector_store


class RAGRetriever:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def retrieve_context(
        self,
        query: str,
        filters: dict | None = None,
    ) -> tuple[str, int]:
        """
        Returns (context_string, num_chunks_found).
        context_string is empty if nothing relevant was found.
        """
        if not query.strip():
            return "", 0

        query_embedding = embedder.embed_one(query)
        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=self._settings.rag_top_k,
            where=filters,
        )

        # Filter by relevance threshold (cosine distance: lower = more similar)
        threshold = self._settings.rag_score_threshold
        relevant = [r for r in results if r.distance <= threshold]

        if not relevant:
            return "", 0

        context = self._settings.rag_context_prefix
        chunks = []
        for r in relevant:
            source = r.metadata.get("source", "unknown")
            title = r.metadata.get("title", source)
            chunk_idx = r.metadata.get("chunk_index", 0)
            chunks.append(f"[Source: {title}, chunk {chunk_idx}]\n{r.text}")

        context += self._settings.rag_context_separator.join(chunks)
        return context, len(relevant)


_retriever: RAGRetriever | None = None


def init_retriever(settings: Settings) -> RAGRetriever:
    global _retriever
    _retriever = RAGRetriever(settings)
    return _retriever


def get_retriever() -> RAGRetriever:
    if _retriever is None:
        raise RuntimeError("RAGRetriever not initialized")
    return _retriever
