# retrieval/searcher.py
# PGVector cosine similarity search against pcos_kb_chunks (Pipeline 1 table).
# Returns chunk dicts with keys matching what prompt_builder.py and merge_agent.py expect:
#   chunk_text, field_type, topic, similarity, reference_sources, cluster_id
#
# CRITICAL: query_embedding must be produced by the same model as Pipeline 1
#           (all-MiniLM-L6-v2, normalize_embeddings=True)

import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import TOP_K_RETRIEVAL, NEON_TABLE_NAME


async def retrieve_chunks(
    query_embedding: list[float],
    domain: str,
    session: AsyncSession,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Retrieve the most semantically similar knowledge chunks for a query.

    Args:
        query_embedding : 384-dim vector from all-MiniLM-L6-v2 (normalized).
        domain          : Target specialist domain (e.g. 'pcos_mental_health').
                          Always also returns pcos_general rows as a fallback.
        session         : Async SQLAlchemy session (injected by graph.py).
        top_k           : Number of chunks to return (default from config).

    Returns:
        List of dicts with keys:
            chunk_text, field_type, topic, similarity,
            reference_sources (list), cluster_id
    """
    embedding_str = str(query_embedding)   # pgvector expects "[0.1, 0.2, ...]"

    result = await session.execute(
        text(
            f"SELECT "
            f"  COALESCE(l2_text, text)        AS chunk_text, "
            f"  section_name                   AS field_type, "
            f"  section_name                   AS topic, "
            f"  cluster_id, "
            f"  reference_sources, "
            f"  1 - (embedding <=> CAST(:query_emb AS vector)) AS similarity "
            f"FROM {NEON_TABLE_NAME} "
            f"WHERE category = :domain OR category = :general "
            f"ORDER BY embedding <=> CAST(:query_emb AS vector) "
            f"LIMIT :top_k"
        ),
        {
            "query_emb": embedding_str,
            "domain"   : domain,
            "general"  : "pcos_general",
            "top_k"    : top_k,
        },
    )
    rows = result.fetchall()
    return [_map_row(row) for row in rows]


def _map_row(row) -> dict:
    """
    Map a DB row to the dict shape expected by:
      - prompt_builder.build_chunks_block()  → chunk_text, field_type, topic, similarity
      - merge_agent.merge_agent_node()       → reference_sources (must be a list)
      - specialist_agent._compute_confidence → similarity
    """
    raw_refs = row.reference_sources or "[]"
    if isinstance(raw_refs, str):
        try:
            refs = json.loads(raw_refs)
        except (ValueError, TypeError):
            refs = []
    else:
        refs = list(raw_refs)

    return {
        "chunk_text"       : row.chunk_text or "",
        "field_type"       : row.field_type or "content",
        "topic"            : row.topic or "",
        "cluster_id"       : row.cluster_id or "",
        "similarity"       : float(row.similarity or 0.0),
        "reference_sources": refs,
    }
