# agents/specialist_agent.py
# Generic specialist agent — one class handles ALL domains.
# Domain is passed as a parameter from config so no duplication exists.
#
# 4-step workflow per agent:
#   A — Embed the rewritten query (same model as Pipeline 1 — CRITICAL)
#   B — PGVector retrieval (filtered by domain + pcos_general)
#   C — Assemble 5-component prompt
#   D — Groq generation → raw_response

from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from agents.state import AgentState
from retrieval.searcher import retrieve_chunks
from utils.prompt_builder import build_system_prompt, assemble_prompt
from config import GROQ_MODEL, GROQ_TIMEOUT, GROQ_TIMEOUT_MESSAGE


class SpecialistAgent:
    """
    Generic RAG specialist agent.

    Usage (in graph.py):
        pcos_mental_health_agent = SpecialistAgent(domain="pcos_mental_health", embedder=embedder)
        pcos_nutrition_agent     = SpecialistAgent(domain="pcos_nutrition",      embedder=embedder)
        # one instance per domain, all sharing the same embedder

    The embedder MUST be the exact same SentenceTransformer model used in
    Pipeline 1 (controlled by EMBEDDING_MODEL in config.py).
    """

    def __init__(
        self,
        domain  : str,
        embedder: SentenceTransformer,
    ) -> None:
        self.domain         = domain
        self.embedder       = embedder
        self._system_prompt = build_system_prompt(domain)

    async def run(
        self,
        state     : AgentState,
        groq_client: AsyncGroq,
        db_session : AsyncSession,
    ) -> AgentState:
        """
        Execute the 4-step RAG workflow and return an updated AgentState.
        """
        # ── Step A: Embed the rewritten query ───────────────────────────────
        # CRITICAL: Must use the exact same model as Pipeline 1 Step 4.
        query_embedding: list[float] = self._embed_query(state["rewritten_query"])

        # ── Step B: PGVector retrieval ───────────────────────────────────────
        chunks: list[dict] = await retrieve_chunks(
            query_embedding=query_embedding,
            domain=self.domain,
            session=db_session,
        )
        confidence = self._compute_confidence(chunks)

        # ── Step C: Prompt assembly ──────────────────────────────────────────
        user_prompt = assemble_prompt(state, chunks)

        # ── Step D: Groq generation ──────────────────────────────────────────
        raw_response = await self._generate(user_prompt, groq_client)

        return {
            **state,
            "retrieved_context": chunks,
            "raw_response"     : raw_response,
            "agent_node_used"  : f"{self.domain}_agent",
            "confidence_score" : confidence,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string using the locked embedding model.
        normalize_embeddings=True is non-negotiable — matches Pipeline 1.
        """
        vectors = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()

    async def _generate(
        self,
        user_prompt: str,
        groq_client: AsyncGroq,
    ) -> str:
        """
        Send the assembled prompt to Groq and return the text response.
        Returns a safe timeout message if Groq does not respond in time.
        """
        try:
            response = await groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=500,
                timeout=GROQ_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return GROQ_TIMEOUT_MESSAGE

    def _compute_confidence(self, chunks: list[dict]) -> float:
        """
        Simple confidence score from the top chunk similarity.
        0.0 → no chunks returned.  1.0 → perfect cosine similarity.
        """
        if not chunks:
            return 0.0
        return round(float(chunks[0].get("similarity", 0.0)), 4)
