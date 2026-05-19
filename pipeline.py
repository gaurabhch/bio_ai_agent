# ai_agent/pipeline.py
# ─────────────────────────────────────────────────────────────────
# End-to-end RAG pipeline for the BioCanvas PCOS chatbot.
#
# Flow:
#   user query
#     → HierarchicalRetriever.retrieve_chunks()   [NeonDB pgvector]
#     → build_context_string()                    [format for LLM]
#     → Groq LLM                                  [generate answer]
#     → return { answer, references }
# ─────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from retrieval.searcher import HierarchicalRetriever
from config import LLM_MODEL

client    = Groq()
retriever = HierarchicalRetriever()

SYSTEM_PROMPT = """You are an empathetic PCOS health assistant for BioCanvas.
Answer questions about PCOS/PCOD using only the provided context.
Be warm, clear, and medically accurate. Never diagnose — always
recommend consulting a doctor for personal medical decisions.
If the answer is not in the context, say so honestly."""


def answer(
    query   : str,
    domain  : str  = None,
    verbose : bool = False,
) -> dict:
    """
    Full RAG pipeline.

    Args:
        query   : raw user question
        domain  : optional domain filter e.g. "pcos_nutrition"
        verbose : print retrieved context blocks if True

    Returns:
        {
            "answer"     : str,          LLM response
            "references" : list[str],    unique source URLs
        }
    """
    # ── Step 1: Retrieve context ──────────────────────────────────
    contexts    = retriever.retrieve_chunks(query, domain=None)
    context_str = retriever.build_context_string(contexts)

    if verbose:
        print(f"\nRetrieved {len(contexts)} context blocks:")
        for ctx in contexts:
            print(f"  [{ctx.level}] {ctx.cluster_id} | "
                  f"{ctx.section_name[:35]:35s} | "
                  f"score={ctx.score:.3f} | {ctx.token_count} words")

    if not context_str.strip():
        return {
            "answer"    : "I couldn't find relevant information in the knowledge base for that question.",
            "references": [],
        }

    # ── Step 2: Build prompt ──────────────────────────────────────
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Context from PCOS Knowledge Base:\n\n{context_str}\n\n"
            f"Question: {query}\n\nAnswer:"
        )},
    ]

    # ── Step 3: Generate answer ───────────────────────────────────
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 600,
    )
    llm_answer = response.choices[0].message.content.strip()

    # ── Step 4: Collect unique references ────────────────────────
    seen = set()
    unique_refs = []
    for ctx in contexts:
        for url in ctx.reference_sources:
            if url and url not in seen:
                seen.add(url)
                unique_refs.append(url)

    return {
        "answer"    : llm_answer,
        "references": unique_refs,
    }
