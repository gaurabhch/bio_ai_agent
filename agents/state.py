# agents/state.py
# Shared data object that travels through every LangGraph node.
# Every node reads from it and writes results back into it.
# Build this first — every other Pipeline 2 file imports from it.

from typing import TypedDict, Optional


class AgentState(TypedDict):

    # ── INPUT ─────────────────────────────────────────────────────────────────
    # Assembled by FastAPI in Step 6. Never modified after assembly.

    messages:             list[dict]   # last 10 messages from Redis
    user_id:              str
    use_case:             str          # pcos | pregnancy | period_issues
    user_tags:            list[str]    # e.g. ['pcos_user', 'high_stress']
    user_health_summary:  dict         # recent symptom log summary
    user_memory:          dict         # long-term preferences and history
    conversation_id:      str

    # ── SUPERVISOR WRITES (Step 8) ────────────────────────────────────────────
    rewritten_query:      str          # clean, searchable version of message
    next_agent:           str          # which specialist to route to
    routing_reason:       str          # keyword | llm_fallback | crisis
    response_mode:        str          # information | emotional | clarification | crisis
    is_crisis:            bool
    is_flagged:           bool
    flag_reason:          Optional[str]

    # ── SPECIALIST AGENT WRITES (Step 9) ─────────────────────────────────────
    retrieved_context:    list[dict]   # top 5 chunks from PGVector search
    raw_response:         str          # unverified Gemini output
    agent_node_used:      str
    confidence_score:     float

    # ── PUBMED VERIFIER WRITES (Step 10) ─────────────────────────────────────
    verified_response:    str
    citations:            list[str]    # PubMed article URLs

    # ── MERGE AGENT WRITES (Step 11) ─────────────────────────────────────────
    final_response:       str          # ready to stream to Flutter
    sources:              list[str]    # all citation + reference URLs
