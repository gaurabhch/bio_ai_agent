# agents/supervisor.py
# First LangGraph node in Pipeline 2.
# Runs four sub-steps in strict order before routing to a specialist.
#
# Sub-step A — Crisis keyword check (pure string match, no LLM)
# Sub-step B — Query rewriting (Groq + conversation history)
# Sub-step C — Response mode detection
# Sub-step D — Routing to specialist (keyword scoring, Groq fallback)

from groq import AsyncGroq

from agents.state import AgentState
from utils.query_rewriter import rewrite_query
from config import (
    CRISIS_KEYWORDS,
    FALSE_POSITIVE_GUARD,
    HELPLINE_RESPONSE,
    EMOTIONAL_KEYWORDS,
    VAGUE_PATTERNS,
    DOMAIN_KEYWORDS,
    KEYWORD_ROUTING_THRESHOLD,
    RESPONSE_MODES,
    GROQ_MODEL,
    GROQ_TIMEOUT,
)

# ── Sub-step A helpers ────────────────────────────────────────────────────────

def _is_crisis(message: str) -> bool:
    """
    Returns True if the message contains a crisis keyword
    and is NOT a known false positive phrase.
    """
    lowered = message.lower()
    if any(fp in lowered for fp in FALSE_POSITIVE_GUARD):
        return False
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


# ── Sub-step C helpers ────────────────────────────────────────────────────────

def _detect_response_mode(message: str, rewritten: str) -> str:
    """
    Classifies the message into one of four response modes.
    Uses pure string/heuristic logic — no LLM call needed here.
    """
    if rewritten == "CLARIFICATION_NEEDED":
        return RESPONSE_MODES["CLARIFICATION"]

    lowered = message.lower()

    if any(kw in lowered for kw in EMOTIONAL_KEYWORDS):
        return RESPONSE_MODES["EMOTIONAL"]

    if any(p in lowered for p in VAGUE_PATTERNS):
        return RESPONSE_MODES["CLARIFICATION"]

    return RESPONSE_MODES["INFORMATION"]


# ── Sub-step D helpers ────────────────────────────────────────────────────────

def _score_domains(rewritten_query: str, use_case: str, user_tags: list[str]) -> dict[str, int]:
    """
    Counts domain keyword hits in the rewritten query.
    Higher score → more likely correct domain.
    """
    lowered = rewritten_query.lower()
    scores: dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in lowered)

    # Boost the user's primary use-case domain slightly
    primary = f"{use_case}_general"
    if primary in scores:
        scores[primary] += 1

    # Boost domains mentioned in user tags
    for tag in user_tags:
        if tag in scores:
            scores[tag] += 1

    return scores


async def _llm_classify_domain(
    rewritten_query: str,
    groq_client: AsyncGroq,
) -> str:
    """
    Fallback: asks Groq to classify the domain when keyword scoring is ambiguous.
    Returns only the domain string (e.g. 'pcos_mental_health').
    """
    domain_list = ", ".join(DOMAIN_KEYWORDS.keys())
    prompt = (
        f"Classify this health query into exactly ONE domain from the list below.\n"
        f"Domains: {domain_list}\n\n"
        f"Query: {rewritten_query}\n\n"
        f"Return only the domain string. No explanation."
    )
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            timeout=GROQ_TIMEOUT,
        )
        domain = response.choices[0].message.content.strip().lower()
        return domain if domain in DOMAIN_KEYWORDS else "pcos_general"

    except Exception:
        return "pcos_general"


# ── Main supervisor node ──────────────────────────────────────────────────────

async def supervisor_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    """
    LangGraph node — runs all four sub-steps and returns an updated AgentState.

    The graph reads 'next_agent' from the returned state to decide
    which specialist node runs next.
    """
    message = state["messages"][-1]["text"]

    # ── SUB-STEP A: Crisis check — always first, no exceptions ───────────────
    if _is_crisis(message):
        return {
            **state,
            "is_crisis":      True,
            "is_flagged":     True,
            "flag_reason":    "crisis_keyword",
            "final_response": HELPLINE_RESPONSE,
            "next_agent":     "crisis_end",
            "routing_reason": "crisis",
            "response_mode":  RESPONSE_MODES["CRISIS"],
            "rewritten_query": message,
        }

    # ── SUB-STEP B: Query rewriting ──────────────────────────────────────────
    history  = state["messages"][-6:]   # last 3 exchanges
    rewritten = await rewrite_query(message, history, state["use_case"], groq_client)
    state = {**state, "rewritten_query": rewritten}

    # ── SUB-STEP C: Response mode detection ─────────────────────────────────
    mode  = _detect_response_mode(message, rewritten)
    state = {**state, "response_mode": mode}

    # Clarification mode → skip retrieval, route to clarification node
    if mode == RESPONSE_MODES["CLARIFICATION"]:
        return {**state, "next_agent": "clarification_agent", "routing_reason": "clarification"}

    # ── SUB-STEP D: Domain routing ───────────────────────────────────────────
    domain_scores = _score_domains(rewritten, state["use_case"], state["user_tags"])
    best_domain   = max(domain_scores, key=domain_scores.get)
    best_score    = domain_scores[best_domain]

    if best_score > KEYWORD_ROUTING_THRESHOLD:
        routing_reason = "keyword"
    else:
        best_domain    = await _llm_classify_domain(rewritten, groq_client)
        routing_reason = "llm_fallback"

    return {
        **state,
        "next_agent":     f"{best_domain}_agent",
        "routing_reason": routing_reason,
        "is_crisis":      False,
        "is_flagged":     False,
        "flag_reason":    None,
    }
