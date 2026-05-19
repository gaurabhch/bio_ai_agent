# agents/merge_agent.py
# Personalises the verified response based on the user's profile tags and response mode.
# One focused Groq call with a tight tone instruction.

from groq import AsyncGroq

from agents.state import AgentState
from config import GROQ_MODEL, GROQ_TIMEOUT, RESPONSE_MODES

# ── Tone rule definitions ─────────────────────────────────────────────────────

_TAG_TONE_RULES: dict[str, str] = {
    "high_stress"         : "Open with a warm, empathetic acknowledgment before any clinical information.",
    "goal:actively_manage": "End with one specific, concrete action she can take today.",
    "goal:learn_more"     : "Use an educational tone — explain the 'why' behind each recommendation.",
    "undiagnosed"         : "Use cautious language throughout and include at least one reminder to consult a doctor.",
}

_MODE_TONE_RULES: dict[str, str] = {
    RESPONSE_MODES["EMOTIONAL"]     : "Start with an empathetic acknowledgment paragraph before any medical content.",
    RESPONSE_MODES["CLARIFICATION"] : "Ask exactly one focused follow-up question. Include no medical content.",
    RESPONSE_MODES["INFORMATION"]   : "Be clear, warm, and factual.",
    RESPONSE_MODES["CRISIS"]        : "",
}


def _build_tone_prompt(user_tags: list[str], response_mode: str) -> str:
    rules: list[str] = []

    mode_rule = _MODE_TONE_RULES.get(response_mode)
    if mode_rule:
        rules.append(mode_rule)

    for tag in user_tags:
        if tag in _TAG_TONE_RULES:
            rules.append(_TAG_TONE_RULES[tag])

    tone_lines = "\n".join(f"- {r}" for r in rules) if rules else "- Be warm, clear, and supportive."

    return (
        "You are personalising a PCOS health response.\n\n"
        "Apply these tone adjustments:\n"
        f"{tone_lines}\n\n"
        "Rules:\n"
        "- Do not add new medical information.\n"
        "- Do NOT include any citation labels, cluster IDs, or reference tags like [PCOS-001] in the response.\n"
        "- Write in natural, flowing paragraphs only — no bullet points, no headers.\n"
        "- Keep the response under 300 words unless a clarification question is asked.\n"
        "- Return only the personalised response text. Nothing else."
    )


# ── Main merge node ───────────────────────────────────────────────────────────

async def merge_agent_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    """
    LangGraph node — personalises tone and assembles final_response + sources.
    Uses verified_response if available, falls back to raw_response.
    Crisis responses bypass tone personalisation (already set by supervisor).
    """
    if state.get("final_response"):
        return state                  # crisis path — already complete

    verified      = state.get("verified_response") or state.get("raw_response", "")
    citations     = state.get("citations", [])
    user_tags     = state.get("user_tags", [])
    response_mode = state.get("response_mode", RESPONSE_MODES["INFORMATION"])

    tone_prompt = _build_tone_prompt(user_tags, response_mode)
    user_prompt = f"Personalise this response for the user:\n\n{verified}"

    try:
        result = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": tone_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=600,
            timeout=GROQ_TIMEOUT,
        )
        final_response = result.choices[0].message.content.strip()
    except Exception:
        final_response = verified     # fallback — never return empty

    # ── Collect sources (3 KB + 2 PubMed) ──────────────
    kb_sources: list[str] = []
    for chunk in state.get("retrieved_context", []):
        refs = chunk.get("reference_sources") or []
        if isinstance(refs, list):
            kb_sources.extend(refs)
        elif isinstance(refs, str) and refs:
            kb_sources.append(refs)

    # Deduplicate while preserving order, then cap at exactly 3
    kb_sources = list(dict.fromkeys(kb_sources))[:3]

    # PubMed citations already capped at 2 in pubmed_verifier.py
    pubmed_citations = state.get("citations", [])[:2]

    # Final list: KB first, PubMed second — always max 5
    all_sources = kb_sources + pubmed_citations

    return {
        **state,
        "final_response": final_response,
        "sources"       : all_sources,   # max 5
    }