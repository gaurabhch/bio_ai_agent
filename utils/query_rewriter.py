# utils/query_rewriter.py
# Rewrites vague, emotional, or follow-up messages into clean searchable queries.
# Uses Groq with the last 3 conversation exchanges as context.

from groq import AsyncGroq
from config import GROQ_MODEL, GROQ_TIMEOUT

REWRITE_SYSTEM_PROMPT = """You are a medical query rewriter for a PCOS health assistant.

Your job:
- Rewrite the user's message into a short, clear, searchable query.
- Use conversation history to resolve pronouns and references ("that", "it", "this").
- Translate Hinglish or emotional language into plain English medical terms.
- Keep it concise — one sentence, under 15 words.
- If the message is too vague to rewrite meaningfully, return exactly: CLARIFICATION_NEEDED

Examples:
"I've been so bloated and tired lately"      → "PCOS symptoms bloating fatigue"
"And how does that affect my weight?"        → "How does insulin resistance affect weight in PCOS"
"Mujhe fatigue aur irregular periods hain"  → "PCOS fatigue irregular periods symptoms"
"I don't feel like myself lately"            → CLARIFICATION_NEEDED

Return ONLY the rewritten query or CLARIFICATION_NEEDED. No explanation."""


async def rewrite_query(
    message    : str,
    history    : list[dict],
    use_case   : str,
    groq_client: AsyncGroq,
) -> str:
    """
    Rewrites the user message into a clean searchable query.

    Args:
        message    : Raw user message.
        history    : Last 6 messages (3 exchanges) from AgentState.messages.
        use_case   : User's health context (e.g. 'pcos').
        groq_client: Shared AsyncGroq client instance.

    Returns:
        Rewritten query string, or 'CLARIFICATION_NEEDED'.
    """
    history_text = _format_history(history)
    prompt = (
        f"Health context: {use_case}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"User message: {message}\n\n"
        f"Rewrite:"
    )

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=50,
            timeout=GROQ_TIMEOUT,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else message
    except Exception:
        return message                # fallback — pipeline never stops


def _format_history(history: list[dict]) -> str:
    """Format last N messages into a readable conversation block."""
    if not history:
        return "(no previous messages)"
    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)
