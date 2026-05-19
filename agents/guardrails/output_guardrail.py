# agents/guardrails/output_guardrail.py
#
# OUTPUT GUARDRAIL NODE — last node before END in the LangGraph pipeline.
# Runs after merge_agent (or after a blocked input_guardrail response).
#
# Checks (in strict order):
#   1. Definitive diagnosis language softening  (regex — no LLM, deterministic)
#   2. Citation presence verification
#   3. Tone check  (LLM — only for high-sensitivity domains)
#   4. Disclaimer injection  (hardcoded append — always last)

import re
from groq import AsyncGroq
from agents.state import AgentState
from config import GROQ_MODEL, GROQ_TIMEOUT

# ── Standard disclaimer ───────────────────────────────────────────────────────

_DISCLAIMER = (
    "\n\n---\n"
    "⚕️ *This information is for educational purposes only. "
    "Please consult a qualified healthcare provider for personal medical advice.*"
)

# ── Diagnosis language softening map ─────────────────────────────────────────
# (regex pattern → replacement)
# All replacements are softer, non-diagnostic equivalents.

_DIAGNOSIS_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\byou have PCOS\b",                    "these symptoms may be associated with PCOS"),
    (r"\byou have PCOD\b",                    "these symptoms may be associated with PCOD"),
    (r"\byou (are|have been) diagnosed with\b","it may be worth discussing with your doctor about"),
    (r"\bthis confirms you have\b",            "this may suggest"),
    (r"\bthis indicates you have\b",           "this may be associated with"),
    (r"\byou definitely have\b",               "you may want to discuss with your doctor about"),
    (r"\byou are suffering from\b",            "you may be experiencing symptoms of"),
    (r"\bthis proves you have\b",              "this could be a sign of"),
    (r"\byou are diabetic\b",                  "these signs may be associated with insulin resistance"),
    (r"\byou are infertile\b",                 "there may be fertility-related factors worth discussing with a specialist"),
]

def _soften_diagnosis_language(text: str) -> tuple[str, bool]:
    """Returns (processed_text, was_anything_changed)."""
    changed = False
    for pattern, replacement in _DIAGNOSIS_REPLACEMENTS:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            changed = True
            text = new_text
    return text, changed


# ── Citation presence check ───────────────────────────────────────────────────

# Clinical claim markers — if any are present in the response, a citation is expected.
_CLINICAL_CLAIM_MARKERS = [
    r"\d+%",                            # percentage figures
    r"\bstud(y|ies)\b",
    r"\bresearch\b",
    r"\baccording to\b",
    r"\bevidence\b",
    r"\bclinical\b",
    r"\btrial\b",
    r"\bpublished\b",
]

_CITATION_DISCLAIMER = (
    " *(Please verify this information with a healthcare provider or "
    "refer to published medical literature.)*"
)

def _check_citations(response: str, sources: list[str]) -> tuple[str, bool]:
    """
    If the response contains a clinical claim but no citations were attached,
    inject a soft inline disclaimer rather than blocking the entire response.
    Returns (processed_text, citations_ok).
    """
    has_clinical_claim = any(
        re.search(p, response, re.IGNORECASE) for p in _CLINICAL_CLAIM_MARKERS
    )
    citations_present = bool(sources)

    if has_clinical_claim and not citations_present:
        # Inject disclaimer at end of first sentence containing a clinical marker
        sentences = response.split(". ")
        new_sentences = []
        injected = False
        for sent in sentences:
            if not injected and any(re.search(p, sent, re.IGNORECASE) for p in _CLINICAL_CLAIM_MARKERS):
                new_sentences.append(sent + _CITATION_DISCLAIMER)
                injected = True
            else:
                new_sentences.append(sent)
        return ". ".join(new_sentences), False  # False = citations not fully verified

    return response, True


# ── Tone check ────────────────────────────────────────────────────────────────

# Only run for high-sensitivity domains — saves LLM cost on factual domains.
_TONE_CHECK_DOMAINS = {"pcos_mental_health", "pcos_fertility", "pcos_symptoms"}

async def _tone_check(
    response: str,
    agent_node_used: str,
    groq_client: AsyncGroq,
) -> str:
    """
    Runs only if agent_node_used domain is in _TONE_CHECK_DOMAINS.
    Asks Groq to rewrite response if tone is cold, clinical, or judgmental.
    Returns the (possibly rewritten) response.
    """
    domain = agent_node_used.replace("_agent", "")
    if domain not in _TONE_CHECK_DOMAINS:
        return response

    prompt = (
        "You are a tone reviewer for a women's health chatbot. "
        "The response below must feel warm, empathetic, and supportive — never cold, "
        "clinical, or judgmental. If the tone is already warm, return it unchanged. "
        "If not, rewrite it to be warmer while keeping all medical facts intact. "
        "Do not add new information. Do not remove citations.\n\n"
        f"Response:\n{response}\n\n"
        "Return only the final response text."
    )
    try:
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            timeout=GROQ_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return response  # fail-safe: return original on error


# ── Main node ─────────────────────────────────────────────────────────────────

async def output_guardrail_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    """
    LangGraph node — final checkpoint before END.
    Always runs regardless of whether input_guardrail blocked or passed.
    Reads state['final_response'] and writes back a safe, compliant version.
    """
    response = state.get("final_response", "")
    sources   = state.get("sources", [])
    agent_used = state.get("agent_node_used", "")

    # Blocked responses (crisis / off-topic / injection) skip checks 1-3,
    # but still receive the disclaimer.
    guardrail_passed = state.get("guardrail_passed", True)

    diagnosis_softened  = False
    citations_verified  = True

    if guardrail_passed:
        # ── CHECK 1: Soften diagnosis language ───────────────────────────────
        response, diagnosis_softened = _soften_diagnosis_language(response)

        # ── CHECK 2: Citation presence ───────────────────────────────────────
        response, citations_verified = _check_citations(response, sources)

        # ── CHECK 3: Tone check (selective) ──────────────────────────────────
        response = await _tone_check(response, agent_used, groq_client)

    # ── CHECK 4: Disclaimer injection — ALWAYS, on every response ────────────
    response = response + _DISCLAIMER

    return {
        **state,
        "final_response":    response,
        "diagnosis_softened": diagnosis_softened,
        "disclaimer_injected": True,
        "citations_verified":  citations_verified,
    }
