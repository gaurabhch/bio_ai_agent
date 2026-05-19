# agents/pubmed_verifier.py
# Cross-checks medical claims in raw_response against PubMed.
# Claims with evidence → citations collected as URLs only (NOT injected into response text).
# Claims with no evidence → language softened automatically.
# PubMed unavailable → raw_response passes through unchanged (fail silently).

import httpx
from groq import AsyncGroq

from agents.state import AgentState
from config import PUBMED_TIMEOUT, GROQ_MODEL, GROQ_TIMEOUT

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

CLAIM_EXTRACTION_PROMPT = """Extract specific, verifiable medical claims from the text below.
Return ONLY a JSON array of short claim strings.
Example: ["cortisol elevation worsens PCOS symptoms", "inositol improves insulin sensitivity"]

If there are no specific medical claims, return an empty array: []

Text:
{text}

Return ONLY the JSON array. No explanation."""

SOFTENING_PROMPT = """Rewrite the response below to soften this unverified claim:
Claim: "{claim}"

Rules:
- Change definitive language ("X causes Y") to tentative language ("research suggests X may affect Y").
- Keep all other sentences unchanged.
- Do NOT add any citation labels, source tags, or reference markers like [PCOS-001] into the text.
- Return only the full rewritten response text."""


# ── PubMed search ─────────────────────────────────────────────────────────────

def _search_pubmed(claim: str) -> list[str]:
    """
    Query PubMed for a medical claim.
    Returns up to 3 article URLs, or an empty list on any failure.
    Always fails silently — never blocks the pipeline.
    """
    try:
        r = httpx.get(
            f"{PUBMED_BASE}esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    claim,
                "retmax":  3,
                "retmode": "json",
            },
            timeout=PUBMED_TIMEOUT,
        )
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
        return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]

    except Exception:
        return []


# ── Claim extraction ──────────────────────────────────────────────────────────

async def _extract_claims(text: str, groq_client: AsyncGroq) -> list[str]:
    """
    Use Groq to extract verifiable medical claims from the response text.
    Returns an empty list if the call fails.
    """
    import json

    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            timeout=GROQ_TIMEOUT,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        return claims if isinstance(claims, list) else []

    except Exception:
        return []


# ── Claim softening ───────────────────────────────────────────────────────────

async def _soften_claim(
    response_text: str,
    claim: str,
    groq_client: AsyncGroq,
) -> str:
    """
    Ask Groq to rewrite the response with softened language for an unverified claim.
    Returns original response_text unchanged if the call fails.
    """
    prompt = SOFTENING_PROMPT.format(claim=claim, response=response_text)
    try:
        result = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
            timeout=GROQ_TIMEOUT,
        )
        return result.choices[0].message.content.strip()

    except Exception:
        return response_text


# ── Main verifier node ────────────────────────────────────────────────────────

async def pubmed_verifier_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    """
    LangGraph node — verifies medical claims and updates AgentState.

    Citations are collected as URLs only — they are NEVER injected
    into the response text as inline labels or tags.
    """
    raw_response = state.get("raw_response", "")

    if state.get("final_response"):
        return state

    claims = await _extract_claims(raw_response, groq_client)
    verified = raw_response
    citations: list[str] = []

    for claim in claims:
        urls = _search_pubmed(claim)
        if urls:
            citations.extend(urls)
        else:
            verified = await _soften_claim(verified, claim, groq_client)

    return {
        **state,
        "verified_response": verified,
        "citations": citations[:2],
    }