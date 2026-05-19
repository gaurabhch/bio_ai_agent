# utils/prompt_builder.py
# Assembles the complete prompt sent to Groq in the specialist agent.
# Five components in strict order:
# 1. System prompt — agent role and hard constraints
# 2. User context — use_case, tags, health summary
# 3. Retrieved chunks — top 5 KB pieces (NO cluster ID labels in output)
# 4. Conversation history — last 3 exchanges for continuity
# 5. User message — original message (not rewritten)


def build_system_prompt(domain: str) -> str:
    """
    Returns the system prompt for the specialist domain.
    Sets the agent's role, tone, and hard constraints.
    """
    return (
        f"You are a compassionate and medically informed {domain.replace('_', ' ')} "
        f"specialist assistant for women with PCOS.\n\n"
        "Guidelines:\n"
        "- Be warm, empathetic, and non-judgmental.\n"
        "- Base your response strictly on the retrieved knowledge provided below.\n"
        "- Do NOT include any citation labels, cluster IDs, or reference tags like [PCOS-001] in your response.\n"
        "- For serious medical concerns, always recommend consulting a qualified doctor.\n"
        "- Never diagnose. Never prescribe specific medications or dosages.\n"
        "- Use plain, clear English. Avoid excessive jargon.\n"
        "- Always respond in full, well-structured paragraphs — never use bullet points or headers.\n"
        "- Keep the response focused and under 250 words unless the topic requires more.\n"
        "- If no knowledge was retrieved, say so clearly but still provide a warm, informative response based on general PCOS knowledge.\n"
    )


def build_user_context_block(state: dict) -> str:
    """Formats the user's profile context for inclusion in the prompt."""
    tags = ", ".join(state.get("user_tags", [])) or "none"
    summary = state.get("user_health_summary", {})
    memory = state.get("user_memory", {})

    lines = [
        "--- USER CONTEXT ---",
        f"Health focus: {state.get('use_case', 'pcos')}",
        f"User tags: {tags}",
    ]
    if summary:
        lines.append(f"Recent health summary: {summary}")
    if memory:
        lines.append(f"Long-term preferences: {memory}")
    lines.append("--- END USER CONTEXT ---")
    return "\n".join(lines)


def build_chunks_block(chunks: list[dict]) -> str:
    """
    Formats retrieved KB chunks for inclusion in the prompt.
    Cluster IDs intentionally excluded — model must NOT reference them in output.
    """
    if not chunks:
        return (
            "--- RETRIEVED KNOWLEDGE ---\n"
            "No specific knowledge retrieved. Respond using general PCOS knowledge "
            "and clearly state: 'Based on general knowledge — no specific research was retrieved for this query.'\n"
            "--- END RETRIEVED KNOWLEDGE ---"
        )

    lines = ["--- RETRIEVED KNOWLEDGE ---"]
    for i, chunk in enumerate(chunks, start=1):
        ftype = chunk.get("field_type", "content")
        topic = chunk.get("topic", "")
        text = chunk.get("chunk_text", "")
        similarity = chunk.get("similarity", 0.0)
        lines.append(
            f"[{i}] ({ftype} — {topic}) "
            f"[similarity: {similarity:.2f}]\n{text}"
        )
    lines.append("--- END RETRIEVED KNOWLEDGE ---")
    return "\n\n".join(lines)


def build_history_block(messages: list[dict]) -> str:
    """Formats the last 3 conversation exchanges for the prompt."""
    recent = messages[-6:]
    if not recent:
        return ""
    lines = ["--- CONVERSATION HISTORY ---"]
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)


def assemble_prompt(state: dict, chunks: list[dict]) -> str:
    """
    Assembles the complete user-turn prompt from all five components.
    The system prompt is passed separately to Groq's 'system' role message.

    Returns:
        The full user-turn string ready to be sent to Groq.
    """
    original_message = state["messages"][-1]["text"]

    parts = [
        build_user_context_block(state),
        build_chunks_block(chunks),
        build_history_block(state.get("messages", [])),
        f"--- USER MESSAGE ---\n{original_message}\n--- END MESSAGE ---",
        (
            "\nPlease respond to the user's message using the retrieved knowledge above. "
            "Write your response as warm, connected paragraphs. "
            "Do not use bullet points, headers, or any citation labels like [PCOS-001]. "
            "Just natural, flowing text."
        ),
    ]
    return "\n\n".join(filter(None, parts))