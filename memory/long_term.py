# memory/long_term.py
# Long-term user memory stored in PostgreSQL.
# Records previous symptoms, medication history, recurring concerns,
# and personalization preferences across all conversations.

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class LongTermMemory:
    """Reads and writes long-term user memory from PostgreSQL."""

    def __init__(self, session: AsyncSession = None):
        self.session = session

    # ── Read methods ──────────────────────────────────────────────────────────

    async def get_user_memory(
        self,
        user_id: str,
    ) -> dict:
        result = await self.session.execute(
            text(
                "SELECT memory_data FROM user_memory "
                "WHERE user_id = :uid LIMIT 1"
            ),
            {"uid": user_id},
        )
        row = result.fetchone()
        return row[0] if row else {}

    async def get_user_profile(
        self,
        user_id: str,
    ) -> dict:
        result = await self.session.execute(
            text(
                "SELECT use_case, user_tags, health_summary "
                "FROM user_profiles WHERE user_id = :uid LIMIT 1"
            ),
            {"uid": user_id},
        )
        row = result.fetchone()

        if row is None:
            return {
                "use_case":       "pcos",
                "user_tags":      [],
                "health_summary": {},
            }

        return {
            "use_case":       row[0],
            "user_tags":      row[1] or [],
            "health_summary": row[2] or {},
        }

    # ── Write methods ─────────────────────────────────────────────────────────

    async def update_user_memory(
        self,
        user_id: str,
        new_memory: dict,
    ) -> None:
        """
        Upsert the user's long-term memory record.
        Called after streaming finishes — the user never waits for this.
        """
        await self.session.execute(
            text(
                "INSERT INTO user_memory (user_id, memory_data, updated_at) "
                "VALUES (:user_id, :memory_data, NOW()) "
                "ON CONFLICT (user_id) DO UPDATE "
                "SET memory_data = :memory_data, updated_at = NOW()"
            ),
            {"user_id": user_id, "memory_data": new_memory},
        )
        await self.session.commit()

    async def save_chat_message(
        self,
        state: dict,
        session: AsyncSession = None,
    ) -> None:
        """
        Persist both the user message and the AI response to chat_messages.
        Also updates the conversation's last_message_at timestamp.
        Called after streaming finishes.
        """
        # Allow passing an external session for persist_after_stream use
        db = session or self.session

        await db.execute(
            text(
                "INSERT INTO chat_messages "
                "(conversation_id, user_id, user_message, ai_response, "
                " agent_node_used, confidence_score, sources, "
                " is_flagged, flag_reason, created_at) "
                "VALUES "
                "(:conv_id, :user_id, :user_msg, :ai_resp, "
                " :agent_node, :confidence, :sources, "
                " :is_flagged, :flag_reason, NOW())"
            ),
            {
                "conv_id":    state["conversation_id"],
                "user_id":    state["user_id"],
                "user_msg":   state["messages"][-1]["text"],
                "ai_resp":    state["final_response"],
                "agent_node": state.get("agent_node_used", ""),
                "confidence": state.get("confidence_score", 0.0),
                "sources":    state.get("sources", []),
                "is_flagged": state.get("is_flagged", False),
                "flag_reason": state.get("flag_reason"),
            },
        )

        await db.execute(
            text(
                "UPDATE chat_conversations "
                "SET last_message_at = NOW() "
                "WHERE id = :conv_id"
            ),
            {"conv_id": state["conversation_id"]},
        )
        await db.commit()


# Module-level singleton — session injected per request via LongTermMemory(session)
long_term_memory = LongTermMemory()