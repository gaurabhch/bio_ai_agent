# memory/short_term.py
# Redis conversation cache — stores last N messages per conversation.
# Reads in microseconds. 24-hour TTL.
# Falls back gracefully if Redis is unavailable (handled in main.py).

import json
import redis.asyncio as redis

from config import REDIS_URL, MAX_HISTORY_MSGS, CONVERSATION_TTL


class ShortTermMemory:
    """Async Redis client for per-conversation message history."""

    def __init__(self) -> None:
        self._client: redis.Redis = redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )

    def _key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:messages"

    async def get_history(self, conversation_id: str) -> list[dict]:
        """
        Return the last MAX_HISTORY_MSGS messages for this conversation.
        Returns an empty list if the key does not exist yet.
        """
        raw = await self._client.lrange(self._key(conversation_id), 0, -1)
        return [json.loads(msg) for msg in raw]

    async def append_message(
        self,
        conversation_id: str,
        role           : str,
        text           : str,
    ) -> None:
        """
        Append one message to the conversation list.
        Trims to MAX_HISTORY_MSGS and refreshes the TTL.
        """
        key     = self._key(conversation_id)
        message = json.dumps({"role": role, "text": text})

        pipe = self._client.pipeline()
        pipe.rpush(key, message)
        pipe.ltrim(key, -MAX_HISTORY_MSGS, -1)
        pipe.expire(key, CONVERSATION_TTL)
        await pipe.execute()

    async def clear(self, conversation_id: str) -> None:
        """Delete the conversation cache (e.g. when a session ends)."""
        await self._client.delete(self._key(conversation_id))

    async def close(self) -> None:
        await self._client.aclose()


# Module-level singleton — imported by main.py
short_term_memory = ShortTermMemory()
