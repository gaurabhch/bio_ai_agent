# main.py
# FastAPI entry point for Pipeline 2 — Online Inference.
#
# Startup:
#   1. Creates AsyncGroq client (one per app lifetime)
#   2. Creates SQLAlchemy async engine + session factory
#   3. Compiles the LangGraph (loads SentenceTransformer + builds all nodes)
#
# Endpoints:
#   POST /chat         → graph.ainvoke() → JSON response
#   GET  /chat/stream  → SSE streaming of final_response word by word
#   GET  /health       → liveness probe

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

import asyncio
import json
import ssl
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from agents.graph import build_graph
from agents.state import AgentState
from memory.long_term import LongTermMemory
from memory.short_term import short_term_memory
from config import (
    DATABASE_URL_ASYNC,
    GROQ_API_KEY,
    STREAM_WORD_DELAY,
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id        : str
    conversation_id: str
    message        : str
    use_case       : str = "pcos"         # pcos | pregnancy | period_issues


class ChatResponse(BaseModel):
    answer         : str
    sources        : list[str]
    agent_node_used: str
    confidence_score: float
    is_crisis      : bool


# ── App lifespan — initialise shared resources once ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── SSL context for asyncpg ───────────────────────────────────────────
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE

    # ── Startup ───────────────────────────────────────────────────────────
    app.state.groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    engine = create_async_engine(
        DATABASE_URL_ASYNC,
        pool_size    = 10,
        max_overflow = 20,
        connect_args = {"ssl": ssl_ctx},   
    )

    app.state.db_session_factory = async_sessionmaker(
        engine, expire_on_commit=False
    )

    # Compile graph — loads SentenceTransformer once
    app.state.graph = build_graph(
        groq_client        = app.state.groq_client,
        db_session_factory = app.state.db_session_factory,
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    await app.state.groq_client.close()
    await short_term_memory.close()
    await engine.dispose()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="BioCanvas PCOS AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",          # url_for('static', path=...) uses this name
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _build_initial_state(
    request: ChatRequest,
    db_session_factory,
) -> AgentState:
    """
    Assemble the full AgentState before invoking the graph.
    Reads conversation history from Redis and user profile from PostgreSQL.
    """
    # 1 — Conversation history from Redis
    try:
        history = await short_term_memory.get_history(request.conversation_id)
    except Exception:
        history = []

    # 2 — User profile + memory from PostgreSQL
    try:
        async with db_session_factory() as session:
            ltm     = LongTermMemory(session)
            profile = await ltm.get_user_profile(request.user_id)
            memory  = await ltm.get_user_memory(request.user_id)
    except Exception:
        profile = {"use_case": request.use_case, "user_tags": [], "health_summary": {}}
        memory  = {}

    # 3 — Append the new user message to history
    history.append({"role": "user", "text": request.message})

    return AgentState(
        messages            = history,
        user_id             = request.user_id,
        use_case            = profile.get("use_case", request.use_case),
        user_tags           = profile.get("user_tags", []),
        user_health_summary = profile.get("health_summary", {}),
        user_memory         = memory,
        conversation_id     = request.conversation_id,
        # Supervisor fills these in:
        rewritten_query     = "",
        next_agent          = "",
        routing_reason      = "",
        response_mode       = "",
        is_crisis           = False,
        is_flagged          = False,
        flag_reason         = None,
        # Specialist fills these in:
        retrieved_context   = [],
        raw_response        = "",
        agent_node_used     = "",
        confidence_score    = 0.0,
        # Verifier fills these in:
        verified_response   = "",
        citations           = [],
        # Merge agent fills these in:
        final_response      = "",
        sources             = [],
    )


async def _persist_after_response(state: AgentState, db_session_factory) -> None:
    """
    Fire-and-forget: save conversation to Redis + PostgreSQL after streaming.
    Called without awaiting so it never delays the response.
    """
    try:
        # Save AI turn to Redis
        await short_term_memory.append_message(
            state["conversation_id"], "assistant", state["final_response"]
        )
        # Save user message to Redis
        await short_term_memory.append_message(
            state["conversation_id"], "user", state["messages"][-1]["text"]
        )
        # Persist full exchange to PostgreSQL
        async with db_session_factory() as session:
            ltm = LongTermMemory(session)
            await ltm.save_chat_message(state, session)
    except Exception:
        pass    # persistence failures must never crash the response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Runs the full LangGraph pipeline and returns the final response.
    """
    initial_state = await _build_initial_state(
        request, app.state.db_session_factory
    )

    try:
        final_state: AgentState = await app.state.graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist asynchronously — do not await
    asyncio.create_task(
        _persist_after_response(final_state, app.state.db_session_factory)
    )

    return ChatResponse(
        answer          = final_state["final_response"],
        sources         = final_state.get("sources", []),
        agent_node_used = final_state.get("agent_node_used", ""),
        confidence_score= final_state.get("confidence_score", 0.0),
        is_crisis       = final_state.get("is_crisis", False),
    )


# @app.get("/chat/stream")
# async def chat_stream(
#     user_id        : str,
#     conversation_id: str,
#     message        : str,
#     use_case       : str = "pcos",
# ) -> StreamingResponse:
#     """
#     SSE streaming endpoint.
#     Streams final_response word-by-word with Server-Sent Events.
#     Sources and metadata sent as a final [DONE] event.
#     """
#     request = ChatRequest(
#         user_id        =user_id,
#         conversation_id=conversation_id,
#         message        =message,
#         use_case       =use_case,
#     )

#     async def _event_generator() -> AsyncGenerator[str, None]:
#         initial_state = await _build_initial_state(
#             request, app.state.db_session_factory
#         )
#         try:
#             final_state: AgentState = await app.state.graph.ainvoke(initial_state)
#         except Exception as exc:
#             yield f"data: {json.dumps({'error': str(exc)})}\n\n"
#             return

#         # Stream word by word
#         words = final_state["final_response"].split()
#         for i, word in enumerate(words):
#             chunk = word + (" " if i < len(words) - 1 else "")
#             yield f"data: {json.dumps({'token': chunk})}\n\n"
#             await asyncio.sleep(STREAM_WORD_DELAY)

#         # Final metadata event
#         yield f"data: {json.dumps({'done': True, 'sources': final_state.get('sources', []), 'agent_node_used': final_state.get('agent_node_used', ''), 'confidence_score': final_state.get('confidence_score', 0.0), 'is_crisis': final_state.get('is_crisis', False)})}\n\n"

#         asyncio.create_task(
#             _persist_after_response(final_state, app.state.db_session_factory)
#         )

#     return StreamingResponse(
#         _event_generator(),
#         media_type="text/event-stream",
#         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
#     )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BioCanvas PCOS AI Agent"}
