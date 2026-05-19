# agents/graph.py
# LangGraph graph for Pipeline 2 — Online Inference.
# Connects all nodes and defines conditional routing from the supervisor.
#
# Flow:
#   input_guardrail
#       ├── BLOCKED → output_guardrail → END
#       └── PASSED  → supervisor
#                       ├── crisis_end        → merge_agent ──┐
#                       ├── clarification_agent ───────────────┤
#                       └── {domain}_agent                     │
#                               └── pubmed_verifier ───────────┤
#                                       └── merge_agent ───────┘
#                                               └── output_guardrail → END

from groq import AsyncGroq
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.specialist_agent import SpecialistAgent
from agents.pubmed_verifier import pubmed_verifier_node
from agents.merge_agent import merge_agent_node
from agents.guardrails.input_guardrail import input_guardrail_node
from agents.guardrails.output_guardrail import output_guardrail_node
from config import EMBEDDING_MODEL, DOMAIN_KEYWORDS, GROQ_TIMEOUT, GROQ_MODEL

# ── Clarification node ────────────────────────────────────────────────────────

async def clarification_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    """
    Lightweight node for vague messages.
    Asks the user one focused follow-up question. No RAG retrieval.
    """
    message = state["messages"][-1]["text"]
    prompt = (
        "The user sent a message that is too vague to answer specifically: "
        f'"{message}"\n\n'
        "Ask exactly one short, friendly follow-up question to help clarify "
        "what they need. Do not include any medical information yet."
    )
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
            timeout=GROQ_TIMEOUT,
        )
        clarification_q = response.choices[0].message.content.strip()
    except Exception:
        clarification_q = "Could you tell me a bit more about what you're experiencing?"

    return {
        **state,
        "raw_response":      clarification_q,
        "verified_response": clarification_q,
        "retrieved_context": [],
        "confidence_score":  0.0,
        "agent_node_used":   "clarification_agent",
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_input_guardrail(state: AgentState) -> str:
    """
    If input guardrail blocked the query → skip entire pipeline,
    go straight to output_guardrail which appends disclaimer and returns.
    If guardrail passed → proceed to supervisor as normal.
    """
    if not state.get("guardrail_passed", True):
        return "output_guardrail"
    return "supervisor"


def _route_after_supervisor(state: AgentState) -> str:
    """
    Reads AgentState.next_agent and returns the node name to visit next.
    LangGraph uses this return value to follow the correct conditional edge.
    """
    next_agent = state.get("next_agent", "pcos_general_agent")

    if next_agent == "crisis_end":
        return "merge_agent"   # crisis response already set in final_response

    if next_agent == "clarification_agent":
        return "clarification_agent"

    return next_agent   # e.g. "pcos_menstrual_agent"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    groq_client: AsyncGroq,
    db_session_factory,   # async_sessionmaker — returns AsyncSession
) -> StateGraph:
    """
    Assembles and compiles the full Pipeline 2 LangGraph.

    Args:
        groq_client        : Shared AsyncGroq client (one per app lifetime).
        db_session_factory : SQLAlchemy async_sessionmaker instance.

    Returns:
        A compiled LangGraph ready for await graph.ainvoke(state)
    """
    # ── Shared embedder — loaded ONCE, same model as Pipeline 1 ──────────
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # ── One SpecialistAgent per domain — all share the same embedder ──────
    specialist_agents: dict[str, SpecialistAgent] = {
        domain: SpecialistAgent(domain=domain, embedder=embedder)
        for domain in DOMAIN_KEYWORDS
    }

    # ── Node wrappers (bind shared groq_client via closure) ───────────────

    async def _input_guardrail(state: AgentState) -> AgentState:
        return await input_guardrail_node(state, groq_client)

    async def _supervisor(state: AgentState) -> AgentState:
        return await supervisor_node(state, groq_client)

    async def _clarification(state: AgentState) -> AgentState:
        return await clarification_node(state, groq_client)

    async def _pubmed(state: AgentState) -> AgentState:
        return await pubmed_verifier_node(state, groq_client)

    async def _merge(state: AgentState) -> AgentState:
        return await merge_agent_node(state, groq_client)

    async def _output_guardrail(state: AgentState) -> AgentState:
        return await output_guardrail_node(state, groq_client)

    def _make_specialist_node(agent: SpecialistAgent):
        async def _node(state: AgentState) -> AgentState:
            async with db_session_factory() as session:
                return await agent.run(state, groq_client, session)
        return _node

    # ── Build the graph ───────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Guardrail nodes
    graph.add_node("input_guardrail",  _input_guardrail)
    graph.add_node("output_guardrail", _output_guardrail)

    # Pipeline nodes (unchanged)
    graph.add_node("supervisor",          _supervisor)
    graph.add_node("clarification_agent", _clarification)
    graph.add_node("pubmed_verifier",     _pubmed)
    graph.add_node("merge_agent",         _merge)

    # Specialist nodes per domain (unchanged)
    for domain, agent in specialist_agents.items():
        graph.add_node(f"{domain}_agent", _make_specialist_node(agent))

    # ── Entry point: input_guardrail (was: supervisor) ────────────────────
    graph.set_entry_point("input_guardrail")

    # ── Conditional edge: input_guardrail → supervisor | output_guardrail ─
    graph.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {
            "supervisor":       "supervisor",
            "output_guardrail": "output_guardrail",
        },
    )

    # ── Conditional edges from supervisor (unchanged logic) ───────────────
    possible_next_nodes = (
        ["clarification_agent", "merge_agent"]
        + [f"{d}_agent" for d in DOMAIN_KEYWORDS]
    )
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {node: node for node in possible_next_nodes},
    )

    # ── Fixed edges: specialist → verifier → merge (unchanged) ───────────
    for domain in DOMAIN_KEYWORDS:
        graph.add_edge(f"{domain}_agent", "pubmed_verifier")

    graph.add_edge("clarification_agent", "merge_agent")
    graph.add_edge("pubmed_verifier",     "merge_agent")

    # ── merge_agent → output_guardrail → END (was: merge_agent → END) ────
    graph.add_edge("merge_agent",      "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
