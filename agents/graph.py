"""LangGraph Multi-Agent Orchestration State Machine for OmniBrain.

Implements the multi-agent reasoning core as an explicit LangGraph StateGraph:
- Supervisor Node: classifies query & chooses parallel sub-agents (RETRIEVAL, VISION, SQL)
- Sub-Agent Nodes: execute specialized domain tasks with Self-RAG retry loops
- Merge Node: aggregates multimodal context & citation lineages
- Generation Node: synthesizes grounded answer with inline citations via Gemini
- Guardrails Node: validates response grounding with NeMo Guardrails
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from omnibrain.agents.retrieval_agent import AgentResult, RetrievalAgent
from omnibrain.agents.sql_agent import SQLAgent
from omnibrain.agents.supervisor import SupervisorAgent
from omnibrain.agents.vision_agent import VisionAgent
from omnibrain.config import Settings
from omnibrain.generation.engine import GenerationEngine
from omnibrain.guardrails.engine import GuardrailsEngine
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.schemas.questions import AnswerResponse, Citation

logger = logging.getLogger(__name__)


class OmniBrainState(TypedDict):
    """Complete execution state flowing through the LangGraph state machine."""

    question: str
    document_id: str
    selected_agents: list[str]
    retrieval_result: AgentResult | None
    vision_result: AgentResult | None
    sql_result: AgentResult | None
    merged_context: str
    citations: list[Citation]
    agents_used: list[str]
    final_answer: str
    guardrails_passed: bool
    guardrails_reason: str | None
    generation_retry_count: int
    error: str | None


class OmniBrainGraphOrchestrator:
    """Compiles and executes the LangGraph state machine for document Q&A."""

    def __init__(
        self,
        settings: Settings,
        supervisor_agent: SupervisorAgent,
        retrieval_agent: RetrievalAgent,
        vision_agent: VisionAgent,
        sql_agent: SQLAgent,
        generation_engine: GenerationEngine,
        guardrails_engine: GuardrailsEngine,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._supervisor = supervisor_agent
        self._retrieval_agent = retrieval_agent
        self._vision_agent = vision_agent
        self._sql_agent = sql_agent
        self._generation_engine = generation_engine
        self._guardrails = guardrails_engine
        self._tracer = tracer
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph StateGraph."""
        builder = StateGraph(OmniBrainState)

        # Register nodes
        builder.add_node("supervisor_node", self._supervisor_node)
        builder.add_node("retrieval_node", self._retrieval_node)
        builder.add_node("vision_node", self._vision_node)
        builder.add_node("sql_node", self._sql_node)
        builder.add_node("merge_node", self._merge_node)
        builder.add_node("generation_node", self._generation_node)
        builder.add_node("guardrails_node", self._guardrails_node)

        # Flow starting at supervisor
        builder.add_edge(START, "supervisor_node")

        # Conditional fan-out to parallel agents
        builder.add_conditional_edges(
            "supervisor_node",
            self._route_next_agents,
            ["retrieval_node", "vision_node", "sql_node", "merge_node"],
        )

        # Sub-agents all feed forward to merge_node
        builder.add_edge("retrieval_node", "merge_node")
        builder.add_edge("vision_node", "merge_node")
        builder.add_edge("sql_node", "merge_node")

        # Merge -> Generation -> Guardrails
        builder.add_edge("merge_node", "generation_node")
        builder.add_edge("generation_node", "guardrails_node")

        # Guardrails conditional loop or finish
        builder.add_conditional_edges(
            "guardrails_node",
            self._guardrails_decision,
            {"retry_generation": "generation_node", "finish": END},
        )

        return builder.compile()

    # ── Node Implementations ──────────────────────────────────────────

    async def _supervisor_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Classify user question and choose active sub-agent branches."""
        logger.info("LangGraph: executing supervisor_node for doc=%s", state["document_id"])
        selected = await self._supervisor.classify_route(
            state["document_id"], state["question"]
        )
        return {"selected_agents": selected}

    async def _retrieval_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Execute semantic retrieval agent with Self-RAG loop."""
        logger.info("LangGraph: executing retrieval_node")
        try:
            res = await self._retrieval_agent.run(state["document_id"], state["question"])
            return {"retrieval_result": res}
        except Exception as exc:
            logger.error("Error in retrieval_node: %s", exc, exc_info=True)
            return {"retrieval_result": AgentResult(
                context_text="", citations=[], agent_name="RETRIEVAL",
                agent_errored=True, error_message=str(exc),
            )}

    async def _vision_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Execute vision agent with CLIP selection and VLM reasoning."""
        logger.info("LangGraph: executing vision_node")
        try:
            res = await self._vision_agent.run(state["document_id"], state["question"])
            return {"vision_result": res}
        except Exception as exc:
            logger.error("Error in vision_node: %s", exc, exc_info=True)
            return {"vision_result": AgentResult(
                context_text="", citations=[], agent_name="VISION",
                agent_errored=True, error_message=str(exc),
            )}

    async def _sql_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Execute SQL agent with Text-to-SQL and Self-RAG retry."""
        logger.info("LangGraph: executing sql_node")
        try:
            res = await self._sql_agent.run(state["document_id"], state["question"])
            return {"sql_result": res}
        except Exception as exc:
            logger.error("Error in sql_node: %s", exc, exc_info=True)
            return {"sql_result": AgentResult(
                context_text="", citations=[], agent_name="SQL",
                agent_errored=True, error_message=str(exc),
            )}

    async def _merge_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Merge outputs from all executed agents into a unified, cited context block."""
        logger.info("LangGraph: executing merge_node")
        context_blocks: list[str] = []
        all_citations: list[Citation] = []
        agents_used: list[str] = []

        for res in [state.get("retrieval_result"), state.get("vision_result"), state.get("sql_result")]:
            if res and res.context_text:
                context_blocks.append(f"--- Context from {res.agent_name} Agent ---\n{res.context_text}")
                all_citations.extend(res.citations)
                if res.agent_name not in agents_used:
                    agents_used.append(res.agent_name)

        merged = "\n\n".join(context_blocks)
        return {
            "merged_context": merged,
            "citations": all_citations,
            "agents_used": agents_used,
        }

    async def _generation_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Call Gemini to generate a synthesized, citation-indexed answer."""
        logger.info("LangGraph: executing generation_node (retry=%d)", state.get("generation_retry_count", 0))
        merged = state.get("merged_context", "")

        if not merged:
            # Check if any agent actually errored (vs just finding nothing)
            any_errored = any(
                r and r.agent_errored
                for r in [state.get("retrieval_result"), state.get("vision_result"), state.get("sql_result")]
            )
            if any_errored:
                return {
                    "final_answer": "Something went wrong while researching this question — please try again.",
                    "guardrails_passed": True,
                }
            return {
                "final_answer": "Based on the available document content, I could not find relevant information to answer this question.",
                "guardrails_passed": True,
            }

        answer_resp = await self._generation_engine.generate_raw(
            question=state["question"],
            context=merged,
            retry_count=state.get("generation_retry_count", 0),
        )

        return {"final_answer": answer_resp}

    async def _guardrails_node(self, state: OmniBrainState) -> dict[str, Any]:
        """Validate response grounding and document-scope adherence via NeMo Guardrails."""
        logger.info("LangGraph: executing guardrails_node")
        answer = state.get("final_answer", "")
        context = state.get("merged_context", "")
        question = state["question"]

        if not context or not answer:
            return {"guardrails_passed": True}

        result = await self._guardrails.validate_response(
            question=question,
            answer=answer,
            context=context,
        )

        if not result.passed:
            logger.warning("Guardrails rejected response: %s", result.reason)
            return {
                "guardrails_passed": False,
                "guardrails_reason": result.reason,
                "final_answer": result.sanitized_answer,
                "generation_retry_count": state.get("generation_retry_count", 0) + 1,
            }

        return {
            "guardrails_passed": True,
            "guardrails_reason": None,
            "final_answer": result.sanitized_answer,
        }

    # ── Conditional Edges ─────────────────────────────────────────────

    def _route_next_agents(self, state: OmniBrainState) -> list[str]:
        """Determine which agent nodes to invoke from the supervisor's decision."""
        selected = state.get("selected_agents", ["RETRIEVAL"])
        routes = []
        if "RETRIEVAL" in selected:
            routes.append("retrieval_node")
        if "VISION" in selected:
            routes.append("vision_node")
        if "SQL" in selected:
            routes.append("sql_node")
        return routes if routes else ["retrieval_node"]

    def _guardrails_decision(self, state: OmniBrainState) -> str:
        """Route to END if passed or max retry reached; otherwise retry generation once."""
        if state.get("guardrails_passed", True):
            return "finish"
        if state.get("generation_retry_count", 0) < 1:
            return "retry_generation"
        return "finish"

    # ── Public Query Execution ────────────────────────────────────────

    async def ainvoke(
        self,
        document_id: str,
        question: str,
        *,
        trace: Any = None,
    ) -> AnswerResponse:
        """Run the full LangGraph state machine end-to-end for a question."""
        span = None
        if self._tracer and self._tracer.enabled:
            span = self._tracer.start_span(
                "LangGraph.OmniBrainGraph.ainvoke",
                trace=trace,
                input_data={"document_id": document_id, "question": question},
            )

        initial_state: OmniBrainState = {
            "question": question,
            "document_id": document_id,
            "selected_agents": [],
            "retrieval_result": None,
            "vision_result": None,
            "sql_result": None,
            "merged_context": "",
            "citations": [],
            "agents_used": [],
            "final_answer": "",
            "guardrails_passed": True,
            "guardrails_reason": None,
            "generation_retry_count": 0,
            "error": None,
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)

            answer_obj = AnswerResponse(
                answer=final_state["final_answer"],
                citations=final_state["citations"],
                agents_used=final_state["agents_used"],
                document_id=document_id,
                question=question,
            )

            if self._tracer and span:
                self._tracer.end_span(
                    span,
                    output={
                        "answer_len": len(final_state["final_answer"]),
                        "agents_used": final_state["agents_used"],
                        "citations_count": len(final_state["citations"]),
                    },
                )

            return answer_obj

        except Exception as exc:
            logger.error("LangGraph execution failed: %s", exc, exc_info=True)
            if self._tracer and span:
                self._tracer.end_span(
                    span, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise
