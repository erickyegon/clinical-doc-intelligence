"""
Multi-Agent Coordinator
Supervisor pattern that routes tasks to specialized agents,
manages shared state, and aggregates results.

Module 9: Multi-Agent System Designs
- Supervisor, hierarchical, and network-based architectures
- Inter-Agent Collaboration & Coordination
- Human-in-the-Loop Mechanisms
- Cost & Execution Management
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from src.agents.base import AgentState, AgentMemory
from src.agents.clinical_agents import DrugAnalysisAgent, SafetyReviewAgent, ComparisonAgent
from src.agents.tools import (
    RAGSearchTool, FDALabelLookupTool, ClinicalTrialSearchTool,
    DrugComparisonTool, SafetyCheckTool, SynthesizeTool,
)

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    """Result from the multi-agent coordinator."""
    answer: str
    agent_used: str
    task_type: str
    total_steps: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    execution_trace: list[dict] = field(default_factory=list)
    human_review_required: bool = False
    human_review_reason: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "metadata": {
                "agent_used": self.agent_used,
                "task_type": self.task_type,
                "total_steps": self.total_steps,
                "total_tokens": self.total_tokens,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "human_review_required": self.human_review_required,
                "human_review_reason": self.human_review_reason,
                "confidence": round(self.confidence, 3),
            },
            "execution_trace": self.execution_trace,
        }


class TaskClassifier:
    """
    Classify incoming tasks to route to the appropriate agent.
    Module 9: LLMs as the Reasoning & Decision Core
    """

    TASK_TYPES = {
        "drug_analysis": {
            "keywords": ["tell me about", "what is", "information on", "overview",
                         "how does", "mechanism", "pharmacology"],
            "agent": "drug_analysis",
        },
        "safety_review": {
            "keywords": ["safe", "safety", "warning", "contraindic", "pregnancy",
                         "interact", "risk", "adverse", "side effect", "black box"],
            "agent": "safety_review",
        },
        "comparison": {
            "keywords": ["compare", "versus", "vs", "difference between",
                         "which is better", "prefer", "switch from"],
            "agent": "comparison",
        },
    }

    def classify(self, task: str) -> str:
        """Classify a task into a task type."""
        task_lower = task.lower()

        # Score each task type by keyword matches
        scores = {}
        for task_type, config in self.TASK_TYPES.items():
            score = sum(1 for kw in config["keywords"] if kw in task_lower)
            scores[task_type] = score

        # Return highest scoring type, default to drug_analysis
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "drug_analysis"
        return best

    def get_agent_name(self, task_type: str) -> str:
        config = self.TASK_TYPES.get(task_type, {})
        return config.get("agent", "drug_analysis")


class MultiAgentCoordinator:
    """
    Supervisor-pattern coordinator for clinical intelligence agents.
    
    Architecture:
    ┌──────────────────────────────────┐
    │         Coordinator              │
    │  (task classification, routing,  │
    │   shared state, aggregation)     │
    ├──────────┬───────────┬───────────┤
    │  Drug    │  Safety   │ Comparison│
    │ Analysis │  Review   │  Agent    │
    │  Agent   │  Agent    │           │
    └──────────┴───────────┴───────────┘
    
    Features:
    - Automatic task classification and agent routing
    - Shared tool pool across agents
    - Token budget management across all agents
    - Human-in-the-loop approval gates
    - Execution trace for observability
    """

    def __init__(
        self,
        retriever,
        model_router,
        fda_client=None,
        trials_client=None,
        max_total_tokens: int = 100000,
        human_approval_callback: Optional[Callable] = None,
    ):
        self.model_router = model_router
        self.classifier = TaskClassifier()
        self.max_total_tokens = max_total_tokens
        self.human_approval_callback = human_approval_callback
        self._total_tokens = 0

        # Build shared tool pool
        self.tools = self._build_tools(retriever, model_router, fda_client, trials_client)

        # Initialize specialized agents
        self.agents = {
            "drug_analysis": DrugAnalysisAgent(
                name="DrugAnalysis",
                role="Comprehensive drug information analyst",
                tools=list(self.tools.values()),
                model_router=model_router,
                max_steps=8,
                max_tokens_budget=40000,
            ),
            "safety_review": SafetyReviewAgent(
                name="SafetyReview",
                role="Drug safety and risk assessment specialist",
                tools=list(self.tools.values()),
                model_router=model_router,
                max_steps=6,
                max_tokens_budget=30000,
            ),
            "comparison": ComparisonAgent(
                name="Comparison",
                role="Cross-drug comparison analyst",
                tools=list(self.tools.values()),
                model_router=model_router,
                max_steps=10,
                max_tokens_budget=50000,
            ),
        }

        logger.info(
            f"Coordinator initialized with {len(self.agents)} agents "
            f"and {len(self.tools)} tools"
        )

    def _build_tools(self, retriever, model_router, fda_client, trials_client) -> dict:
        """Build the shared tool pool."""
        tools = {
            "rag_search": RAGSearchTool(retriever),
            "drug_comparison": DrugComparisonTool(retriever),
            "safety_check": SafetyCheckTool(retriever),
            "synthesize": SynthesizeTool(model_router),
        }
        if fda_client:
            tools["fda_label_lookup"] = FDALabelLookupTool(fda_client)
        if trials_client:
            tools["clinical_trial_search"] = ClinicalTrialSearchTool(trials_client)
        return tools

    async def process(self, task: str) -> CoordinatorResult:
        """
        Main entry point: classify → route → execute → aggregate.
        
        Includes human-in-the-loop gates for safety-critical results.
        """
        start_time = time.time()

        # Step 1: Classify task
        task_type = self.classifier.classify(task)
        agent_name = self.classifier.get_agent_name(task_type)

        logger.info(f"Coordinator: task_type={task_type}, routing to {agent_name}")

        # Step 2: Check token budget
        if self._total_tokens >= self.max_total_tokens:
            return CoordinatorResult(
                answer="Token budget exceeded. Please start a new session.",
                agent_used=agent_name,
                task_type=task_type,
                confidence=0.0,
            )

        # Step 3: Route to appropriate agent
        agent = self.agents.get(agent_name)
        if not agent:
            return CoordinatorResult(
                answer=f"No agent available for task type: {task_type}",
                agent_used="none",
                task_type=task_type,
                confidence=0.0,
            )

        # Step 4: Execute agent
        try:
            agent_result = await agent.run(task)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return CoordinatorResult(
                answer=f"Analysis failed: {str(e)}. Please try rephrasing your query.",
                agent_used=agent_name,
                task_type=task_type,
                confidence=0.0,
            )

        self._total_tokens += agent_result.get("total_tokens", 0)

        # Step 5: Extract answer from agent findings
        answer = self._extract_answer(agent_result)

        # Step 6: Human-in-the-loop check
        human_review = False
        review_reason = None

        if task_type == "safety_review" and self._has_safety_concerns(agent_result):
            human_review = True
            review_reason = "Safety-critical query with boxed warning information"

        if self.human_approval_callback and human_review:
            approved = await self._request_human_approval(task, answer, review_reason)
            if not approved:
                answer += "\n\n⚠️ This response is pending human expert review."

        # Step 7: Build result
        elapsed = (time.time() - start_time) * 1000

        return CoordinatorResult(
            answer=answer,
            agent_used=agent_name,
            task_type=task_type,
            total_steps=agent_result.get("steps_taken", 0),
            total_tokens=agent_result.get("total_tokens", 0),
            total_latency_ms=elapsed,
            execution_trace=agent_result.get("execution_trace", {}).get("steps", []),
            human_review_required=human_review,
            human_review_reason=review_reason,
            confidence=self._estimate_confidence(agent_result),
        )

    def _extract_answer(self, agent_result: dict) -> str:
        """Extract the final answer from agent execution results."""
        findings = agent_result.get("findings", [])
        if not findings:
            return "I was unable to find relevant information for your query."

        # Look for a synthesis finding (last step typically)
        for finding in reversed(findings):
            if len(finding) > 200:  # Synthesis tends to be longer
                return finding

        # Fallback: combine all findings
        return "\n\n".join(findings)

    def _has_safety_concerns(self, agent_result: dict) -> bool:
        """Check if the result contains safety-critical information."""
        findings = agent_result.get("findings", [])
        safety_indicators = ["⚠️", "BOXED WARNING", "contraindicated", "fatal", "death"]
        for finding in findings:
            if any(ind in finding for ind in safety_indicators):
                return True
        return False

    def _estimate_confidence(self, agent_result: dict) -> float:
        """Estimate confidence based on agent execution quality."""
        state = agent_result.get("state", "")
        steps = agent_result.get("steps_taken", 0)
        findings = len(agent_result.get("findings", []))

        if state == "completed" and findings >= 2:
            return min(0.9, 0.5 + findings * 0.1)
        elif state == "completed":
            return 0.5
        elif state == "terminated":
            return 0.3
        return 0.1

    async def _request_human_approval(self, task: str, answer: str, reason: str) -> bool:
        """
        Human-in-the-loop approval gate.
        Module 9: Human-in-the-Loop Mechanisms
        """
        if self.human_approval_callback:
            try:
                return await self.human_approval_callback(task, answer, reason)
            except Exception:
                return True  # Default to approved if callback fails
        return True

    def get_stats(self) -> dict:
        """Get coordinator-level statistics."""
        return {
            "total_tokens_used": self._total_tokens,
            "token_budget_remaining": self.max_total_tokens - self._total_tokens,
            "agents": {
                name: {
                    "state": agent.state.value,
                    "steps_executed": agent.memory.step_count,
                }
                for name, agent in self.agents.items()
            },
            "tools_available": list(self.tools.keys()),
        }
