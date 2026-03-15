"""
Agent Foundations
Base classes for single-agent and multi-agent architectures.
Defines the planning → reasoning → acting loop pattern.

Module 9: Agentic AI Fundamentals
- What agents are vs simple LLM pipelines
- Planning, reasoning, acting loops
- Tools as agent interfaces
"""
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Lifecycle states for an agent."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_HUMAN = "waiting_for_human"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"  # Safety: max steps reached


@dataclass
class ToolResult:
    """Standard result from a tool invocation."""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    token_cost: int = 0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": str(self.data)[:500] if self.data else None,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class AgentStep:
    """A single step in the agent's execution trace."""
    step_number: int
    action: str  # What the agent decided to do
    reasoning: str  # Why it chose this action (CoT)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[ToolResult] = None
    state: AgentState = AgentState.EXECUTING
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "action": self.action,
            "reasoning": self.reasoning[:300],
            "tool": self.tool_name,
            "state": self.state.value,
            "tokens": self.tokens_used,
        }


@dataclass
class AgentMemory:
    """
    Short-term and working memory for an agent.
    Module 9: Memory & State Management
    """
    steps: list[AgentStep] = field(default_factory=list)
    context: dict = field(default_factory=dict)  # Shared state
    findings: list[str] = field(default_factory=list)  # Accumulated knowledge
    errors: list[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: AgentStep):
        self.steps.append(step)

    def add_finding(self, finding: str):
        self.findings.append(finding)

    def get_recent_steps(self, n: int = 5) -> list[AgentStep]:
        return self.steps[-n:]

    def get_step_summary(self) -> str:
        """Summarize execution history for the LLM context window."""
        lines = []
        for step in self.steps:
            status = "✓" if step.tool_result and step.tool_result.success else "✗"
            lines.append(f"Step {step.step_number} [{status}]: {step.action}")
            if step.tool_result and step.tool_result.data:
                preview = str(step.tool_result.data)[:200]
                lines.append(f"  Result: {preview}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_steps": self.step_count,
            "findings": self.findings,
            "errors": self.errors,
            "steps": [s.to_dict() for s in self.steps],
        }


class Tool(ABC):
    """
    Base class for agent tools.
    Module 9: Tools as Agent Interfaces
    """
    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> dict:
        """Return a JSON schema describing this tool for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters(),
        }

    def _get_parameters(self) -> dict:
        """Override in subclass to define parameters."""
        return {"type": "object", "properties": {}}


class BaseAgent(ABC):
    """
    Base agent with planning → reasoning → acting loop.
    
    Module 9: Single-Agent Architectures
    
    Lifecycle:
    1. Receive task
    2. Plan approach (which tools to use, in what order)
    3. Execute steps (call tools, process results)
    4. Reflect on results (is the answer sufficient?)
    5. Return final result or iterate
    
    Safety controls (Module 9):
    - Max steps limit
    - Token budget
    - Termination conditions
    - Error handling with graceful degradation
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: list[Tool],
        model_router=None,
        max_steps: int = 10,
        max_tokens_budget: int = 50000,
    ):
        self.name = name
        self.role = role
        self.tools = {t.name: t for t in tools}
        self.model_router = model_router
        self.max_steps = max_steps
        self.max_tokens_budget = max_tokens_budget
        self.memory = AgentMemory()
        self.state = AgentState.IDLE
        self.agent_id = str(uuid.uuid4())[:8]
        self._total_tokens = 0

    @abstractmethod
    async def plan(self, task: str) -> list[dict]:
        """Create an execution plan for the task."""
        pass

    @abstractmethod
    async def execute_step(self, step_plan: dict) -> AgentStep:
        """Execute a single planned step."""
        pass

    @abstractmethod
    async def reflect(self) -> bool:
        """Reflect on progress. Return True if task is complete."""
        pass

    async def run(self, task: str) -> dict:
        """
        Main agent loop: plan → execute → reflect.
        Includes safety controls for loop prevention.
        """
        self.state = AgentState.PLANNING
        self.memory = AgentMemory()
        self.memory.context["task"] = task
        start_time = time.time()

        logger.info(f"Agent [{self.name}] starting task: {task[:100]}...")

        try:
            # Step 1: Plan
            plan = await self.plan(task)
            self.memory.context["plan"] = plan

            # Step 2: Execute steps
            self.state = AgentState.EXECUTING
            for i, step_plan in enumerate(plan):
                # Safety: max steps check
                if self.memory.step_count >= self.max_steps:
                    logger.warning(f"Agent [{self.name}] hit max steps ({self.max_steps})")
                    self.state = AgentState.TERMINATED
                    break

                # Safety: token budget check
                if self._total_tokens >= self.max_tokens_budget:
                    logger.warning(f"Agent [{self.name}] exceeded token budget")
                    self.state = AgentState.TERMINATED
                    break

                step = await self.execute_step(step_plan)
                self.memory.add_step(step)
                self._total_tokens += step.tokens_used

                # Step 3: Reflect after each step
                self.state = AgentState.REFLECTING
                is_complete = await self.reflect()
                if is_complete:
                    self.state = AgentState.COMPLETED
                    break

                self.state = AgentState.EXECUTING

            if self.state == AgentState.EXECUTING:
                self.state = AgentState.COMPLETED

        except Exception as e:
            logger.error(f"Agent [{self.name}] failed: {e}")
            self.state = AgentState.FAILED
            self.memory.errors.append(str(e))

        elapsed = (time.time() - start_time) * 1000

        result = {
            "agent": self.name,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "task": task,
            "findings": self.memory.findings,
            "steps_taken": self.memory.step_count,
            "total_tokens": self._total_tokens,
            "latency_ms": round(elapsed, 1),
            "execution_trace": self.memory.to_dict(),
        }

        logger.info(
            f"Agent [{self.name}] finished: state={self.state.value}, "
            f"steps={self.memory.step_count}, tokens={self._total_tokens}"
        )
        return result

    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Invoke a registered tool by name."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}",
            )

        start = time.time()
        try:
            result = await tool.execute(**kwargs)
            result.latency_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    def get_tool_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM function calling."""
        return [t.get_schema() for t in self.tools.values()]
