"""
Tests for the Agentic Intelligence Layer
Covers: base agent, tools, coordinator routing, MCP protocol.
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.base import (
    BaseAgent, AgentState, AgentStep, AgentMemory,
    Tool, ToolResult,
)
from src.agents.tools import (
    RAGSearchTool, SafetyCheckTool, DrugComparisonTool, SynthesizeTool,
)
from src.agents.clinical_agents import DrugAnalysisAgent, SafetyReviewAgent, ComparisonAgent
from src.agents.coordinator import MultiAgentCoordinator, TaskClassifier
from src.agents.mcp_server import MCPServer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_retriever():
    """Mock retriever that returns canned results."""
    retriever = MagicMock()

    @dataclass
    class MockDoc:
        content: str = "[Contraindications] JARDIANCE is contraindicated in severe renal impairment"
        metadata: dict = None
        score: float = 0.85
        rerank_score: float = 0.9
        drug_name: str = "JARDIANCE"
        section_type: str = "contraindications"
        citation: str = "[JARDIANCE | Contraindications | lbl_001]"

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {
                    "drug_name": self.drug_name,
                    "section_type": self.section_type,
                    "section_display_name": "Contraindications",
                    "label_id": "lbl_001",
                }

    mock_result = MagicMock()
    mock_result.documents = [MockDoc(), MockDoc(
        content="[Dosage] 10 mg once daily",
        metadata={"drug_name": "JARDIANCE", "section_type": "dosage_and_administration",
                  "section_display_name": "Dosage", "label_id": "lbl_001"},
        section_type="dosage_and_administration",
        citation="[JARDIANCE | Dosage | lbl_001]",
    )]
    mock_result.top_documents = mock_result.documents

    retriever.retrieve.return_value = mock_result
    return retriever


@pytest.fixture
def mock_model_router():
    """Mock model router that returns canned LLM responses."""
    router = MagicMock()
    router.generate = AsyncMock(return_value={
        "content": json.dumps([
            {"action": "Check safety", "tool": "safety_check",
             "params": {"drug_name": "JARDIANCE"}, "reasoning": "Safety first"},
            {"action": "Search labels", "tool": "rag_search",
             "params": {"query": "Jardiance"}, "reasoning": "Get info"},
            {"action": "Synthesize", "tool": "synthesize",
             "params": {"task": "test", "output_format": "narrative"},
             "reasoning": "Combine findings"},
        ]),
        "model": "test-model",
        "total_tokens": 500,
    })
    return router


@pytest.fixture
def mock_fda_client():
    return MagicMock()


@pytest.fixture
def mock_trials_client():
    return MagicMock()


# ============================================================
# Test: Agent Memory (Module 9)
# ============================================================

class TestAgentMemory:
    def test_empty_memory(self):
        mem = AgentMemory()
        assert mem.step_count == 0
        assert mem.findings == []

    def test_add_step(self):
        mem = AgentMemory()
        step = AgentStep(step_number=1, action="Test", reasoning="Testing")
        mem.add_step(step)
        assert mem.step_count == 1

    def test_add_finding(self):
        mem = AgentMemory()
        mem.add_finding("Jardiance has a boxed warning")
        assert len(mem.findings) == 1

    def test_recent_steps(self):
        mem = AgentMemory()
        for i in range(10):
            mem.add_step(AgentStep(step_number=i, action=f"Step {i}", reasoning=""))
        recent = mem.get_recent_steps(3)
        assert len(recent) == 3
        assert recent[0].step_number == 7

    def test_step_summary(self):
        mem = AgentMemory()
        step = AgentStep(step_number=1, action="Search drugs", reasoning="Need info")
        step.tool_result = ToolResult(tool_name="rag_search", success=True, data="Found 3 results")
        mem.add_step(step)
        summary = mem.get_step_summary()
        assert "Search drugs" in summary
        assert "✓" in summary


# ============================================================
# Test: Tool Result (Module 9)
# ============================================================

class TestToolResult:
    def test_success_result(self):
        result = ToolResult(tool_name="rag_search", success=True, data={"docs": []})
        assert result.success
        d = result.to_dict()
        assert d["tool_name"] == "rag_search"

    def test_failure_result(self):
        result = ToolResult(tool_name="fda_lookup", success=False, error="API timeout")
        assert not result.success
        assert result.error == "API timeout"


# ============================================================
# Test: Task Classifier (Module 9)
# ============================================================

class TestTaskClassifier:
    def setup_method(self):
        self.classifier = TaskClassifier()

    def test_classify_safety(self):
        assert self.classifier.classify("Is Jardiance safe during pregnancy?") == "safety_review"

    def test_classify_comparison(self):
        assert self.classifier.classify("Compare Jardiance vs Farxiga") == "comparison"

    def test_classify_analysis(self):
        assert self.classifier.classify("Tell me about Ozempic") == "drug_analysis"

    def test_classify_adverse_reactions(self):
        assert self.classifier.classify("What are the side effects of metformin?") == "safety_review"

    def test_classify_default(self):
        # Ambiguous query defaults to drug_analysis
        assert self.classifier.classify("Jardiance") == "drug_analysis"

    def test_get_agent_name(self):
        assert self.classifier.get_agent_name("safety_review") == "safety_review"
        assert self.classifier.get_agent_name("comparison") == "comparison"
        assert self.classifier.get_agent_name("drug_analysis") == "drug_analysis"


# ============================================================
# Test: RAG Search Tool (Module 9)
# ============================================================

class TestRAGSearchTool:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_retriever):
        tool = RAGSearchTool(mock_retriever)
        result = await tool.execute(query="Jardiance contraindications")
        assert result.success
        assert len(result.data["documents"]) > 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_retriever):
        tool = RAGSearchTool(mock_retriever)
        result = await tool.execute(
            query="contraindications",
            drug_name="JARDIANCE",
            section_type="contraindications",
        )
        assert result.success
        mock_retriever.retrieve.assert_called_once()

    def test_tool_schema(self, mock_retriever):
        tool = RAGSearchTool(mock_retriever)
        schema = tool.get_schema()
        assert schema["name"] == "rag_search"
        assert "query" in schema["parameters"]["properties"]


# ============================================================
# Test: Safety Check Tool
# ============================================================

class TestSafetyCheckTool:
    @pytest.mark.asyncio
    async def test_safety_check(self, mock_retriever):
        tool = SafetyCheckTool(mock_retriever)
        result = await tool.execute(drug_name="JARDIANCE")
        assert result.success
        assert result.data["drug_name"] == "JARDIANCE"


# ============================================================
# Test: Synthesize Tool
# ============================================================

class TestSynthesizeTool:
    @pytest.mark.asyncio
    async def test_synthesis(self, mock_model_router):
        mock_model_router.generate = AsyncMock(return_value={
            "content": "Jardiance is an SGLT2 inhibitor used for diabetes.",
            "model": "test",
            "total_tokens": 200,
        })
        tool = SynthesizeTool(mock_model_router)
        result = await tool.execute(
            task="Tell me about Jardiance",
            findings=["Finding 1: SGLT2 inhibitor", "Finding 2: Used for diabetes"],
        )
        assert result.success
        assert "synthesis" in result.data


# ============================================================
# Test: Drug Analysis Agent (Module 9: Single-Agent)
# ============================================================

class TestDrugAnalysisAgent:
    @pytest.mark.asyncio
    async def test_plan_creates_steps(self, mock_retriever, mock_model_router):
        tools = [RAGSearchTool(mock_retriever), SafetyCheckTool(mock_retriever),
                 SynthesizeTool(mock_model_router)]
        agent = DrugAnalysisAgent(
            name="TestAgent", role="test", tools=tools,
            model_router=mock_model_router, max_steps=5,
        )
        plan = await agent.plan("Tell me about Jardiance")
        assert len(plan) >= 2
        # Should always have safety check
        tool_names = [s.get("tool") for s in plan]
        assert "safety_check" in tool_names

    @pytest.mark.asyncio
    async def test_plan_always_ends_with_synthesis(self, mock_retriever, mock_model_router):
        tools = [RAGSearchTool(mock_retriever), SafetyCheckTool(mock_retriever),
                 SynthesizeTool(mock_model_router)]
        agent = DrugAnalysisAgent(
            name="TestAgent", role="test", tools=tools,
            model_router=mock_model_router,
        )
        plan = await agent.plan("What is metformin?")
        assert plan[-1]["tool"] == "synthesize"


# ============================================================
# Test: Coordinator Routing (Module 9: Multi-Agent)
# ============================================================

class TestCoordinator:
    def test_coordinator_init(self, mock_retriever, mock_model_router):
        coordinator = MultiAgentCoordinator(
            retriever=mock_retriever,
            model_router=mock_model_router,
        )
        assert "drug_analysis" in coordinator.agents
        assert "safety_review" in coordinator.agents
        assert "comparison" in coordinator.agents
        assert len(coordinator.tools) >= 4

    def test_coordinator_stats(self, mock_retriever, mock_model_router):
        coordinator = MultiAgentCoordinator(
            retriever=mock_retriever,
            model_router=mock_model_router,
        )
        stats = coordinator.get_stats()
        assert "total_tokens_used" in stats
        assert "agents" in stats
        assert "tools_available" in stats


# ============================================================
# Test: MCP Server (Module 12)
# ============================================================

class TestMCPServer:
    def setup_method(self):
        self.server = MCPServer()

    @pytest.mark.asyncio
    async def test_initialize(self):
        response = await self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in response["result"]["capabilities"]

    @pytest.mark.asyncio
    async def test_tools_list(self):
        response = await self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tools = response["result"]["tools"]
        assert len(tools) == 5
        tool_names = [t["name"] for t in tools]
        assert "query_drug_labels" in tool_names
        assert "compare_drugs" in tool_names
        assert "safety_check" in tool_names
        assert "list_available_drugs" in tool_names

    @pytest.mark.asyncio
    async def test_tools_have_schemas(self):
        response = await self.server.handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {},
        })
        for tool in response["result"]["tools"]:
            assert "inputSchema" in tool
            assert "description" in tool
            assert tool["description"]  # Non-empty

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        response = await self.server.handle_message({
            "jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {},
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_ping(self):
        response = await self.server.handle_message({
            "jsonrpc": "2.0", "id": 5, "method": "ping", "params": {},
        })
        assert "result" in response

    @pytest.mark.asyncio
    async def test_notification_no_response(self):
        """Notifications (no id) should not return a response."""
        response = await self.server.handle_message({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        })
        assert response is None
