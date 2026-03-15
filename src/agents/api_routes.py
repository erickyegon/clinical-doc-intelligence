"""
Agentic API Endpoints
Adds multi-agent intelligence capabilities to the platform API.

Module 9: Agents, Multi-Agent & Deep Agent Systems
Module 12: MCP integration (via /mcp endpoint)
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agentic Intelligence"])

# Module-level coordinator reference (set during app startup)
_coordinator = None


def set_coordinator(coordinator):
    """Set the coordinator instance during app initialization."""
    global _coordinator
    _coordinator = coordinator


# === Request/Response Schemas ===

class AgentQueryRequest(BaseModel):
    """Request for agentic analysis — goes through the full plan→execute→reflect loop."""
    task: str = Field(..., min_length=5, max_length=2000,
                      description="Clinical intelligence task (e.g., 'Analyze the safety profile of Jardiance for elderly patients')")
    force_agent: Optional[str] = Field(None, description="Override auto-routing: drug_analysis, safety_review, comparison")


class AgentQueryResponse(BaseModel):
    answer: str
    metadata: dict
    execution_trace: list


class AgentStatsResponse(BaseModel):
    total_tokens_used: int
    token_budget_remaining: int
    agents: dict
    tools_available: list


# === Endpoints ===

@router.post("/analyze", response_model=AgentQueryResponse)
async def agent_analyze(request: AgentQueryRequest):
    """
    Run a multi-agent clinical intelligence analysis.
    
    The coordinator automatically:
    1. Classifies your task (drug analysis, safety review, or comparison)
    2. Routes to the appropriate specialized agent
    3. The agent plans its approach, calls tools, and synthesizes results
    4. Safety-critical results trigger human-in-the-loop flags
    
    Example tasks:
    - "Tell me about Ozempic for type 2 diabetes management"
    - "Is Jardiance safe for patients with renal impairment?"
    - "Compare the safety profiles of Jardiance vs Farxiga vs Invokana"
    """
    if _coordinator is None:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        result = await _coordinator.process(request.task)
        return AgentQueryResponse(
            answer=result.answer,
            metadata=result.to_dict().get("metadata", {}),
            execution_trace=result.execution_trace,
        )
    except Exception as e:
        logger.error(f"Agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")


@router.get("/stats", response_model=AgentStatsResponse)
async def agent_stats():
    """Get multi-agent system statistics including token usage and agent states."""
    if _coordinator is None:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    stats = _coordinator.get_stats()
    return AgentStatsResponse(**stats)


@router.get("/tools")
async def list_agent_tools():
    """List all tools available to the agent system."""
    if _coordinator is None:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    tools = []
    for name, tool in _coordinator.tools.items():
        tools.append({
            "name": name,
            "description": tool.description,
            "schema": tool.get_schema(),
        })
    return {"tools": tools, "total": len(tools)}
