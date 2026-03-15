"""
Clinical Intelligence Tools
Concrete tool implementations that agents use to interact with
the knowledge base, FDA APIs, and LLMs.

Module 9: Tools as Agent Interfaces
- APIs, functions, search, RAG, code execution as tools
"""
import json
import logging
import time
from typing import Optional

from src.agents.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class RAGSearchTool(Tool):
    """Search the FDA drug label vector store."""

    name = "rag_search"
    description = (
        "Search the FDA drug label knowledge base for information about drugs, "
        "contraindications, dosing, warnings, adverse reactions, and drug interactions. "
        "Returns relevant document chunks with citations."
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query about drug labels"},
                "drug_name": {"type": "string", "description": "Optional: filter to specific drug"},
                "section_type": {"type": "string", "description": "Optional: filter by section (e.g., contraindications, adverse_reactions)"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, drug_name: str = None, section_type: str = None, **kwargs) -> ToolResult:
        try:
            result = self.retriever.retrieve(
                query=query,
                drug_name=drug_name,
                section_type=section_type,
            )
            documents = []
            for doc in result.top_documents:
                documents.append({
                    "content": doc.content[:800],
                    "drug_name": doc.drug_name,
                    "section": doc.metadata.get("section_display_name", ""),
                    "citation": doc.citation,
                    "score": round(doc.rerank_score or doc.score, 3),
                })

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"documents": documents, "total_found": len(result.documents)},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class FDALabelLookupTool(Tool):
    """Fetch fresh drug label data from the openFDA API."""

    name = "fda_label_lookup"
    description = (
        "Look up the latest FDA drug label directly from the openFDA API. "
        "Use when the vector store doesn't have information about a specific drug, "
        "or when you need the most current label data."
    )

    def __init__(self, fda_client):
        self.fda_client = fda_client

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "drug_name": {"type": "string", "description": "Brand or generic drug name"},
                "section": {"type": "string", "description": "Specific label section to retrieve"},
            },
            "required": ["drug_name"],
        }

    async def execute(self, drug_name: str, section: str = None, **kwargs) -> ToolResult:
        try:
            raw_labels = self.fda_client.search_labels(drug_name=drug_name, limit=3)
            if not raw_labels:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={"message": f"No FDA labels found for '{drug_name}'", "labels": []},
                )

            labels = []
            for raw in raw_labels[:2]:
                label = self.fda_client.parse_label(raw)
                label_data = {
                    "drug_name": label.drug_name,
                    "generic_name": label.generic_name,
                    "manufacturer": label.manufacturer,
                }
                if section and section in label.sections:
                    label_data["requested_section"] = {
                        "name": label.sections[section]["display_name"],
                        "content": label.sections[section]["content"][:1500],
                    }
                else:
                    label_data["available_sections"] = list(label.sections.keys())
                    # Return most important sections
                    for key in ["boxed_warning", "contraindications", "indications_and_usage"]:
                        if key in label.sections:
                            label_data[key] = label.sections[key]["content"][:500]
                labels.append(label_data)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"labels": labels, "count": len(labels)},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class ClinicalTrialSearchTool(Tool):
    """Search ClinicalTrials.gov for trial information."""

    name = "clinical_trial_search"
    description = (
        "Search ClinicalTrials.gov for clinical trials related to a drug or condition. "
        "Returns trial metadata: phase, status, endpoints, enrollment, sponsor."
    )

    def __init__(self, trials_client):
        self.trials_client = trials_client

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "drug_name": {"type": "string", "description": "Drug or intervention name"},
                "condition": {"type": "string", "description": "Medical condition"},
                "phase": {"type": "string", "description": "Trial phase (e.g., PHASE3)"},
            },
            "required": ["drug_name"],
        }

    async def execute(self, drug_name: str = None, condition: str = None, phase: str = None, **kwargs) -> ToolResult:
        try:
            result = self.trials_client.search_trials(
                intervention=drug_name,
                condition=condition,
                phase=phase,
                page_size=10,
            )
            trials = []
            for study in result.get("studies", [])[:5]:
                try:
                    trial = self.trials_client.parse_trial(study)
                    trials.append({
                        "nct_id": trial.nct_id,
                        "title": trial.title[:200],
                        "status": trial.status,
                        "phase": trial.phase,
                        "enrollment": trial.enrollment,
                        "sponsor": trial.sponsor,
                        "conditions": trial.conditions[:3],
                        "primary_outcomes": [o["measure"][:100] for o in trial.primary_outcomes[:3]],
                    })
                except Exception:
                    continue

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"trials": trials, "total": len(trials)},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class DrugComparisonTool(Tool):
    """Compare information across multiple drugs."""

    name = "drug_comparison"
    description = (
        "Compare specific aspects (safety, dosing, indications) across multiple drugs. "
        "Retrieves and structures data from the knowledge base for side-by-side analysis."
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "drug_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of drug names to compare",
                },
                "aspect": {"type": "string", "description": "What to compare: safety, dosing, indications, adverse_reactions"},
            },
            "required": ["drug_names", "aspect"],
        }

    async def execute(self, drug_names: list, aspect: str = "safety", **kwargs) -> ToolResult:
        try:
            section_map = {
                "safety": ["contraindications", "boxed_warning", "warnings_and_cautions"],
                "dosing": ["dosage_and_administration"],
                "indications": ["indications_and_usage"],
                "adverse_reactions": ["adverse_reactions"],
                "interactions": ["drug_interactions"],
            }
            sections = section_map.get(aspect, ["contraindications", "warnings_and_cautions"])

            comparison = {}
            for drug in drug_names[:5]:  # Cap at 5 drugs
                drug_data = {}
                for section in sections:
                    result = self.retriever.retrieve(
                        query=f"{drug} {section.replace('_', ' ')}",
                        drug_name=drug,
                        section_type=section,
                    )
                    if result.documents:
                        drug_data[section] = result.top_documents[0].content[:600]
                comparison[drug] = drug_data

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"comparison": comparison, "aspect": aspect, "drugs": drug_names},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class SafetyCheckTool(Tool):
    """Check for safety signals and critical warnings for a drug."""

    name = "safety_check"
    description = (
        "Retrieve all safety-critical information for a drug: black box warnings, "
        "contraindications, REMS requirements, and pregnancy category. "
        "Always use this before providing dosing or treatment recommendations."
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "drug_name": {"type": "string", "description": "Drug name to check safety profile"},
            },
            "required": ["drug_name"],
        }

    async def execute(self, drug_name: str, **kwargs) -> ToolResult:
        try:
            safety_sections = [
                "boxed_warning", "contraindications",
                "warnings_and_cautions", "warnings",
            ]
            safety_data = {"drug_name": drug_name, "sections": {}}

            for section in safety_sections:
                result = self.retriever.retrieve(
                    query=f"{drug_name} {section.replace('_', ' ')}",
                    drug_name=drug_name,
                    section_type=section,
                )
                if result.documents:
                    safety_data["sections"][section] = {
                        "content": result.top_documents[0].content[:1000],
                        "citation": result.top_documents[0].citation,
                    }

            safety_data["has_boxed_warning"] = "boxed_warning" in safety_data["sections"]
            safety_data["sections_found"] = list(safety_data["sections"].keys())

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=safety_data,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class SynthesizeTool(Tool):
    """Use LLM to synthesize findings into a structured response."""

    name = "synthesize"
    description = (
        "Synthesize collected findings into a coherent, citation-grounded response. "
        "Use after gathering data from search and comparison tools."
    )

    def __init__(self, model_router):
        self.model_router = model_router

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Original user question/task"},
                "findings": {"type": "array", "items": {"type": "string"}, "description": "Accumulated findings to synthesize"},
                "output_format": {"type": "string", "description": "Format: narrative, comparison_table, safety_brief"},
            },
            "required": ["task", "findings"],
        }

    async def execute(self, task: str, findings: list, output_format: str = "narrative", **kwargs) -> ToolResult:
        try:
            findings_text = "\n\n".join(f"Finding {i+1}: {f}" for i, f in enumerate(findings))

            system_prompt = (
                "You are a pharmaceutical regulatory intelligence analyst. "
                "Synthesize the provided findings into a clear, citation-grounded response. "
                "Use ⚠️ for safety-critical information. Cite sources with [Source: ...] notation."
            )

            format_instructions = {
                "narrative": "Write a clear narrative answer with inline citations.",
                "comparison_table": "Structure the response as a comparison with clear drug-by-drug sections.",
                "safety_brief": "Lead with safety-critical information. Use ⚠️ markers. Be comprehensive on warnings.",
            }

            user_prompt = f"""Task: {task}

Collected Findings:
{findings_text}

Output Format: {format_instructions.get(output_format, format_instructions['narrative'])}

Synthesize these findings into a complete, professional response."""

            response = await self.model_router.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"synthesis": response.get("content", ""), "model": response.get("model", "")},
                token_cost=response.get("total_tokens", 0),
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
