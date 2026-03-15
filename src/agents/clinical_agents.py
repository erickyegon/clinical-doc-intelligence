"""
Clinical Intelligence Agents
Specialized agents for drug analysis, safety review, and comparison tasks.

Module 9: Single-Agent Architectures & Multi-Agent System Designs
- Planning, reasoning, acting loops
- Role-based prompts (Module 9: Prompting Strategies for Agents)
"""
import json
import logging
from typing import Optional

from src.agents.base import BaseAgent, AgentStep, AgentState, Tool, ToolResult

logger = logging.getLogger(__name__)


class DrugAnalysisAgent(BaseAgent):
    """
    Single agent for comprehensive drug analysis.
    
    Handles: "Tell me everything about [drug]" queries.
    
    Plan:
    1. Safety check (always first — safety-critical info surfaces first)
    2. RAG search for indications and dosing
    3. Clinical trial search for ongoing studies
    4. Synthesize findings
    """

    PLANNER_PROMPT = """You are a drug analysis planning agent. Given a user task about 
a drug, create an execution plan.

Available tools:
{tool_descriptions}

Task: {task}

Create a JSON array of steps. Each step has:
- "action": what to do
- "tool": which tool to use
- "params": parameters for the tool
- "reasoning": why this step is needed

Return ONLY valid JSON array. Example:
[
  {{"action": "Check safety profile", "tool": "safety_check", "params": {{"drug_name": "Jardiance"}}, "reasoning": "Safety info must come first"}},
  {{"action": "Search indications", "tool": "rag_search", "params": {{"query": "Jardiance indications", "drug_name": "JARDIANCE"}}, "reasoning": "Core clinical information"}}
]"""

    REFLECTION_PROMPT = """Review your progress on this task.

Task: {task}
Steps completed: {step_count}
Findings so far:
{findings}

Questions:
1. Have we gathered enough information to answer the task comprehensively?
2. Is there safety-critical information we might have missed?
3. Should we continue searching or synthesize what we have?

Respond with JSON: {{"complete": true/false, "reasoning": "...", "missing": ["..."]}}"""

    async def plan(self, task: str) -> list[dict]:
        """Create an execution plan using LLM reasoning."""
        tool_desc = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )

        prompt = self.PLANNER_PROMPT.format(
            tool_descriptions=tool_desc, task=task
        )

        try:
            response = await self.model_router.generate(
                system_prompt="You are an expert drug analysis planner. Return only valid JSON.",
                user_prompt=prompt,
                temperature=0.0,
            )
            content = response.get("content", "[]")
            # Clean markdown fences if present
            content = content.strip().strip("```json").strip("```").strip()
            plan = json.loads(content)

            # Always ensure safety check is first
            has_safety = any(s.get("tool") == "safety_check" for s in plan)
            if not has_safety:
                # Extract drug name from task
                drug_name = self._extract_drug_name(task)
                if drug_name:
                    plan.insert(0, {
                        "action": "Safety check (auto-inserted)",
                        "tool": "safety_check",
                        "params": {"drug_name": drug_name},
                        "reasoning": "Safety information must always be checked first.",
                    })

            # Always end with synthesis
            has_synthesis = any(s.get("tool") == "synthesize" for s in plan)
            if not has_synthesis:
                plan.append({
                    "action": "Synthesize findings",
                    "tool": "synthesize",
                    "params": {"task": task, "output_format": "narrative"},
                    "reasoning": "Combine all findings into a coherent response.",
                })

            logger.info(f"Agent [{self.name}] created plan with {len(plan)} steps")
            return plan

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Plan generation failed: {e}. Using default plan.")
            return self._default_plan(task)

    async def execute_step(self, step_plan: dict) -> AgentStep:
        """Execute a single planned step by calling the appropriate tool."""
        tool_name = step_plan.get("tool", "")
        params = step_plan.get("params", {})
        action = step_plan.get("action", "Unknown action")
        reasoning = step_plan.get("reasoning", "")

        step = AgentStep(
            step_number=self.memory.step_count + 1,
            action=action,
            reasoning=reasoning,
            tool_name=tool_name,
            tool_input=params,
        )

        # Special handling for synthesize: inject accumulated findings
        if tool_name == "synthesize":
            params["findings"] = self.memory.findings
            params["task"] = self.memory.context.get("task", "")

        # Call the tool
        result = await self.call_tool(tool_name, **params)
        step.tool_result = result

        # Extract key findings from tool results
        if result.success and result.data:
            finding = self._extract_finding(tool_name, result.data)
            if finding:
                self.memory.add_finding(finding)

        step.tokens_used = result.token_cost
        return step

    async def reflect(self) -> bool:
        """Reflect on whether the task is complete."""
        # Quick checks first
        if self.memory.step_count < 2:
            return False

        # Check if synthesis was the last step (natural completion)
        last_step = self.memory.steps[-1]
        if last_step.tool_name == "synthesize" and last_step.tool_result and last_step.tool_result.success:
            return True

        # Use LLM for deeper reflection if we have many steps
        if self.memory.step_count >= 4 and self.model_router:
            try:
                findings_text = "\n".join(f"- {f[:200]}" for f in self.memory.findings)
                prompt = self.REFLECTION_PROMPT.format(
                    task=self.memory.context.get("task", ""),
                    step_count=self.memory.step_count,
                    findings=findings_text or "No findings yet.",
                )

                response = await self.model_router.generate(
                    system_prompt="You are evaluating agent progress. Return only valid JSON.",
                    user_prompt=prompt,
                    max_tokens=200,
                    temperature=0.0,
                )
                content = response.get("content", "{}").strip().strip("```json").strip("```").strip()
                result = json.loads(content)
                return result.get("complete", False)
            except Exception:
                pass

        return False

    def _extract_finding(self, tool_name: str, data) -> Optional[str]:
        """Extract a key finding from a tool result."""
        if isinstance(data, dict):
            if tool_name == "rag_search":
                docs = data.get("documents", [])
                if docs:
                    top = docs[0]
                    return f"[{top.get('citation', 'Source')}] {top.get('content', '')[:400]}"
            elif tool_name == "safety_check":
                sections = data.get("sections", {})
                parts = []
                if data.get("has_boxed_warning"):
                    parts.append("⚠️ HAS BOXED WARNING")
                for section, info in sections.items():
                    parts.append(f"[{section}] {info.get('content', '')[:300]}")
                return "\n".join(parts) if parts else None
            elif tool_name == "fda_label_lookup":
                labels = data.get("labels", [])
                if labels:
                    label = labels[0]
                    return f"FDA Label: {label.get('drug_name', '')} ({label.get('generic_name', '')})"
            elif tool_name == "clinical_trial_search":
                trials = data.get("trials", [])
                if trials:
                    summaries = [f"{t['nct_id']}: {t['title'][:100]} ({t['status']}, {t['phase']})" for t in trials[:3]]
                    return "Clinical trials found:\n" + "\n".join(summaries)
            elif tool_name == "synthesize":
                return data.get("synthesis", "")[:500]
        return str(data)[:300] if data else None

    def _extract_drug_name(self, task: str) -> Optional[str]:
        """Simple drug name extraction from task text."""
        # Common drug names to check
        task_upper = task.upper()
        common_drugs = [
            "JARDIANCE", "FARXIGA", "INVOKANA", "OZEMPIC", "TRULICITY",
            "MOUNJARO", "METFORMIN", "LIPITOR", "CRESTOR", "KEYTRUDA",
            "OPDIVO", "LISINOPRIL", "JANUVIA", "GLUCOPHAGE",
        ]
        for drug in common_drugs:
            if drug in task_upper:
                return drug
        # Return first capitalized word that might be a drug name
        words = task.split()
        for word in words:
            clean = word.strip("?.,!")
            if clean and clean[0].isupper() and len(clean) > 3:
                return clean
        return None

    def _default_plan(self, task: str) -> list[dict]:
        """Fallback plan if LLM planning fails."""
        drug_name = self._extract_drug_name(task) or "unknown"
        return [
            {"action": "Check safety profile", "tool": "safety_check",
             "params": {"drug_name": drug_name}, "reasoning": "Safety first"},
            {"action": "Search knowledge base", "tool": "rag_search",
             "params": {"query": task}, "reasoning": "Find relevant label information"},
            {"action": "Synthesize findings", "tool": "synthesize",
             "params": {"task": task, "output_format": "narrative"},
             "reasoning": "Combine findings into answer"},
        ]


class SafetyReviewAgent(BaseAgent):
    """
    Specialized agent focused exclusively on drug safety analysis.
    
    Handles: "Is [drug] safe for [population]?" type queries.
    Always checks: boxed warnings, contraindications, drug interactions, pregnancy.
    """

    async def plan(self, task: str) -> list[dict]:
        drug_name = self._extract_drug_name(task)
        plan = [
            {"action": "Retrieve all safety sections", "tool": "safety_check",
             "params": {"drug_name": drug_name or "unknown"},
             "reasoning": "Comprehensive safety profile is the primary deliverable"},
            {"action": "Check drug interactions", "tool": "rag_search",
             "params": {"query": f"{drug_name} drug interactions", "drug_name": drug_name,
                        "section_type": "drug_interactions"},
             "reasoning": "Interactions are a critical safety concern"},
            {"action": "Check special populations", "tool": "rag_search",
             "params": {"query": f"{drug_name} pregnancy pediatric geriatric",
                        "drug_name": drug_name,
                        "section_type": "use_in_specific_populations"},
             "reasoning": "Population-specific safety information"},
            {"action": "Generate safety brief", "tool": "synthesize",
             "params": {"task": task, "output_format": "safety_brief"},
             "reasoning": "Structured safety output"},
        ]
        return plan

    async def execute_step(self, step_plan: dict) -> AgentStep:
        tool_name = step_plan.get("tool", "")
        params = step_plan.get("params", {})

        step = AgentStep(
            step_number=self.memory.step_count + 1,
            action=step_plan.get("action", ""),
            reasoning=step_plan.get("reasoning", ""),
            tool_name=tool_name,
            tool_input=params,
        )

        if tool_name == "synthesize":
            params["findings"] = self.memory.findings
            params["task"] = self.memory.context.get("task", "")

        result = await self.call_tool(tool_name, **params)
        step.tool_result = result

        if result.success and result.data:
            finding = self._extract_safety_finding(tool_name, result.data)
            if finding:
                self.memory.add_finding(finding)

        step.tokens_used = result.token_cost
        return step

    async def reflect(self) -> bool:
        last_step = self.memory.steps[-1] if self.memory.steps else None
        if last_step and last_step.tool_name == "synthesize" and last_step.tool_result and last_step.tool_result.success:
            return True
        return False

    def _extract_safety_finding(self, tool_name: str, data) -> Optional[str]:
        if isinstance(data, dict):
            if tool_name == "safety_check":
                sections = data.get("sections", {})
                parts = []
                if data.get("has_boxed_warning"):
                    parts.append("⚠️ BOXED WARNING PRESENT")
                for sec, info in sections.items():
                    parts.append(f"⚠️ [{sec}]: {info.get('content', '')[:400]}")
                return "\n".join(parts) if parts else None
            elif tool_name == "rag_search":
                docs = data.get("documents", [])
                if docs:
                    return f"[{docs[0].get('citation', '')}] {docs[0].get('content', '')[:400]}"
            elif tool_name == "synthesize":
                return data.get("synthesis", "")[:500]
        return str(data)[:300] if data else None

    def _extract_drug_name(self, task: str) -> Optional[str]:
        task_upper = task.upper()
        for drug in ["JARDIANCE", "FARXIGA", "INVOKANA", "OZEMPIC", "METFORMIN",
                      "LIPITOR", "CRESTOR", "KEYTRUDA", "LISINOPRIL", "JANUVIA"]:
            if drug in task_upper:
                return drug
        words = task.split()
        for word in words:
            clean = word.strip("?.,!")
            if clean and clean[0].isupper() and len(clean) > 3:
                return clean
        return None


class ComparisonAgent(BaseAgent):
    """
    Agent specialized in cross-drug comparison analysis.
    
    Handles: "Compare [drug A] vs [drug B] for [aspect]" queries.
    """

    async def plan(self, task: str) -> list[dict]:
        drugs = self._extract_drug_names(task)
        aspect = self._extract_aspect(task)

        plan = []
        # Safety check for each drug first
        for drug in drugs[:4]:
            plan.append({
                "action": f"Safety check: {drug}",
                "tool": "safety_check",
                "params": {"drug_name": drug},
                "reasoning": f"Get safety profile for {drug}",
            })

        # Structured comparison
        plan.append({
            "action": "Structured comparison",
            "tool": "drug_comparison",
            "params": {"drug_names": drugs, "aspect": aspect},
            "reasoning": f"Side-by-side {aspect} comparison",
        })

        # Synthesis
        plan.append({
            "action": "Generate comparison report",
            "tool": "synthesize",
            "params": {"task": task, "output_format": "comparison_table"},
            "reasoning": "Structured comparison output",
        })

        return plan

    async def execute_step(self, step_plan: dict) -> AgentStep:
        tool_name = step_plan.get("tool", "")
        params = step_plan.get("params", {})

        step = AgentStep(
            step_number=self.memory.step_count + 1,
            action=step_plan.get("action", ""),
            reasoning=step_plan.get("reasoning", ""),
            tool_name=tool_name,
            tool_input=params,
        )

        if tool_name == "synthesize":
            params["findings"] = self.memory.findings
            params["task"] = self.memory.context.get("task", "")

        result = await self.call_tool(tool_name, **params)
        step.tool_result = result

        if result.success and result.data:
            if tool_name == "safety_check":
                drug = params.get("drug_name", "")
                has_bw = result.data.get("has_boxed_warning", False) if isinstance(result.data, dict) else False
                sections = result.data.get("sections", {}) if isinstance(result.data, dict) else {}
                finding = f"{drug}: {'⚠️ HAS BOXED WARNING. ' if has_bw else ''}{len(sections)} safety sections found."
                self.memory.add_finding(finding)
            elif tool_name == "drug_comparison":
                comparison = result.data.get("comparison", {}) if isinstance(result.data, dict) else {}
                for drug, data in comparison.items():
                    for sec, content in data.items():
                        self.memory.add_finding(f"[{drug} - {sec}]: {content[:300]}")
            elif tool_name == "synthesize":
                synth = result.data.get("synthesis", "") if isinstance(result.data, dict) else ""
                if synth:
                    self.memory.add_finding(synth[:500])

        step.tokens_used = result.token_cost
        return step

    async def reflect(self) -> bool:
        last = self.memory.steps[-1] if self.memory.steps else None
        if last and last.tool_name == "synthesize" and last.tool_result and last.tool_result.success:
            return True
        return False

    def _extract_drug_names(self, task: str) -> list[str]:
        known = ["JARDIANCE", "FARXIGA", "INVOKANA", "OZEMPIC", "TRULICITY",
                  "MOUNJARO", "METFORMIN", "LIPITOR", "CRESTOR", "KEYTRUDA",
                  "OPDIVO", "LISINOPRIL", "JANUVIA", "GLUCOPHAGE", "ZESTRIL"]
        found = [d for d in known if d in task.upper()]
        return found if found else ["Unknown"]

    def _extract_aspect(self, task: str) -> str:
        task_lower = task.lower()
        if any(w in task_lower for w in ["safe", "warning", "risk", "contraindic"]):
            return "safety"
        if any(w in task_lower for w in ["dos", "dose", "dosing"]):
            return "dosing"
        if any(w in task_lower for w in ["side effect", "adverse"]):
            return "adverse_reactions"
        if any(w in task_lower for w in ["interact", "combination"]):
            return "interactions"
        return "safety"
