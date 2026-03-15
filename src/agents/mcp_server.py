"""
Model Context Protocol (MCP) Server
Exposes the Clinical Document Intelligence Platform tools as an MCP server,
enabling integration with Claude, VS Code, and other MCP-compatible hosts.

Module 12: MCP
- MCP Architecture Overview: Client ↔ Server ↔ LLM
- Building MCP Servers: project structure, FastMCP, server lifecycle
- MCP Capabilities: tools, structured outputs, context objects
- MCP for Agentic AI Systems: using MCP as the tool layer

Transport: STDIO (local) or SSE (remote)

Usage:
    # STDIO mode (for Claude Desktop, VS Code)
    python -m src.agents.mcp_server

    # Or register in claude_desktop_config.json:
    {
        "mcpServers": {
            "clinical-intelligence": {
                "command": "python",
                "args": ["-m", "src.agents.mcp_server"],
                "cwd": "/path/to/clinical-doc-intel"
            }
        }
    }
"""
import json
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# MCP protocol constants
MCP_VERSION = "2024-11-05"
JSONRPC_VERSION = "2.0"


class MCPServer:
    """
    Lightweight MCP server implementing the core protocol.
    
    Capabilities exposed:
    - tools/list: Enumerate available clinical intelligence tools
    - tools/call: Execute a tool and return results
    
    This implementation uses STDIO transport for local integration.
    For production SSE/HTTP transport, wrap with FastAPI.
    """

    def __init__(self):
        self.tools = self._define_tools()
        self._vector_store = None
        self._retriever = None
        self._fda_client = None
        self._initialized = False

    def _define_tools(self) -> list[dict]:
        """Define MCP tool schemas."""
        return [
            {
                "name": "query_drug_labels",
                "description": (
                    "Search FDA drug labels for information about medications. "
                    "Returns citation-grounded answers from official FDA prescribing information."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Clinical question about a drug (e.g., 'What are the contraindications for Jardiance?')",
                        },
                        "drug_name": {
                            "type": "string",
                            "description": "Optional: filter to a specific drug name",
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional: filter by label section (contraindications, dosage_and_administration, adverse_reactions, warnings_and_cautions, drug_interactions, boxed_warning)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "compare_drugs",
                "description": (
                    "Compare FDA drug labels across multiple medications. "
                    "Provides side-by-side analysis of safety, dosing, or indications."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "drug_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of drug names to compare (2-5 drugs)",
                        },
                        "aspect": {
                            "type": "string",
                            "enum": ["safety", "dosing", "indications", "adverse_reactions", "interactions"],
                            "description": "What aspect to compare",
                        },
                    },
                    "required": ["drug_names", "aspect"],
                },
            },
            {
                "name": "safety_check",
                "description": (
                    "Retrieve comprehensive safety profile for a drug: black box warnings, "
                    "contraindications, major warnings, and REMS requirements."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "Drug name to check (brand or generic)",
                        },
                    },
                    "required": ["drug_name"],
                },
            },
            {
                "name": "list_available_drugs",
                "description": "List all drugs currently indexed in the knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "get_platform_status",
                "description": "Get the status of the Clinical Document Intelligence Platform including document count and available tools.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def _ensure_initialized(self):
        """Lazy-initialize platform components."""
        if self._initialized:
            return

        try:
            from src.retrieval.vector_store import VectorStoreManager
            from src.retrieval.hybrid_search import HybridRetriever
            from src.ingestion.fda_labels import FDALabelClient

            self._vector_store = VectorStoreManager()
            self._retriever = HybridRetriever(vector_store=self._vector_store)
            self._fda_client = FDALabelClient()
            self._initialized = True
            logger.info("MCP Server: Platform components initialized")
        except Exception as e:
            logger.error(f"MCP Server: Initialization failed: {e}")
            raise

    async def handle_message(self, message: dict) -> dict:
        """Process a single JSON-RPC message."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
        }

        handler = handlers.get(method)
        if handler:
            try:
                result = await handler(params)
                if msg_id is not None:
                    return {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "result": result}
                return None  # Notifications don't get responses
            except Exception as e:
                logger.error(f"MCP handler error for {method}: {e}")
                if msg_id is not None:
                    return {
                        "jsonrpc": JSONRPC_VERSION,
                        "id": msg_id,
                        "error": {"code": -32000, "message": str(e)},
                    }
                return None
        else:
            if msg_id is not None:
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            return None

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": "clinical-document-intelligence",
                "version": "1.0.0",
            },
        }

    async def _handle_initialized(self, params: dict) -> None:
        """Handle initialized notification."""
        logger.info("MCP Server: Client initialized")
        return None

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    async def _handle_tools_list(self, params: dict) -> dict:
        """Return available tools."""
        return {"tools": self.tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Execute a tool call."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        self._ensure_initialized()

        tool_handlers = {
            "query_drug_labels": self._tool_query,
            "compare_drugs": self._tool_compare,
            "safety_check": self._tool_safety,
            "list_available_drugs": self._tool_list_drugs,
            "get_platform_status": self._tool_status,
        }

        handler = tool_handlers.get(tool_name)
        if not handler:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        try:
            result_text = await handler(arguments)
            return {"content": [{"type": "text", "text": result_text}]}
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Tool error: {str(e)}"}],
                "isError": True,
            }

    async def _tool_query(self, args: dict) -> str:
        """Execute drug label query."""
        query = args.get("query", "")
        drug_name = args.get("drug_name")
        section = args.get("section")

        result = self._retriever.retrieve(
            query=query,
            drug_name=drug_name,
            section_type=section,
        )

        if not result.documents:
            return f"No results found for: {query}"

        output_parts = [f"Results for: {query}\n"]
        for i, doc in enumerate(result.top_documents[:5], 1):
            output_parts.append(f"\n--- Result {i} ---")
            output_parts.append(f"Drug: {doc.drug_name}")
            output_parts.append(f"Section: {doc.metadata.get('section_display_name', 'N/A')}")
            output_parts.append(f"Citation: {doc.citation}")
            output_parts.append(f"Score: {doc.rerank_score or doc.score:.3f}")
            output_parts.append(f"Content:\n{doc.content[:600]}")

        return "\n".join(output_parts)

    async def _tool_compare(self, args: dict) -> str:
        """Execute drug comparison."""
        drug_names = args.get("drug_names", [])
        aspect = args.get("aspect", "safety")

        from src.agents.tools import DrugComparisonTool
        tool = DrugComparisonTool(self._retriever)
        result = await tool.execute(drug_names=drug_names, aspect=aspect)

        if not result.success:
            return f"Comparison failed: {result.error}"

        comparison = result.data.get("comparison", {})
        output_parts = [f"Comparison: {', '.join(drug_names)} — {aspect}\n"]

        for drug, sections in comparison.items():
            output_parts.append(f"\n=== {drug} ===")
            for section, content in sections.items():
                output_parts.append(f"\n[{section}]")
                output_parts.append(content[:500])

        return "\n".join(output_parts)

    async def _tool_safety(self, args: dict) -> str:
        """Execute safety check."""
        drug_name = args.get("drug_name", "")

        from src.agents.tools import SafetyCheckTool
        tool = SafetyCheckTool(self._retriever)
        result = await tool.execute(drug_name=drug_name)

        if not result.success:
            return f"Safety check failed: {result.error}"

        data = result.data
        parts = [f"Safety Profile: {drug_name}\n"]

        if data.get("has_boxed_warning"):
            parts.append("⚠️ THIS DRUG HAS A BOXED WARNING ⚠️\n")

        for section, info in data.get("sections", {}).items():
            parts.append(f"\n--- {section.replace('_', ' ').title()} ---")
            parts.append(info.get("content", "N/A")[:600])
            parts.append(f"Source: {info.get('citation', 'N/A')}")

        if not data.get("sections"):
            parts.append(f"No safety information found for {drug_name} in the knowledge base.")

        return "\n".join(parts)

    async def _tool_list_drugs(self, args: dict) -> str:
        """List available drugs."""
        drugs = self._vector_store.get_all_drug_names()
        if drugs:
            return f"Available drugs ({len(drugs)}):\n" + "\n".join(f"- {d}" for d in drugs)
        return "No drugs indexed. Run the ingestion pipeline first."

    async def _tool_status(self, args: dict) -> str:
        """Get platform status."""
        doc_count = self._vector_store.get_document_count()
        drugs = self._vector_store.get_all_drug_names()
        return (
            f"Clinical Document Intelligence Platform\n"
            f"Status: {'Active' if self._initialized else 'Not initialized'}\n"
            f"Documents indexed: {doc_count}\n"
            f"Unique drugs: {len(drugs)}\n"
            f"Tools available: {len(self.tools)}\n"
            f"MCP Protocol: {MCP_VERSION}"
        )


async def run_stdio():
    """Run MCP server over STDIO transport."""
    import asyncio

    server = MCPServer()

    async def read_message() -> Optional[dict]:
        """Read a JSON-RPC message from stdin."""
        loop = asyncio.get_event_loop()
        try:
            # Read Content-Length header
            header = await loop.run_in_executor(None, sys.stdin.readline)
            if not header:
                return None

            # Handle content-length based protocol
            if header.startswith("Content-Length:"):
                length = int(header.split(":")[1].strip())
                # Skip empty line
                await loop.run_in_executor(None, sys.stdin.readline)
                # Read body
                body = await loop.run_in_executor(None, lambda: sys.stdin.read(length))
                return json.loads(body)
            else:
                # Try parsing as raw JSON line
                line = header.strip()
                if line:
                    return json.loads(line)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse message: {e}")
        return None

    def write_message(msg: dict):
        """Write a JSON-RPC message to stdout."""
        if msg is None:
            return
        body = json.dumps(msg)
        sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
        sys.stdout.flush()

    logger.info("MCP Server starting on STDIO...")

    while True:
        message = await read_message()
        if message is None:
            break

        response = await server.handle_message(message)
        if response:
            write_message(response)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_stdio())
