"""mcp_server.py

Expose Advanced RAG Retriever chains as a Model Context Protocol (MCP) server.
This provides seven MCP tools (one per retrieval chain) that mirror the FastAPI
endpoints in ``src.main_api``.  Run with ``uv run src/mcp_server.py`` or
``python -m src.mcp_server``.  The server uses STDIO transport by default so
it can be registered with Claude Desktop or inspected via ``mcp dev``.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.chain_factory import (
    NAIVE_RETRIEVAL_CHAIN,
    BM25_RETRIEVAL_CHAIN,
    CONTEXTUAL_COMPRESSION_CHAIN,
    MULTI_QUERY_CHAIN,
    PARENT_DOCUMENT_CHAIN,
    ENSEMBLE_CHAIN,
    SEMANTIC_CHAIN,
)
from src.main_api import invoke_chain_logic  # Re-use existing async helper
from src import logging_config  # Ensure logging configured early

# ---------------------------------------------------------------------------
# Initialise MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("advanced_rag")


async def _run_chain(chain: Any, question: str, name: str) -> str:
    """Helper to execute a chain and return the answer string only.

    Parameters
    ----------
    chain : LCEL chain instance or ``None``
    question : str
        Natural-language question from the user.
    name : str
        Human-readable chain name (used for logging).
    """
    result = await invoke_chain_logic(chain, question, name)
    # ``invoke_chain_logic`` returns a Pydantic model instance.
    return result.answer


# ---------------------------------------------------------------------------
# MCP Tools – one per retrieval chain
# ---------------------------------------------------------------------------


@mcp.tool()
async def naive_retriever(question: str) -> str:
    """Answer questions using the Naive vector-similarity retriever."""

    return await _run_chain(NAIVE_RETRIEVAL_CHAIN, question, "Naive Retriever Chain")


@mcp.tool()
async def bm25_retriever(question: str) -> str:
    """Answer questions using the BM25 keyword retriever."""

    return await _run_chain(BM25_RETRIEVAL_CHAIN, question, "BM25 Retriever Chain")


@mcp.tool()
async def contextual_compression_retriever(question: str) -> str:
    """Answer questions using Contextual Compression (Cohere Rerank)."""

    return await _run_chain(
        CONTEXTUAL_COMPRESSION_CHAIN,
        question,
        "Contextual Compression Chain",
    )


@mcp.tool()
async def multi_query_retriever(question: str) -> str:
    """Answer questions using the Multi-Query retriever."""

    return await _run_chain(MULTI_QUERY_CHAIN, question, "Multi-Query Chain")


@mcp.tool()
async def parent_document_retriever(question: str) -> str:
    """Answer questions using the Parent-Document retriever."""

    return await _run_chain(PARENT_DOCUMENT_CHAIN, question, "Parent Document Chain")


@mcp.tool()
async def ensemble_retriever(question: str) -> str:
    """Answer questions using the Ensemble retriever (hybrid)."""

    return await _run_chain(ENSEMBLE_CHAIN, question, "Ensemble Chain")


@mcp.tool()
async def semantic_retriever(question: str) -> str:
    """Answer questions using the Semantic-chunked retriever."""

    return await _run_chain(SEMANTIC_CHAIN, question, "Semantic Chain")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the server using STDIO transport by default.  This is the easiest
    # way to connect from Claude Desktop or the ``mcp`` command-line tools.
    mcp.run(transport="stdio") 