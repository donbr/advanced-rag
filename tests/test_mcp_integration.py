import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

# Import the FastMCP instance from the project
from src.mcp_server import mcp  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_mcp_server_starts_and_lists_tools():
    """Spin up the FastMCP instance in-process and assert expected tools."""

    expected_tool_names = {
        "naive_retriever",
        "bm25_retriever",
        "contextual_compression_retriever",
        "multi_query_retriever",
        "parent_document_retriever",
        "ensemble_retriever",
        "semantic_retriever",
    }

    async with create_connected_server_and_client_session(mcp) as (_server, client):
        # Ensure the server initialises correctly
        tool_list = await client.list_tools()
        registered_names = {tool.name for tool in tool_list.tools}

        # At least the core tools must be present
        missing = expected_tool_names - registered_names
        assert not missing, f"Missing expected MCP tools: {missing}"

        # Minimal sanity check on server metadata
        assert tool_list.server.name == "advanced_rag" 