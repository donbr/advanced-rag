from src.chain_factory import RAG_TEMPLATE_STR


def test_rag_template_contains_placeholders():
    """Verify the core template still contains the required placeholders."""

    assert "{question}" in RAG_TEMPLATE_STR, "RAG template must have {question} placeholder"
    assert "{context}" in RAG_TEMPLATE_STR, "RAG template must have {context} placeholder" 