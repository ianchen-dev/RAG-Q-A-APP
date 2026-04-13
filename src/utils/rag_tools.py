from fastapi import APIRouter

# Import tools and models from the new src.tools package
from src.tools import (
    RAGRequest,
    KnowledgeConfig,
    RerankerConfig,
    retriever_document_tool,
)

testRouter = APIRouter()


# 接口测试-成功✓
knowledge_config_instance = KnowledgeConfig(
    knowledge_base_id="67ff248e0e67faaaae7c5303",
    search_k=10,
    use_bm25=True,
    bm25_k=3,
    reranker_config=RerankerConfig(
        use_reranker=True,
        reranker_type="remote",
    ),
)
rag_request_instance = RAGRequest(
    question="RAG是什么？", knowledge_config=knowledge_config_instance
)

# 构建工具期望的输入字典
tool_input = {"request": rag_request_instance}


@testRouter.post("/retriever_document_tool")
async def main_async():
    # 使用 ainvoke (因为 get_rag_service 是 async def)
    result = await retriever_document_tool.ainvoke(tool_input)
    print(result)
    return result
