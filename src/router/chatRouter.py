import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models.knowledgeBase import KnowledgeBase
from src.service.ChatSev import ChatSev
from src.utils.embedding import get_embedding
from src.utils.Knowledge import Knowledge

ChatRouter = APIRouter()


class LLMConfig(BaseModel):
    supplier: Literal["ollma", "openai", "siliconflow", "oneapi"] = "oneapi"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    # max_length: Optional[int] = 10086


class RerankerConfig(BaseModel):
    """重排序器配置"""

    use_reranker: bool = Field(default=False, description="是否启用重排序")
    reranker_type: Literal["local", "remote"] = Field(
        default="remote",
        description="重排序器类型: 'local' (本地CrossEncoder) 或 'remote' (SiliconFlow API)",
    )

    remote_rerank_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="远程 Reranker (SiliconFlow) 配置，当 reranker_type='remote' 时需要。至少包含 'api_key' 和可选的 'model'。 例如: {'api_key': 'your_sf_key', 'model': 'BAAI/bge-reranker-v2-m3'}",
    )
    rerank_top_n: int = Field(default=3, ge=1, description="重排序后返回的文档数量")


class KnowledgeConfig(BaseModel):
    knowledge_base_id: str
    filter_by_file_md5: Optional[list[str]] = Field(
        default=None,
        description="文件MD5，用于指定检索的若干个文档，如果为None，则检索所选知识库中的全部文件",
    )
    search_k: Optional[int] = Field(
        default=10, ge=1, description="基础检索器返回的文档数量 (应 >= rerank_top_n)"
    )
    # --- BM25 相关配置 ---
    use_bm25: bool = Field(default=False, description="是否启用 BM25 混合检索")
    bm25_k: int = Field(default=3, ge=1, description="BM25 检索器返回的文档数量")
    # --- 重排序器配置 ---
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig, description="重排序器配置"
    )


class ChatConfig(BaseModel):
    chat_history_max_length: Optional[int] = Field(default=8, ge=0)
    prompt_override: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = "你好"
    session_id: str
    llm_config: LLMConfig
    chat_config: Optional[ChatConfig] = Field(default_factory=ChatConfig)
    knowledge_config: Optional[KnowledgeConfig] = None


async def get_chat_service(request: ChatRequest) -> ChatSev:
    """Dependency function to create ChatSev instance based on request config."""
    knowledge_instance: Optional[Knowledge] = None
    if request.knowledge_config:
        knowledge_cfg = request.knowledge_config
        try:
            kb = await KnowledgeBase.get(knowledge_cfg.knowledge_base_id)
            if kb.embedding_config:
                _embedding = get_embedding(
                    kb.embedding_config.embedding_supplier,
                    kb.embedding_config.embedding_model,
                    kb.embedding_config.embedding_apikey,
                )
                reranker_cfg = knowledge_cfg.reranker_config
                knowledge_instance = Knowledge(
                    _embeddings=_embedding,
                    splitter="hybrid",
                    # --- BM25 参数 ---
                    use_bm25=knowledge_cfg.use_bm25,
                    bm25_k=knowledge_cfg.bm25_k,
                    # --- Reranker 参数 ---
                    use_reranker=reranker_cfg.use_reranker,
                    reranker_type=reranker_cfg.reranker_type,
                    remote_rerank_config=reranker_cfg.remote_rerank_config,
                    rerank_top_n=reranker_cfg.rerank_top_n,
                )
            else:
                logging.warning(
                    f"未找到知识库 {knowledge_cfg.knowledge_base_id} 或其 embedding 配置。将不初始化 Knowledge 工具。"
                )
        except Exception as e:
            logging.error(
                f"错误：初始化 Knowledge 工具失败 ({e})。将不使用知识库。",
                exc_info=True,
            )
            knowledge_instance = None

    try:
        chat_sev = ChatSev(
            knowledge=knowledge_instance,
            prompt=request.chat_config.prompt_override if request.chat_config else None,
        )
        return chat_sev
    except Exception as e:
        logging.error(f"错误：初始化 ChatSev 失败 ({e})。", exc_info=True)
        raise HTTPException(status_code=500, detail=f"无法初始化聊天服务: {e}")


async def stream_response_generator(chat_sev: ChatSev, request_data: ChatRequest):
    """异步生成器，用于 StreamingResponse，产生 SSE 格式的事件。"""
    logging.info(f"开始为 session_id={request_data.session_id} 生成流式响应")
    try:
        async for chunk_dict in chat_sev.stream_chat(
            question=request_data.question,
            api_key=request_data.llm_config.api_key,
            supplier=request_data.llm_config.supplier,
            model=request_data.llm_config.model,
            session_id=request_data.session_id,
            knowledge_base_id=request_data.knowledge_config.knowledge_base_id
            if request_data.knowledge_config
            else None,
            filter_by_file_md5=request_data.knowledge_config.filter_by_file_md5
            if request_data.knowledge_config
            else None,
            search_k=request_data.knowledge_config.search_k
            if request_data.knowledge_config
            else 3,
            max_length=None,
            temperature=request_data.llm_config.temperature,
        ):
            yield f"data: {json.dumps(chunk_dict)}\n\n"
            logging.debug(
                f"Sent chunk: {chunk_dict['type']} for session {request_data.session_id}"
            )

    except Exception as e:
        logging.error(
            f"在 stream_response_generator 中发生错误 (session: {request_data.session_id}): {e}",
            exc_info=True,
        )
        error_payload = json.dumps(
            {"type": "error", "data": f"流处理中发生严重错误: {e}"}
        )
        yield f"data: {error_payload}\n\n"
    finally:
        logging.info(f"结束为 session_id={request_data.session_id} 的流式响应")


@ChatRouter.post(
    "/stream",
    summary="AI Chat (Streaming)",
    description="与 AI 进行流式对话，可选使用知识库和重排序。",
)
async def chat_stream_endpoint(
    request: ChatRequest, chat_sev: ChatSev = Depends(get_chat_service)
):
    """处理流式聊天请求。"""
    logging.info(
        f"接收到流式请求: session_id={request.session_id}, question='{request.question[:50]}...'"
    )
    if request.knowledge_config:
        logging.info(
            f", kb_id={request.knowledge_config.knowledge_base_id}, use_reranker={request.knowledge_config.reranker_config.use_reranker}, reranker_type='{request.knowledge_config.reranker_config.reranker_type}'"
        )
    return StreamingResponse(
        stream_response_generator(chat_sev, request), media_type="text/event-stream"
    )


# @ChatRouter.post(
#     "/",
#     summary="AI Chat (Non-Streaming)",
#     description="(旧) 与 AI 进行对话，可选使用知识库和重排序。推荐使用 /stream 端点。",
# )
# async def chat_endpoint(
#     request: ChatRequest, chat_sev: ChatSev = Depends(get_chat_service)
# ):
#     """
#     处理非流式聊天请求 (旧版，使用 invoke)。
#     """
#     logging.warning(
#         f"调用旧的非流式 /chat 端点 (session: {request.session_id})。建议迁移到 /stream。"
#     )

#     knowledge_base_id_to_use = (
#         request.knowledge_config.knowledge_base_id if request.knowledge_config else None
#     )
#     filter_md5_to_use = (
#         request.knowledge_config.filter_by_file_md5
#         if request.knowledge_config
#         else None
#     )
#     search_k_to_use = (
#         request.knowledge_config.search_k if request.knowledge_config else 3
#     )

#     try:
#         response = await chat_sev.invoke(
#             question=request.question,
#             session_id=request.session_id,
#             api_key=request.llm_config.api_key,
#             supplier=request.llm_config.supplier,
#             model=request.llm_config.model,
#             temperature=request.llm_config.temperature,
#             max_length=None,
#             knowledge_base_id=knowledge_base_id_to_use,
#             filter_by_file_md5=filter_md5_to_use,
#             search_k=search_k_to_use,
#         )

#         if "error" in response:
#             logging.error(
#                 f"聊天服务 invoke 返回错误 (session: {request.session_id}): {response['error']}"
#             )
#             raise HTTPException(
#                 status_code=500, detail=response.get("error", "聊天处理失败")
#             )

#         api_response = {}
#         if "answer" in response:
#             api_response["answer"] = response["answer"]
#         else:
#             logging.warning(f"聊天服务响应字典中缺少 'answer' 键: {response}")
#             raise HTTPException(status_code=500, detail="聊天服务未能生成答案")

#         if "context" in response and isinstance(response["context"], list):
#             processed_context = []
#             for doc in response["context"]:
#                 if isinstance(doc, Document):
#                     processed_context.append(
#                         {"page_content": doc.page_content, "metadata": doc.metadata}
#                     )
#                 elif (
#                     isinstance(doc, dict)
#                     and "page_content" in doc
#                     and "metadata" in doc
#                 ):
#                     processed_context.append(
#                         {
#                             "page_content": doc["page_content"],
#                             "metadata": doc["metadata"],
#                         }
#                     )
#                 else:
#                     logging.warning(f"context 列表中包含非预期对象: {type(doc)}")
#             if processed_context:
#                 api_response["context"] = processed_context

#         if "context_display_name" in response:
#             api_response["context_display_name"] = response["context_display_name"]

#         return api_response

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logging.error(
#             f"错误：调用非流式聊天服务时出错 (session: {request.session_id}): {e}",
#             exc_info=True,
#         )
#         raise HTTPException(status_code=500, detail=f"聊天处理失败: {e}")
