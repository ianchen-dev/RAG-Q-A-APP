import os

from langchain_ollama import OllamaEmbeddings  # ollama本地模型
from langchain_openai import OpenAIEmbeddings

ONEAPI_BASE_URL = os.getenv("ONEAPI_BASE_URL")


def get_embedding(
    supplier: str, model_name: str, inference_api_key: str = None, chunk_size: int = 64
):
    """
    获取嵌入模型实例

    Args:
        supplier: 供应商名称 ('ollama', 'oneapi','siliconflow')
        model_name: 模型名称
        inference_api_key: API密钥
        chunk_size: 批次大小，默认64（适配OneAPI限制）

    Returns:
        嵌入模型实例
    """
    if supplier == "ollama":
        embeddings = OllamaEmbeddings(model=model_name)
    elif supplier == "oneapi":
        embeddings = OpenAIEmbeddings(
            base_url=ONEAPI_BASE_URL,
            model=model_name,
            api_key=inference_api_key,
            chunk_size=chunk_size,  # 设置批次大小限制
        )
    elif supplier == "siliconflow":
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn",
            model="BAAI/bge-m3",  # SiliconFlow支持的嵌入模型
            chunk_size=chunk_size,
        )
    # elif supplier == "openai":
    #     # OpenAI embedding模型，会自动从环境变量OPENAI_API_KEY获取密钥
    #     embeddings = OpenAIEmbeddings(model=model_name, chunk_size=chunk_size)
    else:
        raise ValueError("Invalid supplier or model name")
    return embeddings
