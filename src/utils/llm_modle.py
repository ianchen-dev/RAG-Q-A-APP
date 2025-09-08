import os

from langchain_ollama import ChatOllama

# llm
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

ONEAPI_BASE_URL = os.getenv("ONEAPI_BASE_URL")


def get_llms(
    supplier: str,
    model: str,
    api_key: str = None,
    max_length: int = 10,
    temperature: float = 0.8,
    streaming: bool = True,
):
    try:
        """
        获取LLM模型
        """
        if supplier == "openai":
            return ChatOpenAI(model=model, temperature=temperature, streaming=streaming)
        elif supplier == "siliconflow":
            return BaseChatOpenAI(
                # model=os.getenv("SILICONFLOW_MODEL"),
                model="deepseek-ai/DeepSeek-V3",  #
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                openai_api_base="https://api.siliconflow.cn/v1",
                streaming=streaming,
            )
        elif supplier == "volces":
            return BaseChatOpenAI(
                model="deepseek-v3-250324",
                openai_api_key=os.getenv("VOLCES_API_KEY"),
                openai_api_base="https://ark.cn-beijing.volces.com/api/v3/",
                # streaming=True,
                stream=True
                # max_tokens=int(os.getenv("MAX_TOKENS")),
            )

        elif supplier == "ollama":
            return ChatOllama(model=model)
        elif supplier == "oneapi":
            return ChatOpenAI(
                api_key=api_key,
                base_url=ONEAPI_BASE_URL,
                model=model,
                temperature=temperature,
                streaming=streaming,
            )
        else:
            raise ValueError(f"Unsupported supplier: {supplier}")
    except Exception as e:
        print(f"Error: {e}")
        raise ConnectionError(f"Error: {e}")


if __name__ == "__main__":
    model = "deepseek-r1:latest"  # 使用DeepSeek聊天模型
    llm = get_llms(supplier="ollama", model=model, max_length=10086)
    print(llm.invoke("你好"))
