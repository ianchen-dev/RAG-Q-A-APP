# import getpass
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true" #LangSmith
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 2. Create model
# model = ChatOpenAI(os.environ["OPENAI_API_KEY"])
from langchain_openai.chat_models.base import BaseChatOpenAI

model = BaseChatOpenAI(
    model=os.environ["MODEL"],  # 使用DeepSeek聊天模型
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    max_tokens=os.environ["MAX_TOKENS"],
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser
# // 历史消息测试
from src.utils.with_msg_history import with_message_history

config = {"configurable": {"session_id": "abc2"}}
with_message_history = with_message_history(chain)
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

response.content


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

# Integration tests have been migrated to test/integration/router/test_myllm.py
# Run with: uv run pytest test/integration/router/test_myllm.py -v
