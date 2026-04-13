"""Knowledge base tool for agents.

This module provides a tool for agents to query the list of available
knowledge bases.
"""

from langchain_core.tools import tool

# Import the actual function from knowledgeSev
# The get_knowledge_list function remains in knowledgeSev because
# knowledgeRouter.py also uses it directly
from src.service.knowledgeSev import get_knowledge_list


@tool
async def get_knowledge_list_tool():
    """获取所有知识库列表：
    [
    {
        "_id": "67ff248e0e67faaaae7c5303",
        "title": "RAG知识库",
        "tag": [
            "RAG知识库"
        ],
        "description": "RAG知识库",
        "creator": "user1",
        "filesList": [
            {
                "file_md5": "1d49477ffcd597016fbdcf09e95c7e41",
                "file_path": "C:\\Users\\hbche\\AppData\\Local\\Temp\\tmpv71f5vt8_RAG-QA-PRD.pdf",
                "file_name": "RAG-QA-PRD.pdf",
                "upload_time": "2025-04-26T09:23:10.054000"
            },

            {
                "file_md5": "b7c89c84a612852fe4c2e1c6c1882530",
                "file_name": "RAG.md",
                "upload_time": "2025-05-04T19:00:25.540000"
            }

        ],
        "embedding_config": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_supplier": "oneapi",
            "embedding_apikey": "sk-enlDKhEcgGKyeJPx5b8c65Dc9d9b4842A24f5223A4Fb50C3"
        },
        "create_at": "2025-04-16T11:31:26.195000"
    }
    ]
    """

    return await get_knowledge_list()


__all__ = ["get_knowledge_list_tool"]
