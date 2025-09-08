import json
import logging


def utils_format_doc_list(context):
    processed_context = []
    if isinstance(context, list):
        for doc in context:
            try:
                # 使用 model_dump() 获取字典表示，而不是 model_dump_json()
                # exclude_none=True 可以使输出更简洁，不包含值为 None 的字段
                doc_dict = doc.model_dump(exclude_none=True)
                processed_context.append(doc_dict)
            except AttributeError as e:
                # 如果 Document 对象没有 model_dump 方法 (例如，如果它不是 Pydantic 模型)
                # 或者发生其他与序列化相关的错误，则记录警告并尝试回退
                logging.warning(
                    f"尝试对 Document 对象调用 model_dump() 时出错: {e}. 文档内容: {doc}. 将尝试手动提取。"
                )
                if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    processed_context.append(
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                    )
                else:
                    processed_context.append(
                        {
                            "error": "Invalid document structure for fallback",
                            "original_doc": str(doc),
                        }
                    )
            except Exception as e:
                logging.error(
                    f"序列化 Document 对象时发生未知错误: {e}. 文档内容: {doc}"
                )
                processed_context.append(
                    {
                        "error": "Unknown serialization error",
                        "original_doc": str(doc),
                    }
                )
        else:
            logging.warning(f"context is not a list as expected: {context}")
            # 根据需要，可以决定在这种情况下 processed_context 应该是什么
            # 例如，如果 context_part 本身就是单个字典，可以直接添加，或者包装在列表中
            if isinstance(context, dict):  # 简单处理 context 是单个字典的情况
                processed_context.append(context)
            else:
                processed_context.append(
                    {
                        "error": "Context is not a list",
                        "original_context": str(context),
                    }
                )

        # 将 processed_context (字典列表) 转换为格式化的 JSON 字符串
        return json.dumps(processed_context, indent=2, ensure_ascii=False)
