import random

from locust import HttpUser, between, task


class RagUser(HttpUser):
    # 定义用户行为间的等待时间 (模拟用户思考/操作间隔)
    wait_time = between(1, 3)  # 用户执行每个任务后会等待1-3秒

    # host 属性定义了测试目标的基础 URL
    # 也可以在启动 Locust 时通过 --host 参数指定
    # host = "http://localhost:8081"

    @task  # @task 装饰器标记这是一个用户任务
    def get_knowledge_bases(self):
        """模拟用户获取知识库列表"""
        self.client.get("/knowledge/")  # 向 /knowledge/ 发送 GET 请求

    @task(3)  # 数字表示任务权重，权重为3的任务被选中的概率是权重为1的3倍
    def post_chat_message(self):
        """模拟用户发送聊天消息"""
        # 准备请求体 (根据你的API实际情况调整)
        session_id = f"session_{random.randint(1, 100)}"  # 模拟不同的会话
        payload = {
            "query": "你好，请介绍一下 RAG 技术",
            "knowledge_base_id": "your_knowledge_base_id",  # 替换为有效的知识库ID
            "model_name": "default_model",  # 根据需要调整
            "history_len": 5,
            "temperature": 0.7,
            "prompt_name": "default",
        }
        headers = {"Content-Type": "application/json"}

        # 向 /chat/{session_id} 发送 POST 请求 (假设你的路由是这样设计的)
        # 请根据你的 router/chatRouter.py 调整 URL 和参数传递方式
        # 例如，如果 session_id 在 URL 路径中:
        # self.client.post(f"/chat/{session_id}", json=payload, headers=headers)
        # 如果 session_id 在 body 中:
        # payload["session_id"] = session_id
        # self.client.post("/chat/", json=payload, headers=headers)

        # **重要:** 请根据你实际的 chatRouter.py 实现来修改这里的 URL 和 payload
        # 这里仅为示例，你需要替换为正确的 API 端点和请求体结构
        # 假设 session_id 是 body 的一部分:
        payload["session_id"] = session_id
        # 假设聊天端点是 /chat/stream 或者 /chat/request (具体看你的路由)
        # 假设是 /chat/request
        self.client.post(
            "/chat/request",
            json=payload,
            headers=headers,
            name="/chat/request [example]",
        )  # 使用 name 参数可以更好地在 Locust UI 中聚合统计

    # 可以添加更多 @task 来模拟其他操作，如文件上传等
    # @task
    # def upload_file(self):
    #     kb_id = "your_knowledge_base_id" # 需要一个有效的知识库ID
    #     files = {'files': ('example.txt', b'file content here', 'text/plain')}
    #     self.client.post(f"/knowledge/{kb_id}/upload", files=files)

    def on_start(self):
        """用户启动时执行的操作，例如登录 (如果需要认证)"""
        # print("A new user has started")
        pass

    def on_stop(self):
        """用户停止时执行的操作"""
        # print("A user has stopped")
        pass
