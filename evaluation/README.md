# RAG 评估系统使用指南

基于 RAGAS 0.3.5+构建的可扩展 RAG 评估体系，支持配置文件驱动，集成 ChatSev 和 Knowledge 系统。

## 🚀 快速开始

### 环境准备

#### 1. 激活虚拟环境 (Windows PowerShell)

```powershell
# 进入项目根目录
cd D:\aProject\fastapi

# 激活uv虚拟环境
.\.venv\Scripts\Activate.ps1

# 验证环境激活成功
python --version
which python
```

#### 2. 安装依赖

```powershell
# 如果还未安装评估系统依赖
uv add ragas>=0.3.5
uv add datasets
uv add pandas
uv add pyyaml
```

#### 3. 环境变量配置

确保您的 `.env` 或 `.env.dev` 文件包含必要的 API 密钥：

```env
# .env.dev 文件示例
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
MONGODB_URL=your_mongodb_connection_string
MONGO_DB_NAME=your_database_name
# ... 其他必要的环境变量
```

## 📋 使用方法

### 方法一：使用配置文件运行完整评估

```powershell
# 使用示例配置文件运行评估
python evaluation/rag_evaluation.py --config evaluation/examples/sample_config.yaml

# 启用详细日志输出
python evaluation/rag_evaluation.py --config evaluation/examples/sample_config.yaml --verbose

# 仅验证配置不执行评估（干运行）
python evaluation/rag_evaluation.py --config evaluation/examples/sample_config.yaml --dry-run
```

### 方法二：快速测试模式

```powershell
# 运行内置的快速测试
python evaluation/rag_evaluation.py --quick-test

# 快速测试 + 详细日志
python evaluation/rag_evaluation.py --quick-test --verbose
```

### 方法三：自定义配置文件

1. 复制示例配置文件：

```powershell
cp evaluation/examples/sample_config.yaml my_evaluation_config.yaml
```

2. 编辑配置文件，修改以下关键参数：

   - `questions_path`: 问题文件路径
   - `ground_truths_path`: 真实答案文件路径
   - `contexts_path`: 上下文文件路径（可选）
   - `llm_config`: LLM 配置
   - `metrics`: 评估指标选择

3. 运行评估：

```powershell
python evaluation/rag_evaluation.py --config my_evaluation_config.yaml
```

## 📁 数据格式要求

### 问题文件 (JSON 格式)

```json
{
  "questions": [
    "什么是检索增强生成（RAG）？",
    "RAGAS评估框架的作用是什么？",
    "如何提高RAG系统的准确性？"
  ]
}
```

### 真实答案文件 (JSON 格式)

```json
{
  "ground_truths": [
    [
      "RAG是检索增强生成技术，结合了信息检索和文本生成。",
      "它通过先检索相关文档，再生成答案来提高准确性。"
    ],
    ["RAGAS是专门用于评估RAG系统性能的评估框架，提供多种评估指标。"],
    ["可以通过优化检索质量、改进重排序、调整提示词等方式提高RAG系统准确性。"]
  ]
}
```

### 上下文文件 (JSON 格式，可选)

```json
{
  "contexts": [
    [
      "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术...",
      "相关的技术文档或知识片段..."
    ],
    ["RAGAS评估框架的相关上下文信息..."]
  ]
}
```

## ⚙️ 配置文件详解

### 基本配置

```yaml
evaluation:
  project_name: "我的RAG评估项目"
  description: "项目描述"
  version: "1.0.0"
```

### 数据集配置

```yaml
dataset:
  type: "file" # 或 "knowledge_base"
  questions_path: "path/to/questions.json"
  ground_truths_path: "path/to/ground_truths.json"
  contexts_path: "path/to/contexts.json" # 可选
```

### LLM 配置

```yaml
llm_config:
  supplier: "siliconflow" # openai, siliconflow, volces, ollama, oneapi
  model: "deepseek-ai/DeepSeek-V3"
  api_key: "${SILICONFLOW_API_KEY}" # 从环境变量读取
  temperature: 0.1
```

### 评估指标配置

```yaml
metrics:
  - "ContextRelevance" # 上下文相关性
  - "ContextPrecision" # 上下文精确度
  - "ContextRecall" # 上下文召回率
  - "Faithfulness" # 忠实度
  - "AnswerRelevancy" # 答案相关性
```

### 输出配置

```yaml
output:
  results_dir: "evaluation/results"
  export_format: ["json", "csv", "html"]
  save_individual_scores: true
  generate_charts: true
```

## 📊 结果输出

评估完成后，结果将保存在 `evaluation/results/` 目录下：

- `evaluation_results_YYYYMMDD_HHMMSS.json`: 完整的评估结果
- `evaluation_results_YYYYMMDD_HHMMSS.csv`: 详细数据表格
- `evaluation_report_YYYYMMDD_HHMMSS.html`: 可视化 HTML 报告

### 控制台输出示例

```
============================================================
评估结果摘要
============================================================
项目: RAG系统评估示例
描述: 基于RAGAS的RAG系统性能评估示例项目
------------------------------------------------------------
各指标分数:
  ContextRelevance    : 0.8542
  ContextPrecision    : 0.7891
  ContextRecall       : 0.8234
  Faithfulness        : 0.9123
  AnswerRelevancy     : 0.8756
------------------------------------------------------------
平均分数: 0.8509
最佳指标: Faithfulness (0.9123)
最差指标: ContextPrecision (0.7891)
执行时间: 0:02:34.567890
答案生成成功率: 100.0% (5/5)
============================================================
```

## 🔧 高级用法

### 知识库模式

如果您想使用现有的知识库进行评估：

```yaml
dataset:
  type: "knowledge_base"
  knowledge_base:
    kb_id: "your_knowledge_base_id"
    search_k: 5
    filter_by_file_md5: null

knowledge_config:
  use_bm25: true
  bm25_k: 3
  reranker_config:
    use_reranker: true
    reranker_type: "remote"
    remote_rerank_config:
      api_key: "${SILICONFLOW_API_KEY}"
      model: "BAAI/bge-reranker-v2-m3"
    rerank_top_n: 3
```

### 批量评估

系统自动支持批量并发处理：

- 问题数量 ≤ 10: 使用串行模式
- 问题数量 > 10: 使用批量并发模式（批次大小=5）

## 🧪 测试验证

### 运行单元测试

```powershell
# 进入项目根目录并激活环境
cd D:\aProject\fastapi
.\.venv\Scripts\Activate.ps1

# 运行评估系统测试
python -m pytest test/test_evaluation_system.py -v

# 运行特定测试类
python -m pytest test/test_evaluation_system.py::TestConfigManagement -v
```

### 验证配置文件

```powershell
# 验证配置文件格式（不执行评估）
python evaluation/rag_evaluation.py --config your_config.yaml --dry-run
```

## 🚨 常见问题

### 1. 环境激活失败

```powershell
# 如果 Activate.ps1 无法执行，可能需要修改执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后重新激活
.\.venv\Scripts\Activate.ps1
```

### 2. RAGAS 导入错误

```powershell
# 安装或更新RAGAS
uv add ragas>=0.3.5

# 如果版本冲突，强制重新安装
uv remove ragas
uv add ragas>=0.3.5
```

### 3. API 密钥配置问题

- 确保 `.env.dev` 文件在项目根目录
- 检查环境变量名称是否正确
- 验证 API 密钥是否有效

### 4. 文件路径问题

- 使用相对路径时，确保相对于配置文件所在目录
- Windows 环境下使用正斜杠 `/` 或双反斜杠 `\\`

### 5. 内存不足

对于大型数据集，可以：

- 减少批次大小
- 分批次运行评估
- 减少评估指标数量

## 📝 日志查看

评估过程中的详细日志保存在：

- `evaluation/logs/fastapi_dev_YYYY-MM-DD.log`
- `evaluation/logs/fastapi_dev_all.log`
- `evaluation/logs/fastapi_dev_error.log`

查看实时日志：

```powershell
# 查看最新日志
Get-Content evaluation/logs/fastapi_dev_all.log -Tail 50

# 实时监控日志
Get-Content evaluation/logs/fastapi_dev_all.log -Wait
```

## 🤝 贡献指南

如果您想为这个项目做贡献：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

---

## 快速执行命令总结

```powershell
# 1. 激活环境
cd D:\aProject\fastapi
.\.venv\Scripts\Activate.ps1

# 2. 运行快速测试
python evaluation/rag_evaluation.py --quick-test --verbose

# 3. 运行完整评估
python evaluation/rag_evaluation.py --config evaluation/examples/sample_config.yaml --verbose

# 4. 验证配置
python evaluation/rag_evaluation.py --config your_config.yaml --dry-run
```

如有问题，请查看日志文件或联系开发团队。
