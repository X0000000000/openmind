# AI 论文阅读助手 Agent

基于 Python + Streamlit + LangGraph 的智能论文阅读与分析助手。支持上传 PDF 论文，自动制定阅读计划、深度分析、批判性评议并生成结构化阅读报告，并结合 ChromaDB + 向量检索（RAG）提升分析的针对性。

## 功能特点

- 📄 **PDF 解析**：支持上传并解析文字型 PDF 学术论文，自动识别疑似扫描版 PDF 并给出提示
- 🧠 **多步骤 Agent 工作流**：基于 LangGraph 编排的 Plan → Analyze → Critique → Report 四步流程
- 📋 **智能阅读计划**：根据论文内容和研究问题自动制定详细阅读计划
- 🔍 **深度分析 + RAG**：结合向量检索相关片段，提取研究背景、方法、实验、结论与局限
- 💭 **批判性评议**：从创新性、方法论、实验设计、逻辑性、研究价值等多角度客观评议
- 📊 **结构化阅读报告**：自动整合全流程结果，生成包含摘要、方法、发现、批判、启发、总结的完整报告
- 🗄️ **向量知识库**：使用 ChromaDB 持久化存储论文向量，支持单篇论文模式（上传新论文自动清空旧数据）
- ⚙️ **灵活配置**：支持自定义 API Key、Base URL、模型（gpt-4o/gpt-4o-mini）及 Embedding 模型

## 技术栈

| 技术 | 用途 |
| --- | --- |
| [Streamlit](https://streamlit.io/) | Web 界面框架 |
| [LangGraph](https://www.langchain.com/langgraph) | Agent 工作流编排 |
| [LangChain](https://www.langchain.com/) | LLM 应用框架（Core/OpenAI/Community/Text-Splitters） |
| [OpenAI API](https://platform.openai.com/) | 大语言模型与 Embedding |
| [ChromaDB](https://www.trychroma.com/) | 本地向量数据库（RAG） |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | PDF 文本解析 |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | 环境变量管理 |

## 项目结构

```text
ai-paper-reading-agent/
├── app.py                  # Streamlit 主界面（配置、上传、分析、结果展示）
├── agent_core.py           # Agent 核心逻辑（LangGraph 工作流：Plan/Analyze/Critique/Report）
├── pdf_parser.py           # PDF 解析模块（文本提取、扫描版检测）
├── vector_store.py         # 向量存储与 RAG 模块（ChromaDB + OpenAI Embeddings）
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档（本文件）
├── .env.example            # 环境变量示例
├── .gitignore              # Git 忽略配置
└── data/                   # 数据存储目录
    ├── uploads/            # 上传的 PDF 文件
    └── chroma_db/          # ChromaDB 向量数据库（持久化）
```

## 快速开始

### 1. 环境准备

确保已安装 [Python 3.10+](https://www.python.org/downloads/)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的配置：

```bash
cp .env.example .env  # Linux/macOS
# 或手动复制 .env.example 并重命名为 .env
```

`.env` 示例：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DB_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
COLLECTION_NAME=papers
```

> 注：如果使用第三方兼容 API（如 OneAPI、硅基流动等），只需修改 `OPENAI_BASE_URL` 和对应的模型名称即可。

### 4. 启动应用

```bash
streamlit run app.py
```

应用默认地址：`http://localhost:8501`

## 使用流程

1. **配置**：在侧边栏填入 API Key、选择模型（可点击“重置组件”应用新配置）
2. **上传论文**：上传文字型 PDF 文件，系统会自动解析、向量化并存入知识库（上传新文件会自动清空旧数据）
3. **提出问题**：在右侧输入研究问题或阅读目标（如“这篇论文的核心贡献是什么？”）
4. **开始分析**：点击“开始分析”，Agent 将依次执行 Plan → Analyze → Critique → Report
5. **查看结果**：在结果区切换标签页查看阅读计划、深度分析、批判性评议、最终报告及完整对话历史

## 注意事项

- **API Key 必需**：需要有效的 OpenAI（或兼容）API Key
- **PDF 类型**：仅支持文字型 PDF。若上传扫描版（纯图片）PDF，会提示需先 OCR 处理
- **单篇论文模式**：当前设计为单篇论文分析场景，每次上传新 PDF 会清空向量库，避免跨论文干扰
- **分析耗时**：完整分析通常需要 2-5 分钟，具体取决于论文篇幅、模型响应速度及网络状况
- **本地存储**：上传文件和向量库均保存在 `data/` 目录下，已在 `.gitignore` 中排除
- **模型选择**：建议优先使用 `gpt-4o` 获取更好的学术分析效果，`gpt-4o-mini` 更适合日常快速预览

## 常见问题

| 问题 | 解决方案 |
| --- | --- |
| `ModuleNotFoundError: langchain.text_splitter` | 已迁移到 `langchain-text-splitters`，执行 `pip install -r requirements.txt` 即可 |
| API Key 报错 | 检查侧边栏 Key 是否正确，或点击“重置组件（应用新配置）”重新初始化 |
| 向量检索混入旧论文内容 | 上传新 PDF 时会自动清空向量库；如需手动清空可刷新页面或重新上传 |
| 分析结果为空或不相关 | 确保上传的是文字型 PDF，且研究问题描述清晰具体 |

## 许可证

[MIT License](LICENSE)
