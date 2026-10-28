import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from pdf_parser import PDFParser
from vector_store import VectorStore
from agent_core import PaperReadingAgent

load_dotenv()

st.set_page_config(
    page_title="AI 论文阅读助手 Agent",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI 论文阅读助手 Agent")
st.markdown("基于 LangGraph 的智能论文阅读与分析助手 | 支持 PDF 解析、RAG 检索、多步骤 Agent 分析")

# Initialize components
@st.cache_resource
def init_components(api_key: str = None, base_url: str = None, model_name: str = None, embed_model: str = None):
    parser = PDFParser()
    vector_store = VectorStore(api_key=api_key, base_url=base_url)
    agent = PaperReadingAgent(vector_store=vector_store, api_key=api_key, base_url=base_url, model_name=model_name)
    return parser, vector_store, agent

# Sidebar
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    base_url = st.text_input("API Base URL", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name = st.selectbox("模型选择", ["gpt-4o", "gpt-4o-mini"], index=0)
    embed_model = st.selectbox("Embedding 模型", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["MODEL_NAME"] = model_name
    os.environ["EMBEDDING_MODEL"] = embed_model
    
    if st.button("🔄 重置组件（应用新配置）"):
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 上传 PDF 论文（支持文字型 PDF）
    2. 输入研究问题或阅读目标
    3. 点击开始分析，Agent 将自动完成：
       - 制定阅读计划（Plan）
       - 深度分析论文（Analyze）
       - 批判性评议（Critique）
       - 生成阅读报告（Report）
    """)
    
    st.divider()
    st.caption("提示：扫描版 PDF（纯图片）需要先 OCR 处理")

# Initialize with current config
parser, vector_store, agent = init_components(api_key=api_key, base_url=base_url, model_name=model_name, embed_model=embed_model)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 上传论文")
    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file is not None:
        current_file = st.session_state.get("current_file_name")
        if current_file != uploaded_file.name:
            try:
                with st.spinner("正在清空旧数据并加载新论文..."):
                    vector_store.clear_collection()
                st.session_state["current_file_name"] = uploaded_file.name
            except Exception as e:
                st.warning(f"清空向量库时出现警告: {str(e)}")
        
        st.success(f"已上传: {uploaded_file.name}")
        try:
            file_path = parser.save_uploaded_file(uploaded_file)
            with st.spinner("正在解析 PDF..."):
                paper_content = parser.parse_pdf(file_path)
                st.session_state["paper_content"] = paper_content
                st.session_state["paper_name"] = uploaded_file.name
                with st.spinner("正在向量化并存入知识库..."):
                    vector_store.add_document(paper_content, {"filename": uploaded_file.name, "source": file_path})
            st.info(f"论文解析完成，共 {len(paper_content)} 字符 | 向量库文档数：{vector_store.get_count()}")
        except Exception as e:
            st.error(f"处理失败: {str(e)}")

with col2:
    st.header("❓ 研究问题")
    research_question = st.text_area(
        "请输入你的研究问题或阅读目标",
        placeholder="例如：这篇论文的核心贡献是什么？提出了什么方法？实验效果如何？存在哪些局限？",
        height=150,
        key="research_question_input"
    )
    if research_question:
        st.session_state["research_question"] = research_question

# Analysis section
st.divider()
if st.button("🚀 开始分析", type="primary", use_container_width=True):
    if "paper_content" not in st.session_state or not st.session_state.get("paper_content"):
        st.error("请先上传 PDF 论文")
    elif not st.session_state.get("research_question"):
        st.error("请输入研究问题或阅读目标")
    elif not api_key:
        st.error("请先在侧边栏配置 OpenAI API Key")
    else:
        with st.spinner("Agent 正在分析中...这可能需要 2-5 分钟，请耐心等待"):
            try:
                result = agent.run(
                    st.session_state["paper_content"],
                    st.session_state["research_question"],
                    thread_id=f"session_{uuid.uuid4().hex}"
                )
                st.session_state["result"] = result
                st.success("分析完成！")
            except Exception as e:
                st.error(f"分析出错: {str(e)}")
                st.caption("如遇 API Key、网络或模型错误，请检查侧边栏配置后重试。")

# Results section
if "result" in st.session_state:
    result = st.session_state["result"]
    
    st.divider()
    st.header("📊 分析结果")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 阅读计划", "🔍 深度分析", "💭 批判性评议", "📄 最终报告", "💬 完整对话"])
    
    with tab1:
        st.markdown(result.get("plan", ""))
    
    with tab2:
        st.markdown(result.get("analysis", ""))
    
    with tab3:
        st.markdown(result.get("critique", ""))
    
    with tab4:
        st.markdown(result.get("report", ""))
    
    with tab5:
        messages = result.get("messages", [])
        for msg in messages:
            try:
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        st.chat_message("user").write(msg.content)
                    elif msg.type == "ai":
                        st.chat_message("assistant").write(msg.content)
                elif hasattr(msg, "content"):
                    st.chat_message("assistant").write(msg.content)
            except Exception:
                continue

# Footer
st.divider()
st.markdown("Made with ❤️ by AI Paper Reading Agent | Powered by LangGraph & Streamlit | [GitHub](https://github.com/)")
