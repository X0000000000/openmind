import os
import uuid
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from vector_store import VectorStore


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    paper_content: str
    research_question: str
    plan: str
    analysis: str
    critique: str
    report: str
    context: str


class PaperReadingAgent:
    def __init__(self, vector_store: VectorStore = None, api_key: str = None, base_url: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.7,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
        )
        self.vector_store = vector_store or VectorStore(api_key=self.api_key, base_url=self.base_url)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("plan", self.plan_step)
        workflow.add_node("analyze", self.analyze_step)
        workflow.add_node("critique", self.critique_step)
        workflow.add_node("report", self.report_step)
        
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "analyze")
        workflow.add_edge("analyze", "critique")
        workflow.add_edge("critique", "report")
        workflow.add_edge("report", END)
        
        return workflow.compile()

    def plan_step(self, state: AgentState):
        """Step 1: Plan the reading approach"""
        system_prompt = """你是一个专业的论文阅读计划制定专家。
        你的任务是根据论文内容和研究问题，制定一个详细的阅读计划。
        
        请分析：
        1. 论文的核心问题和研究目标
        2. 需要重点关注的章节和要点
        3. 阅读顺序和重点
        4. 预期要解决的问题"""
        
        paper_excerpt = state['paper_content'][:4000] if state.get('paper_content') else ""
        
        user_prompt = f"""
        论文内容（节选）：
        {paper_excerpt}
        
        研究问题：
        {state['research_question']}
        
        请制定详细的阅读计划。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        
        return {
            "plan": response.content,
            "messages": [AIMessage(content=f"【阅读计划】\n{response.content}")]
        }

    def analyze_step(self, state: AgentState):
        """Step 2: Analyze the paper content"""
        # Get relevant context from vector store
        try:
            context_docs = self.vector_store.similarity_search(state['research_question'], k=6)
            context = "\n\n---\n\n".join([doc["content"] for doc in context_docs]) if context_docs else ""
        except Exception:
            context = ""
        
        system_prompt = """你是一个专业的学术论文分析专家。
        你的任务是深入分析论文内容，提取关键信息。
        
        请重点分析：
        1. 研究背景和动机
        2. 核心方法和理论
        3. 实验设计和结果
        4. 主要结论和贡献
        5. 局限性和未来工作
        
        请结合检索到的相关片段和论文内容，针对研究问题给出有依据的分析。"""
        
        paper_excerpt = state['paper_content'][:6000] if state.get('paper_content') else ""
        
        user_prompt = f"""
        研究问题：
        {state['research_question']}
        
        相关上下文（向量检索）：
        {context if context else "（未检索到相关片段）"}
        
        论文内容（节选）：
        {paper_excerpt}
        
        请进行深入分析，回答研究问题。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        
        return {
            "analysis": response.content,
            "context": context,
            "messages": [AIMessage(content=f"【深度分析】\n{response.content}")]
        }

    def critique_step(self, state: AgentState):
        """Step 3: Critique the paper"""
        system_prompt = """你是一个严谨的论文评议专家。
        你的任务是批判性地评估论文的质量。
        
        请从以下角度评议：
        1. 创新性：是否有新颖的想法或方法？
        2. 方法论：方法是否合理、严谨？
        3. 实验：实验设计是否充分？结果是否可靠？
        4. 逻辑性：论证是否清晰？
        5. 意义：研究价值和影响？
        6. 不足：存在哪些缺陷或改进空间？"""
        
        user_prompt = f"""
        分析结果：
        {state['analysis']}
        
        请给出客观、专业的批判性评议。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        
        return {
            "critique": response.content,
            "messages": [AIMessage(content=f"【批判性评议】\n{response.content}")]
        }

    def report_step(self, state: AgentState):
        """Step 4: Generate final report"""
        system_prompt = """你是一个专业的学术报告撰写专家。
        你的任务是整合所有信息，生成一份结构清晰的论文阅读报告。
        
        报告应包括：
        1. 摘要（概括要点）
        2. 研究背景与问题
        3. 核心方法
        4. 主要发现
        5. 批判性思考
        6. 启发与应用
        7. 总结"""
        
        user_prompt = f"""
        阅读计划：
        {state['plan']}
        
        深度分析：
        {state['analysis']}
        
        批判性评议：
        {state['critique']}
        
        研究问题：
        {state['research_question']}
        
        请生成最终的论文阅读报告。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        
        return {
            "report": response.content,
            "messages": [AIMessage(content=f"【最终报告】\n{response.content}")]
        }

    def run(self, paper_content: str, research_question: str, thread_id: str = None):
        thread_id = thread_id or f"session_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [],
            "paper_content": paper_content,
            "research_question": research_question,
            "plan": "",
            "analysis": "",
            "critique": "",
            "report": "",
            "context": ""
        }
        result = self.graph.invoke(initial_state, config=config)
        return result
