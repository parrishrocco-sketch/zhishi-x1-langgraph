import os
import streamlit as st
from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

# --- 1. UI 基础设置 ---
st.set_page_config(page_title="智视 X1 智能助手", layout="centered")
st.title("🤖 智视 X1 智能客服助手")
st.caption("基于 LangGraph 的自纠错 RAG 系统")

# 获取 API Key (修复变量名大小写不一致问题)
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

if not DEEPSEEK_KEY:
    st.error("❌ 未检测到 DEEPSEEK_API_KEY，请在 Streamlit Secrets 中配置")
    st.stop()

# --- 2. 定义 GraphState 和节点逻辑 ---

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    source: str

def retrieve(state: GraphState):
    """检索节点"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    
    if not os.path.exists(db_path):
        return {"documents": ["【系统提示】未发现本地数据库，请确保已运行 Ingest.py 且上传了 chroma_db 文件夹"]}

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": [d.page_content for d in docs]}

def transform_query(state: GraphState):
    llm = ChatOpenAI(model='deepseek-chat', api_key=DEEPSEEK_KEY, openai_api_base='https://api.deepseek.com', temperature=0)
    prompt = f"请优化此搜索词以获得更好结果: {state['question']}\n只输出优化后的文本。"
    better_question = llm.invoke(prompt).content
    return {"question": better_question, "retry_count": state.get("retry_count", 0) + 1}

def web_search(state: GraphState):
    search = TavilySearchResults(max_results=3, api_key=TAVILY_KEY)
    search_results = search.invoke(state["question"])
    web_content = "\n".join([d['content'] for d in search_results])
    return {"documents": [web_content], "source": "web"}

def decide_to_generate(state: GraphState):
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    docs_text = "\n".join(state["documents"])
    grader_prompt = f"问题: {state['question']}\n文档: {docs_text}\n请判断文档是否足以回答问题？仅回复 YES 或 NO。"
    score = llm.invoke(grader_prompt).content.strip().upper()
    if "YES" in score: return "generate"
    if state.get("retry_count", 0) >= 1: return "web_search"
    return "transform_query"

def generate(state: GraphState):
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    source = state.get("source", "local")
    prefix = "【💡 本地知识库回答】\n" if source == "local" else "【⚠️ 联网搜索结果】\n"
    system_rules = "你是一个严谨的产品专家。如果信息缺失，请主动提问。严禁编造。"
    prompt = ChatPromptTemplate.from_messages([("system", system_rules), ("human", "上下文: {context}\n问题: {question}")])
    response = (prompt | llm).invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": prefix + response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges("retrieve", decide_to_generate, {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
langgraph_app = workflow.compile()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请提问，例如：智视 X1 的分辨率是多少？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考并检索知识库..."):
            inputs = {"question": prompt, "retry_count": 0, "documents": []}
            result = langgraph_app.invoke(inputs)
            response = result["generation"]
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})