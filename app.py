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
st.caption("基于 LangGraph 的自纠错 RAG 系统（已修复联网搜索逻辑）")

# 获取并检查 API Key
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
        return {"documents": ["【系统提示】未发现本地数据库，请确保已上传 chroma_db 文件夹"]}

    # 注意：确保 requirements.txt 已包含 sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": [d.page_content for d in docs], "source": "local"}

def transform_query(state: GraphState):
    """查询优化节点"""
    llm = ChatOpenAI(model='deepseek-chat', api_key=DEEPSEEK_KEY, openai_api_base='https://api.deepseek.com', temperature=0)
    prompt = f"请将此问题重写为更适合检索的关键词，如果是用户需求不明确，请保留核心词。问题: {state['question']}"
    better_question = llm.invoke(prompt).content
    return {"question": better_question, "retry_count": state.get("retry_count", 0) + 1}

def web_search(state: GraphState):
    """联网搜索节点 - 已修复 TypeError 并增加异常处理"""
    if not TAVILY_KEY:
        return {"documents": ["错误：未配置 TAVILY_API_KEY，无法执行联网搜索。"], "source": "web"}
        
    search = TavilySearchResults(max_results=3, api_key=TAVILY_KEY)
    try:
        # 使用 invoke 获取结果
        search_results = search.invoke(state["question"])
        
        # 兼容性检查：判断返回的是列表还是格式化字符串
        if isinstance(search_results, list):
            web_content = "\n".join([
                d['content'] if isinstance(d, dict) and 'content' in d else str(d) 
                for d in search_results
            ])
        else:
            web_content = str(search_results)
            
        # 如果搜索结果完全为空的处理
        if not web_content.strip():
            web_content = f"针对问题 '{state['question']}' 未能搜寻到有效互联网信息。"
            
    except Exception as e:
        web_content = f"联网搜索调用失败，错误详情: {str(e)}"
        
    return {"documents": [web_content], "source": "web"}

def decide_to_generate(state: GraphState):
    """决策节点：判断相关性或是否需要反问"""
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    docs_text = "\n".join(state["documents"])
    
    grader_prompt = (
        f"问题: {state['question']}\n文档: {docs_text}\n"
        "判断文档是否足以准确回答问题？回复 YES, NO 或 CLARIFY (问题太模糊需反问)。"
    )
    score = llm.invoke(grader_prompt).content.strip().upper()

    if "YES" in score or "CLARIFY" in score:
        return "generate"
    if state.get("retry_count", 0) >= 1:
        return "web_search"
    return "transform_query"

def generate(state: GraphState):
    """生成回答逻辑"""
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    source = state.get("source", "local")
    
    # 联网搜索免责声明
    prefix = "【💡 本地知识库回答】\n" if source == "local" else "【⚠️ 联网搜索结果（仅供参考，请以说明书实物为准）】\n"
    
    system_rules = (
        "你是一个专业的智视 X1 客服。必须遵守：\n"
        "1. 信息不足或问题模糊时，必须礼貌地反问用户补充细节，严禁编造。\n"
        "2. 优先基于上下文回答，确保参数准确。"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("human", "上下文: {context}\n问题: {question}")
    ])
    response = (prompt | llm).invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": prefix + response.content}

# --- 3. 构建工作流 ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges("retrieve", decide_to_generate, {
    "generate": "generate", 
    "transform_query": "transform_query", 
    "web_search": "web_search"
})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

langgraph_app = workflow.compile()

# --- 4. Streamlit 交互界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请提问，例如：智视 X1 怎么安装？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("系统正在处理中..."):
            inputs = {"question": prompt, "retry_count": 0, "documents": []}
            result = langgraph_app.invoke(inputs)
            response_text = result["generation"]
            st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})