import os
import streamlit as st
from typing import List, TypedDict, Dict, Any
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

# --- 0. 加载环境 ---
load_dotenv() 

# --- 1. UI 基础设置 ---
st.set_page_config(page_title="智视 X1 智能助手", layout="centered")
st.title("🤖 智视 X1 智能客服助手")
st.caption("基于 LangGraph 的自纠错 RAG 系统（支持多轮对话与主动反问）")

# 获取并检查 API Key
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

if not DEEPSEEK_KEY:
    st.error("❌ 未检测到 DEEPSEEK_API_KEY，请检查环境变量或 Streamlit Secrets 配置")
    st.stop()

# --- 2. 定义 GraphState 和节点逻辑 ---

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    source: str 
    chat_history: List[Dict[str, Any]] 

def retrieve(state: GraphState):
    """检索节点"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    
    if not os.path.exists(db_path):
        return {"documents": ["【系统提示】未发现本地数据库，请确保已上传 chroma_db 文件夹"]}

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": [d.page_content for d in docs], "source": "local"}

def transform_query(state: GraphState):
    """查询优化节点 - 核心修改：增加历史上下文感知"""
    llm = ChatOpenAI(model='deepseek-chat', api_key=DEEPSEEK_KEY, openai_api_base='https://api.deepseek.com', temperature=0)
    
    history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
    
    prompt = (
        f"任务：结合对话历史，将用户的最新简短回复重写为一个完整的、独立的检索问题。\n"
        f"1. 如果用户的回复是针对上文反问的回答（例如上文问'安装在哪里'，用户回'玻璃上'），请将其合并为完整问题（如'智视 X1 如何安装在玻璃上'）。\n"
        f"2. 如果用户的回复是全新的问题，请忽略历史，仅优化当前问题关键词。\n\n"
        f"对话历史：\n{history_context}\n"
        f"用户最新回复：{state['question']}\n\n"
        f"重写后的完整问题（仅输出文本，不要解释）："
    )
    
    better_question = llm.invoke(prompt).content
    print(f"【Debug】优化后的问题: {better_question}")
    return {"question": better_question, "retry_count": state.get("retry_count", 0) + 1}

def web_search(state: GraphState):
    """联网搜索节点"""
    if not TAVILY_KEY:
        st.error("❌ 联网搜索失败：未检测到 TAVILY_API_KEY")
        return {"documents": ["TAVILY_API_KEY 未配置"], "source": "web"}
        
    search = TavilySearchResults(max_results=3, api_key=TAVILY_KEY)
    try:
        search_results = search.invoke(state["question"])
        
        if isinstance(search_results, list):
            web_content = "\n".join([
                d['content'] if isinstance(d, dict) and 'content' in d else str(d) 
                for d in search_results
            ])
        else:
            web_content = str(search_results)
            
        if not web_content.strip():
            web_content = "联网搜索未找到结果。"
            
    except Exception as e:
        st.warning(f"⚠️ 联网搜索 API 调用异常: {str(e)}")
        web_content = f"调用失败: {str(e)}"
        
    return {"documents": [web_content], "source": "web"}

def decide_to_generate(state: GraphState):
    """决策节点"""
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    docs_text = "\n".join(state["documents"])
    
    grader_prompt = (
        f"问题: {state['question']}\n文档: {docs_text}\n"
        "判断文档是否足以回答问题？回复 YES, NO 或 CLARIFY。"
    )
    score = llm.invoke(grader_prompt).content.strip().upper()

    if "YES" in score or "CLARIFY" in score:
        return "generate"
   
    if state.get("retry_count", 0) < 1:
        return "transform_query"
    return "web_search"

def generate(state: GraphState):
    """生成回答"""
    llm = ChatOpenAI(model='deepseek-chat', openai_api_base='https://api.deepseek.com', api_key=DEEPSEEK_KEY, temperature=0)
    source = state.get("source", "local")
    
    prefix = "【💡 本地知识库回答】\n" if source == "local" else "【⚠️ 联网搜索结果（仅供参考，请以说明书实物为准）】\n"
    
    system_rules = (
        "你是一个专业的智能摄像机客服。准则：\n"
        "1. 信息不足或问题模糊时，必须礼貌地反问用户补充细节，严禁编造。\n"
        "2. 优先基于上下文回答。"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("human", "上下文: {context}\n问题: {question}")
    ])
    response = (prompt | llm).invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": prefix + response.content}

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            
            inputs = {
                "question": prompt, 
                "retry_count": 0, 
                "documents": [],
                "chat_history": st.session_state.messages[:-1] 
            }
            
            result = langgraph_app.invoke(inputs)
            response_text = result["generation"]
            st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
