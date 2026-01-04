import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

api_key = os.getenv("DEEPSEEK_API_key")

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
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": [d.page_content for d in docs]}

def transform_query(state: GraphState):
    print("--- 节点：优化查询语句 ---")
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com', 
        temperature=0
    )
    
    better_q_prompt = f"请优化此搜索词以获得更好结果: {state['question']}\n只输出优化后的文本。"
    better_question = llm.invoke(better_q_prompt).content
    
    return {"question": better_question, "retry_count": state.get("retry_count", 0) + 1}

def web_search(state: GraphState):
    """联网搜索节点"""
    search = TavilySearchResults(max_results=3)
    search_results = search.invoke(state["question"])
    web_content = "\n".join([d['content'] for d in search_results])
    return {"documents": [web_content], "source": "web"}

def decide_to_generate(state: GraphState):
    """评估检索质量 """
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_base='https://api.deepseek.com', 
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        temperature=0
    )
    
    docs_text = "\n".join(state["documents"])
    grader_prompt = f"问题: {state['question']}\n文档: {docs_text}\n请判断文档是否足以回答问题？仅回复 YES 或 NO。"
    score = llm.invoke(grader_prompt).content.strip().upper()

    if "YES" in score:
        return "generate"
    
    if state.get("retry_count", 0) >= 1:
        return "web_search"
    return "transform_query"

def generate(state: GraphState):
    """生成回答 """
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_base='https://api.deepseek.com', 
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        temperature=0
    )
    source = state.get("source", "local")
    prefix = "【💡 本地知识库回答】\n" if source == "local" else "【⚠️ 联网搜索结果，仅供参考】\n"
    
    system_rules = """你是一个严谨的产品专家。
    行为规则：
    1. 如果上下文（Context）足以回答，请直接回答。
    2. 如果信息缺失或模糊，严禁编造答案，必须向用户主动提问以澄清需求。
    3. 语气要礼貌、专业。"""

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
workflow.add_conditional_edges("retrieve", decide_to_generate, 
    {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()