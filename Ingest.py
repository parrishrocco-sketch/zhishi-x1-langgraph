import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_name = "智视 X1 智能摄像机.pdf"
file_path = os.path.join(current_dir, pdf_name)

if not os.path.exists(file_path):
    print(f"错误：找不到文件 {pdf_name}")
else:
    loader = PyPDFLoader(file_path)
    data = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    print("正在初始化本地向量模型...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    db_path = os.path.join(current_dir, "chroma_db")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print("向量库已使用本地模型创建成功！")