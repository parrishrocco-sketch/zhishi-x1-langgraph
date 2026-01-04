import streamlit as st
import os
from app import app 

st.set_page_config(page_title="智视 X1 全能助手", page_icon="🌐", layout="centered")
st.title("🌐 智视 X1 智能全能助手")
st.caption("优先检索本地库 | 智能自纠错 | 自动联网搜索")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("您可以问我任何问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        final_generation = ""
        
        with st.status("AI 正在处理中...", expanded=True) as status:
            inputs = {"question": prompt, "retry_count": 0, "source": "local"}
            
            for output in app.stream(inputs):
                for node_name, node_output in output.items():
                    if node_name == "retrieve":
                        st.write("**检索节点**：正在翻阅本地说明书...")
                    elif node_name == "transform_query":
                        st.write(f"**纠错节点**：尝试优化搜索词...")
                    elif node_name == "web_search":
                        st.warning("**跳出循环**：本地无果，正在启动全网搜索...")
                    elif node_name == "generate":
                        st.write("**生成节点**：正在组织最终回答...")
                        final_generation = node_output.get("generation", "")
                        if "请问" in final_generation or "能否" in final_generation:
                            st.info("💡 信息不足，助手正在向您询问细节...")
            
            status.update(label="处理完成！", state="complete", expanded=False)

        if final_generation:
            response_placeholder.markdown(final_generation)
            st.session_state.messages.append({"role": "assistant", "content": final_generation})
        else:
            response_placeholder.error("抱歉，未能获取到有效答案。")