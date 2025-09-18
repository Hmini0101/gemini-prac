import streamlit as st
from rag_module import RAGChatbot


@st.cache_resource
def initialize_chatbot():
    chatbot = RAGChatbot(
        api_key_var="GEMINI_API_KEY",
        model_name="gemini-1.5-flash",
        data_paths=[
            "manual.txt",
            "https://ko.wikipedia.org/wiki/%ED%8C%8C%EC%9D%B4%EC%8D%AC",
        ],
    )
    return chatbot


chatbot = initialize_chatbot()
st.title("문서 기반 챗봇 PAGE")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자 입력처리
if prompt := st.chat_input("질문하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각중..."):
            response = chatbot.generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
