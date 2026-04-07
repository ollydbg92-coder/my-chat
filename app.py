import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="PDF 지능형 분석기", page_icon="📄")
st.title("📂 PDF 리포트 분석 챗봇")
st.write("PDF 파일을 업로드하고 궁금한 점을 질문하세요.")

# --- 2. 사이드바 (설정 및 파일 업로드) ---
with st.sidebar:
    st.header("1️⃣ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("2️⃣ 파일 업로드")
    uploaded_file = st.file_uploader("분석할 PDF 파일을 선택하세요", type="pdf")
    process_button = st.button("파일 분석 시작")

# --- 3. PDF 처리 및 RAG 로직 ---
if process_button:
    if not api_key:
        st.error("API Key를 입력해주세요!")
    elif not uploaded_file:
        st.error("PDF 파일을 업로드해주세요!")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        
        with st.spinner("PDF 내용을 분석 중입니다..."):
            # PDF 임시 저장 (Loader가 경로를 필요로 함)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # 1. 문서 로드
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # 2. 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            
            # 3. 임베딩 및 벡터 DB
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            
            # 4. 모델 및 체인 설정
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
            system_prompt = "문맥(context)을 사용하여 답변하세요. 모르면 모른다고 하세요.\n\n{context}"
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
            
            os.remove(tmp_path) # 임시 파일 삭제
        st.success("분석 완료! 이제 질문하세요.")

# --- 4. 채팅 인터페이스 ---
if "rag_chain" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 새 질문 입력
    if user_input := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})