import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun # 1. 新增網路搜尋工具

# 你的 Gemini API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="AI 履歷助理", page_icon="📄")
st.title("李品言的專屬 AI 履歷助理")

@st.cache_resource
def init_rag_system():
    # 1. 讀取履歷 PDF
    loader = PyPDFLoader("李品言.pdf")
    docs = loader.load()
    
    # 2. 文字切塊 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # 3. 建立向量資料庫
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. 設定 Gemini 語言模型與 Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    system_prompt = (
        "你是李品言的 AI 履歷代理人。請嚴格根據以下提供的參考資料回答面試相關問題。"
        "【重要指令】：若參考資料未提及該資訊，請一律只回答「履歷未記載此資訊」，不要回答其他任何字。\n\n"
        "【履歷資料】：\n{context}"
    ) # 2. 更新系統提示詞，強制觸發搜尋條件
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. 組裝檢索鏈
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain), llm # 3. 回傳 llm 供後續搜尋總結使用

# 初始化 RAG 系統與搜尋工具
rag_chain, llm = init_rag_system()
search_tool = DuckDuckGoSearchRun()

# 6. 前端對話輸入框
query = st.text_input("輸入你想詢問的問題：", placeholder="例如：中興大學電機系的評價？")

if query:
    with st.spinner("檢索履歷並整理回答中..."):
        # 先進行 RAG 履歷檢索
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]
        
        # 4. 判斷是否需要觸發網路搜尋
        if "未記載此資訊" in answer:
            st.warning("📄 履歷中無相關紀錄，系統已自動啟動網路搜尋...")
            with st.spinner("🌐 正在網路上為您尋找最新資訊..."):
                try:
                    # 執行網路搜尋
                    search_result = search_tool.invoke(query)
                    
                    # 將搜尋結果交給 LLM 進行總結
                    summary_prompt = f"""
                    請根據以下網路搜尋結果，簡短回答使用者的問題。

                    回答規則：
                    1. 只回答使用者問的內容。
                    2. 最多 2～3 句話。
                    3. 不要列出太多背景資料。
                    4. 不要自行延伸其他相關資訊。
                    5. 如果搜尋結果不足以回答，請直接說明。

                    使用者問題：
                    {query}

                    網路搜尋結果：
                    {search_result}
                    """
                    response = llm.invoke(summary_prompt)

                    # 只取模型真正的文字內容
                    if isinstance(response.content, str):
                        final_answer = response.content
                    else:
                        final_answer = ""

                        for item in response.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                final_answer += item.get("text", "")

                    st.write(final_answer)
                except Exception as e:
                    st.error("網路搜尋目前無法使用，請稍後再試。")
        else:
            st.write("### 📄 回答：")
            st.write(answer)