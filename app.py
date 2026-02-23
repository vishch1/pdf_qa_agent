import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Load Environment Variables
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# ============================================================
# Page Configuration & Custom Styling
# ============================================================
st.set_page_config(page_title="📄 PDF QA Agent", page_icon="🤖", layout="centered")

# --- Elegant Gradient Background and Chat Bubble Styles ---
page_bg = """
<style>
/* === Full Page Pastel Background === */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fffafc, #f3f8ff, #eaf7f3, #fdf6ff);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
    color: #222222;
    font-family: "Poppins", sans-serif;
    min-height: 100vh;
    padding-bottom: 3rem;
}

/* Subtle gradient animation */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #fef6ff, #eaf3ff);
    color: #333333;
    border-right: 1px solid #e0e0e0;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}

/* Header Transparent */
[data-testid="stHeader"] {
    background: rgba(255,255,255,0.0);
}

/* === Chat Bubbles === */
.chat-bubble-user {
    background-color: #d1ebff; /* pastel blue */
    color: #000000;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    margin: 10px 0;
    text-align: right;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.08);
}



/* === Chat Container === */
.main-chat-container {
    background-color: rgba(255, 255, 255, 0.75);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
}

/* === File Uploader Box === */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.8);
    border: 2px dashed #c5c6ca;
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* === Input Box === */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    border: 1px solid #d6d6d6 !important;
    color: #222222 !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* === Buttons === */
.stButton>button {
    background: linear-gradient(90deg, #b5e8f1, #fbd1e0, #f9e3a9);
    color: #333333;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.04);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

/* === Headings and Text === */
h1, h2, h3, h4 {
    color: #222222 !important;
}
p, label, span {
    color: #2b2b2b !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("📚 PDF Question Answering Agent")
st.caption("Ask anything from your uploaded PDF — powered by LangChain & OpenAI ⚡")

# ============================================================
# PDF Upload
# ============================================================
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Processing your PDF..."):
        pdf_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE
        )

        # ============================================================
        # Prompt Setup
        # ============================================================
        BASE_PROMPT = ChatPromptTemplate.from_template("""
        You are a knowledgeable and friendly AI assistant specialized in answering questions from PDFs.
        Use the given context extracted from the PDF to respond to the user's question accurately.

        Rules:
        - Base your answer only on the provided context.
        - If the answer isn't clearly in the context, say "I couldn't find this information in the PDF."
        - Use short paragraphs and bullet points for clarity.
        - Be conversational and concise.

        Context:
        {context}

        User Question:
        {question}

        Answer:
        """)

        SUMMARY_PROMPT = "Summarize the main topics and key insights of the PDF in clear bullet points."
        TOPIC_PROMPT = "List all major topics and subtopics discussed in this PDF, organized hierarchically."
        QUESTION_PROMPT = "Generate 5 practice or exam-style questions based on this PDF content."
        INSIGHT_PROMPT = "Identify the most critical insights or conclusions from this PDF and explain their importance simply."

        def choose_prompt(user_query: str):
            q = user_query.lower()
            if "summarize" in q or "overview" in q:
                return SUMMARY_PROMPT
            elif "topic" in q or "concept" in q:
                return TOPIC_PROMPT
            elif "question" in q or "practice" in q:
                return QUESTION_PROMPT
            elif "insight" in q or "takeaway" in q or "key point" in q:
                return INSIGHT_PROMPT
            else:
                return None

        # ============================================================
        # Build Runnable Chain
        # ============================================================
        pdf_qa_agent = (
            {
                "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough()
            }
            | BASE_PROMPT
            | llm
            | StrOutputParser()
        )

        # ============================================================
        # Chat Interface with History
        # ============================================================
        st.subheader("💬 Ask a Question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            role_class = "chat-bubble-user" if message["role"] == "user" else "chat-bubble-assistant"
            st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

        user_query = st.chat_input("Ask something about your PDF...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)

            chosen_prompt = choose_prompt(user_query)
            full_query = f"{chosen_prompt}\n\nUser Question: {user_query}" if chosen_prompt else user_query

            streamed_answer = ""
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                for chunk in pdf_qa_agent.stream(full_query):
                    streamed_answer += chunk
                    response_placeholder.markdown(streamed_answer + "▌")
                response_placeholder.markdown(streamed_answer)

            st.session_state.messages.append({"role": "assistant", "content": streamed_answer})

else:
    st.info("👆 Please upload a PDF to get started.")
