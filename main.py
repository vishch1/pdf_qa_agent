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


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


st.set_page_config(
    page_title="📄 PDF QA Agent",
    page_icon="🤖",
    layout="centered",
)
st.title("📚 PDF Question Answering Agent")
st.caption("Ask anything from your PDF in real-time! ⚡")


#  PDF Upload

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

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

        # Prompts for Different Intents
        
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

        
        #  Function to Pick Prompt Dynamically
        
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

        # Build Runnable Chain
        
        pdf_qa_agent = (
            {
                "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough()
            }
            | BASE_PROMPT
            | llm
            | StrOutputParser()
        )

        
        st.subheader("💬 Ask a Question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask something about your PDF...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.write(user_query)

            chosen_prompt = choose_prompt(user_query)
            full_query = f"{chosen_prompt}\n\nUser Question: {user_query}" if chosen_prompt else user_query

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                streamed_answer = ""
                for chunk in pdf_qa_agent.stream(full_query):
                    streamed_answer += chunk
                    response_placeholder.markdown(streamed_answer + "▌")
                response_placeholder.markdown(streamed_answer)

            st.session_state.messages.append({"role": "assistant", "content": streamed_answer})
else:
    st.info("👆 Please upload a PDF to get started.")
