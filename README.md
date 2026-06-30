# 📄 PDF Question Answering Agent

An AI-powered PDF Question Answering application built with **Streamlit**, **LangChain**, **OpenAI GPT-4o Mini**, and **FAISS**. Upload any PDF and interact with it through natural language. The application retrieves relevant content from the document and generates accurate answers using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📂 Upload any PDF document
- 💬 Ask questions in natural language
- 📖 Automatic PDF text extraction
- ✂️ Intelligent text chunking
- 🔍 Semantic search using FAISS Vector Database
- 🤖 GPT-4o Mini powered responses
- 📝 Automatic PDF summarization
- 📚 Topic extraction
- ❓ Practice question generation
- 💡 Key insights extraction
- ⚡ Streaming responses for better user experience
- 💾 Chat history using Streamlit Session State
- 🎨 Modern pastel UI (app.py version)

---

## 🏗️ Project Architecture

```
                Upload PDF
                     │
                     ▼
             PyPDFLoader
                     │
                     ▼
      RecursiveCharacterTextSplitter
                     │
                     ▼
        OpenAI Embedding Model
                     │
                     ▼
          FAISS Vector Store
                     │
                     ▼
             Retriever (Top-K)
                     │
                     ▼
          Prompt + User Query
                     │
                     ▼
         GPT-4o Mini (LLM)
                     │
                     ▼
            Generated Answer
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI GPT-4o Mini
- OpenAI Embeddings
- FAISS
- PyPDFLoader
- python-dotenv

---

## 📁 Project Structure

```
PDF-QA-Agent/
│
├── app.py                 # Enhanced UI version
├── main.py                # Basic Streamlit version
├── requirements.txt
├── .env
├── temp/                  # Uploaded PDFs
├── README.md
└── assets/                # (Optional screenshots)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-qa-agent.git

cd pdf-qa-agent
```

### 2. Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / Mac

```bash
python -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1
```

---

## ▶️ Run the Application

For the enhanced UI:

```bash
streamlit run app.py
```

For the basic version:

```bash
streamlit run main.py
```

---

## 💬 Example Questions

You can ask questions like:

- Summarize this PDF.
- Give me an overview.
- What are the main topics?
- Generate 5 practice questions.
- What are the key insights?
- Explain Chapter 2.
- What is the conclusion?
- List important points.
- Define a particular concept.

---

## 🧠 How It Works

1. User uploads a PDF.
2. PyPDFLoader extracts text from the document.
3. Text is divided into manageable chunks.
4. OpenAI Embeddings convert chunks into vector representations.
5. FAISS stores vectors for semantic search.
6. The retriever fetches the most relevant chunks.
7. GPT-4o Mini generates answers using only the retrieved context.
8. Responses are streamed back to the user.

---

## 📦 LangChain Components Used

- PyPDFLoader
- RecursiveCharacterTextSplitter
- OpenAIEmbeddings
- FAISS Vector Store
- Retriever
- ChatPromptTemplate
- RunnablePassthrough
- StrOutputParser
- ChatOpenAI

---

## 🎯 Prompt Engineering

The application automatically detects the user's intent and selects specialized prompts for:

- PDF Summarization
- Topic Extraction
- Practice Question Generation
- Key Insights
- General Question Answering

---


## 🔮 Future Improvements

- Multi-PDF support
- Citation with page numbers
- Conversation memory
- PDF highlighting
- Export chat as PDF
- Voice input
- OCR support for scanned PDFs
- Multiple LLM support (OpenAI, Gemini, Claude)
- Deployment on Streamlit Cloud

---

## 📌 Requirements

```
streamlit
langchain
langchain-community
langchain-openai
langchain-text-splitters
faiss-cpu
python-dotenv
pypdf
openai
```

---

## 👨‍💻 Author

**Vishakha**

GitHub: https://github.com/vishch1

LinkedIn: https://www.linkedin.com/in/vishakha-chaudhary4/

---

## ⭐ If you found this project useful

Give this repository a ⭐ on GitHub and feel free to contribute!
