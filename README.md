# 🧠 Conversational RAG with PDF Uploads

This Streamlit application allows users to **upload PDF documents and chat interactively with their content** using **Retrieval-Augmented Generation (RAG)**.  
It combines **LangChain**, **Hugging Face embeddings**, **Chroma vector store**, and **Groq's LLM** to deliver accurate, context-aware responses from your documents.

---

## 🚀 Features

- 📄 **Upload and process multiple PDFs**
- 💬 **Chat conversationally** with your document content
- 🧠 **Context-aware retrieval** using `create_history_aware_retriever`
- 🧩 **Conversation memory** powered by `RunnableWithMessageHistory`
- 🔍 **Semantic document search** using HuggingFace embeddings + Chroma
- ⚡ **High-speed inference** using Groq's `openai/gpt-oss-120b` model

---

## 🧰 Tech Stack

| Component | Library / Service |
|------------|------------------|
| UI | Streamlit |
| LLM | [Groq API](https://groq.com) |
| Framework | LangChain |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector Store | Chroma |
| Document Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Environment | Python 3.10+ |

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory and add:

```bash
HF_TOKEN=your_huggingface_api_token
GROQ_API_KEY=your_groq_api_key
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/conversational-rag-pdf.git
cd conversational-rag-pdf
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

(If you don't have a `requirements.txt` yet, generate one using:)

```bash
pip freeze > requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🧩 How It Works

1. **PDF Upload:**  
   Users upload one or more PDF documents through the Streamlit interface.

2. **Document Processing:**  
   The app extracts text using `PyPDFLoader`, splits it into chunks using `RecursiveCharacterTextSplitter`, and embeds them with `HuggingFaceEmbeddings`.

3. **Vector Storage:**  
   The embeddings are stored and retrieved using Chroma.

4. **Conversational Retrieval:**  
   LangChain's `create_history_aware_retriever` ensures that user queries are contextualized with prior conversation history.

5. **LLM Response Generation:**  
   Groq's model (`openai/gpt-oss-120b`) generates precise answers based on retrieved context.

---

## 🗂️ Project Structure

```
.
├── streamlit_app.py        # Main application
├── .env                    # Environment variables
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── temp_*.pdf              # Temporary uploaded files
```

---

## 💡 Example Usage

1. Run the app with:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload one or more PDF files.

3. Enter a session ID (e.g., `default_session`) to preserve chat history.

4. Ask questions like:
   - "Summarize the second chapter"
   - "What does the author say about AI ethics?"
   - "Who are the key contributors in this report?"

---

## 🧾 Notes

- This app uses in-memory session storage. If you restart the app, chat history resets.
- Ensure your Hugging Face and Groq API tokens are valid before running.
- Large PDFs may take a few seconds to process.

---

