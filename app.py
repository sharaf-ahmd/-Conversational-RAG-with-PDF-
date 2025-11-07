# streamlit_app.py
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

    
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#streamlit 
st.title("Converstaional Rag with PDF uploads")
st.write("upload pdfs and chat with their content")

api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=api_key,model_name='openai/gpt-oss-120b')

session_id=st.text_input("session id", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for upfile in uploaded_files:
        temp_path = f"./temp_{upfile.name}.pdf"
        with open(temp_path, "wb") as f:
            f.write(upfile.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)
  

        splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200) 
        splits=splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. do not answer the question"
            "just reformualte it if needed and otherwise return as it is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        ) 

        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt=(
            "you are an assistant for question-answering task"
            "use the following pieces of retrieved context to answer"
            "the question. if you don't know the answer, say that you"
            "don't know, use three sentence maximum and keep the"
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chathistory"),
                ("human","{input}"),
            ]
        )

        chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,chain)

        def get_session_his(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]    
        
        conversational_rag=RunnableWithMessageHistory(
            rag_chain,get_session_his,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_his(session_id)
            response = conversational_rag.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                    },  # constructs a key "abc123" in `store`.
                    )
            st.write("Assistant:", response['answer'])
        




