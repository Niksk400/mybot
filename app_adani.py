import streamlit as st
from streamlit_chat import message
import PyPDF2
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Azure OpenAI Setup
openai.api_type = "azure"
openai.api_base = "https://botpractice.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "2AOaWSX5pAyavK4nBw7cWzksKu3FTznte3RpTaGwUxy0H1bVjYuzJQQJ99BFAC77bzfXJ3w3AAABACOGCgwx"

# Your Azure OpenAI Deployment Name
DEPLOYMENT_NAME = "botpractice"

st.set_page_config(page_title="📄 PDF Chatbot (Azure)", page_icon="🤖")

st.title("🤖 Azure PDF Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

# Process PDF
if pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF Loaded Successfully!")

    # Split text into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Vector Store
    embeddings = OpenAIEmbeddings(deployment=DEPLOYMENT_NAME)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    chain = load_qa_chain(OpenAI(deployment_name=DEPLOYMENT_NAME), chain_type="stuff")

    # Chat input
    user_input = st.text_input("Ask a question based on the PDF:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        docs = vectorstore.similarity_search(user_input)
        response = chain.run(input_documents=docs, question=user_input)

        st.session_state.chat_history.append({"role": "bot", "content": response})

    # Display chat
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            message(chat["content"], is_user=True)
        else:
            message(chat["content"])




