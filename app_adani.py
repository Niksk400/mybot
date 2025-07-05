import streamlit as st
from streamlit_chat import message
import PyPDF2
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Azure OpenAI Setup
openai.api_type = "azure"
openai.api_base = "https://foradanitrying.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "F7q7I1inhLYOnzmLhk04tSIufeUlzZHB7spO7s4oWTREILlxrDqNJQQJ99BGACHYHv6XJ3w3AAABACOGyKuF"

# Deployment Names
DEPLOYMENT_NAME_CHATBOT = "chatbot-gpt35"
DEPLOYMENT_NAME_EMBEDDINGS = "embedding-ada"

st.set_page_config(page_title="📄 Azure PDF Chatbot", page_icon="🤖")

st.title("🤖 Azure PDF Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("✅ PDF Loaded Successfully")

    # Split text into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Vector Store with proper Azure Embedding settings
    embeddings = OpenAIEmbeddings(
        deployment=DEPLOYMENT_NAME_EMBEDDINGS,
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type="azure",
        openai_api_version=openai.api_version,
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Chatbot LLM with Azure GPT-3.5
    chain = load_qa_chain(
        OpenAI(
            deployment_name=DEPLOYMENT_NAME_CHATBOT,
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type="azure",
            openai_api_version=openai.api_version,
        ),
        chain_type="stuff",
    )

    user_input = st.text_input("Ask a question based on the PDF:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        docs = vectorstore.similarity_search(user_input)
        response = chain.run(input_documents=docs, question=user_input)

        st.session_state.chat_history.append({"role": "bot", "content": response})

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            message(chat["content"], is_user=True)
        else:
            message(chat["content"])





