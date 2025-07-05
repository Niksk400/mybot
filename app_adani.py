import streamlit as st
from streamlit_chat import message
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# ----------------- Azure OpenAI Setup -----------------
AZURE_OPENAI_API_KEY = "2KfAJHSB64E2t2h2x4uGD4T7dEoEbbctuKN2pl7jWRpfyNI4YzueJQQJ99BGACHYHv6XJ3w3AAABACOGY084"  # Replace with your valid key
AZURE_OPENAI_ENDPOINT = "https://foradanitrying.openai.azure.com/"  # Replace with your correct endpoint
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"

DEPLOYMENT_NAME_CHATBOT = "chatbot-gpt35"      # Must match your Azure GPT-3.5 deployment name
DEPLOYMENT_NAME_EMBEDDINGS = "embedding-ada"   # Must match your Azure embedding deployment name

# ----------------- Streamlit UI Setup -----------------
st.set_page_config(page_title="Azure PDF Chatbot")
st.title("Azure PDF Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- File Upload -----------------
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF Loaded Successfully")

    # ----------------- Text Chunking -----------------
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # ----------------- Azure OpenAI Embeddings -----------------
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=DEPLOYMENT_NAME_EMBEDDINGS,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # ----------------- Azure ChatGPT Setup -----------------
    llm = AzureChatOpenAI(
        deployment_name=DEPLOYMENT_NAME_CHATBOT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    # ----------------- Chat Interface -----------------
    user_input = st.text_input("Ask a question based on the PDF:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        docs = vectorstore.similarity_search(user_input)
        response = chain.run(input_documents=docs, question=user_input)

        st.session_state.chat_history.append({"role": "bot", "content": response})

    for chat in st.session_state.chat_history:
        message(chat["content"], is_user=(chat["role"] == "user"))
