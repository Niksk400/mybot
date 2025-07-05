
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import tempfile

# Safe model loading to avoid meta tensor issue
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

st.set_page_config(page_title="Adani PDF Chatbot", page_icon="⚡", layout="centered")

st.markdown("""
<style>
body {
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Adani_Power_logo.svg/512px-Adani_Power_logo.svg.png');
    background-size: contain;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.stApp {
    background: rgba(255, 255, 255, 0.9);
    max-width: 800px;
    margin: auto;
    padding: 20px;
    border-radius: 15px;
}
.chat-bubble {
    background: linear-gradient(135deg, #e1f5fe, #b3e5fc);
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    font-size: 16px;
}
.user-bubble {
    background: #c8e6c9;
}
</style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Adani_Group_Logo.svg/512px-Adani_Group_Logo.svg.png", width=200)
st.title("💬 Adani PDF Chatbot")
st.write("Ask questions from your PDF, powered by Adani Group.")

pdf_file = st.file_uploader("Upload PDF 📄", type=["pdf"])

if pdf_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

        st.success("PDF loaded! Start chatting below:")

        if "history" not in st.session_state:
            st.session_state.history = []

        query = st.text_input("Ask your question:", key="input", placeholder="Type your question here...")

        if query and len(query.strip()) > 5:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
                st.session_state.history.append((query, answer))
        elif query:
            st.warning("Enter a valid question (min 5 characters).")

        st.divider()

        for q, a in reversed(st.session_state.history):
            st.markdown(f"<div class='chat-bubble user-bubble'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble'><b>Bot:</b> {a}</div>", unsafe_allow_html=True)

