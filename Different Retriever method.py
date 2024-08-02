# web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load, chunk, and index the content of the HTML page
loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title", "post-content", "post-header")
                       )))
text_documents = loader.load()

# For PDF docs to read PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Making tokens
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Create vector in folder named faiss_index
    retriever = vector_store.as_retriever()
    return retriever

def get_conversational_chain():
    prompt_template = """
    The input docs contain the user's CV. Answer the question according to the information provided on the CV.\n\n
    Context:\n {context}\n
    Question:\n{input}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(model, prompt)

    return chain

def user_input(user_question, retriever):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    retrieval_chain = create_retrieval_chain(retriever, chain)

    response = retrieval_chain.invoke({"context": docs, "input": user_question})

    st.write("Reply: ", response["answer"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if 'retriever' in st.session_state:
        retriever = st.session_state['retriever']
    else:
        retriever = None

    if user_question and retriever:
        user_input(user_question, retriever)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                retriever = get_vector_store(text_chunks)
                st.session_state['retriever'] = retriever
                st.success("Done")

if __name__ == "__main__":
    main()
