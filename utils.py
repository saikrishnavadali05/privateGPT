import getpass
import re 
import warnings
import json
import os
import streamlit as st
import textwrap
import streamlit_chat as message
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Set configuration
load_dotenv()
warnings.filterwarnings("ignore")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

DB_FAISS_PATH = 'vectorstores/faiss'

os.makedirs(DB_FAISS_PATH, exist_ok=True)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(db):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer
    also try to make answer more clean and well trimmed.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                               temperatur=0.3,
                               google_api_key=gemini_api_key)
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    prompt = PromptTemplate(template=prompt_template, input_variables = ["context", "question"])
    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQA.from_chain_type(llm=model, 
                                        chain_type="stuff", 
                                        retriever=db.as_retriever(),
                                        chain_type_kwargs={"prompt":prompt})

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(new_db)
    
    response = chain.invoke(user_question)
    st.write()
    st.write()
    answer = wrap_text_preserve_newlines(response['result'])
    st.write(answer)