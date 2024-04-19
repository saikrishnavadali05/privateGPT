import getpass
import re 
import warnings
import json
import os
import streamlit as st
import textwrap
# import streamlit_chat as message
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from utils import *

# Set configuration
load_dotenv()
warnings.filterwarnings("ignore")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

DB_FAISS_PATH = 'vectorstores/faiss'

os.makedirs(DB_FAISS_PATH, exist_ok=True)


def main():
    st.set_page_config(page_title='DocSpeak', page_icon='dockspeak_.png')
    st.header("DocSpeak")
    st.markdown("#### Chat with PDFs")
    st.warning('Be respectful while asking questions')
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.sidebar.markdown("# DocSpeak")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                _ = get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()    
            