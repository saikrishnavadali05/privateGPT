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

def main():
    st.set_page_config(page_title='DocSpeak', page_icon='dockspeak_.png')
    st.header("DocSpeak")
    st.markdown("#### Chat with PDFs")
    st.warning('Be respectful while asking questions')
    user_question = st.text_input("Ask a Question from the PDF Files")
    st.sidebar.markdown("# DocSpeak")
    api_key = st.sidebar.text_input('Enter Gemini API key and Press Enter', type="password")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files.", accept_multiple_files=True)

    if user_question:
        user_input(user_question, api_key)

    # with st.sidebar:
    if st.sidebar.button("Submit & Process"):
        if api_key:
            with st.spinner("Embedding..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                _ = get_vector_store(text_chunks, api_key)
                st.sidebar.success("Ready to Go!")
        else:
            st.sidebar.error('Please provide API key!')


if __name__ == "__main__":
    main()    
            