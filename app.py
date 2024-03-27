import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain.embeddings import HuggingFaceInstructorEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_openai import ChatOpenAI
from htmlTemplate import css, bot_template, user_template
import os
import openai

from keys import OPENAI_API_KEY

from openai import OpenAI

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

#to get all texts fro pdf to single pdf
def get_pdf_text(pdf_docs):
    #all of raw text from PDFs
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    tetx_splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,#1000 characters chunk
    chunk_overlap = 200,#it starts to take 200 words from previous chunk
    length_function = len
    )
    chunks = tetx_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
#complted until to store in vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory =memory 
    )
    return conversation_chain
def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

def main():
    #load_dotenv()
    st.set_page_config("Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Lets chat with PDFs")
    user_question = st.text_input("Ask a about you documents")
    if user_question:
        handle_userinput(user_question)
    

    with st.sidebar:
        st.subheader("Documents")
    
        pdf_docs=st.file_uploader("Please upload PDFs",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get the pdf
                raw_text = get_pdf_text(pdf_docs)

                #get the chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                #cretae vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                #it takes the history of the conversation returns the next element in the conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)
    


if __name__=='__main__':
    main()
