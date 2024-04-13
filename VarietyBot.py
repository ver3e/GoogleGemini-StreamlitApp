import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from PIL import Image
import json


st.set_page_config(
    page_title="ChatCUD",
    page_icon="💬",
    )

gemini_config = {'temperature': 0.7, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048}
page_config = {
    st.markdown(
    "<h1 style='text-align: center; color: #b22222; font-family: Arial, sans-serif; background-color: #292f4598;'>chatCUD 💬</h1>",
    unsafe_allow_html=True
    ),
    st.markdown("<h4 style='text-align: center; color: white; font-size: 20px; animation: bounce-and-pulse 60s infinite;'>Your CUD AI Assistant</h4>", unsafe_allow_html=True),
}

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel(model_name="models/gemini-pro",generation_config=gemini_config)

#Extracting and Splitting PDF
def extract_text(list_of_uploaded_files):
    pdf_text=''
    for uploaded_pdfs in list_of_uploaded_files:
        read_pdf=PdfReader(uploaded_pdfs)
        for page in read_pdf.pages:
            pdf_text+=page.extract_text()
    
    return pdf_text
    

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=4000)
    chunks = text_splitter.split_text(text)
    return chunks

#Embedding and storing the pdf Local
def get_embeddings_and_store_pdf(chunk_text):
    if not isinstance(chunk_text, list):
        raise ValueError("Text must be a list of text documents")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        create_embedding = FAISS.from_texts(chunk_text, embedding=embedding_model)
        create_embedding.save_local("embeddings_index")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")

#Generating user response for the pdf
def get_generated_user_input(user_question):
    text_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    stored_embeddings = FAISS.load_local("embeddings_index", text_embedding, allow_dangerous_deserialization=True)
    check_pdf_similarity = stored_embeddings.similarity_search(user_question)

    prompt = f"Answer this query based on the Context: \n{check_pdf_similarity}?\nQuestion: \n{user_question}"

    pdf_response = st.session_state.chat_history.send_message(prompt)
    
    return pdf_response

#Clearing Chat 
def clear_chat_convo():
    st.session_state.chat_history.history=[]

#Changing Role Names/Icons
def role_name(role):    
    if role == "model":  
        return "bot.png"  
    elif role=='user':
        return 'user.png'
    else:
        return None 

#Text Splits
def stream(response):
    for word in response.text.split(" "):
        yield word + " "
        time.sleep(0.04)

#Extracts the user question from pdf prompt in get_generated_user_input() 
def extract_user_question(prompt_response):
    for part in reversed(prompt_response):
        if "Question:" in part.text:
            return part.text.split("Question:")[1].strip()

def main():
    #CSS File opening
    with open('dark.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 


    start_conversation = model.start_chat(history=[])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = start_conversation
    
    for message in st.session_state.chat_history.history: #This is the how it implements the role names and their display
        avatar = role_name(message.role)
        if avatar:
            with st.chat_message(message.role, avatar=avatar):
                if "content" in message.parts[0].text:  #Extracts the user question from pdf prompt in get_generated_user_input() 
                    user_question = extract_user_question(message.parts)
                    if user_question:
                        st.markdown(user_question)
                else:  
                    st.markdown(message.parts[0].text)
    
    with st.sidebar:
        option=st.selectbox("",("ChatCUD","ChatPDF"),index=None,placeholder="Choose Your Assistant...")
        if option=='ChatCUD':
            pre_loaded_pdfs=['student_hand_book.pdf','catalogue.pdf']
            texts = extract_text(pre_loaded_pdfs) 
            chunk=get_chunks(texts)
            get_embeddings_and_store_pdf(chunk)

            
    user_question = st.chat_input("Ask ChatCUD...")

    if user_question is not None and user_question.strip() != "":

        with st.chat_message("user", avatar="user.png"):
            st.write(user_question)

        if option=="ChatCUD":
            responses = get_generated_user_input(user_question)
            with st.chat_message("assistant", avatar="bot.png"):
                st.write_stream(stream(responses))
        else:
            response_text = st.session_state.chat_history.send_message(user_question)
            with st.chat_message("assistant", avatar="bot.png"):
                st.write_stream(stream(response_text))

    st.sidebar.button("Click to Clear Chat History", on_click=clear_chat_convo)

if __name__ == "__main__":
    main()




