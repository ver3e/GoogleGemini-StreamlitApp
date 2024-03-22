import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

gemini_config = {'temperature': 0.8, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048}
page_config = {st.title('🤖🌐 VarietyBot'),
st.caption("Please ensure clarity in your questions for a smooth conversation. If you've uploaded a PDF, just mention 'my pdf' in your questions. Otherwise, ask usual questions for AI-Generated answers ☺")}

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel(model_name="models/gemini-pro",generation_config=gemini_config)
begin_conversation = model.start_chat(history=[])

def extract_and_split_text(uploaded_files):
    extracted_text = ""
    for uploads in uploaded_files:
        try:
            read_pdf = PdfReader(uploads)
            for page in read_pdf.pages:
                extracted_text += page.extract_text()
        except Exception as e:
            st.warning(f"Error reading PDF file: {uploads}, Error: {e}")
    
    split_document = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    text_chunks = split_document.split_text(extracted_text)
    
    return text_chunks

def convert_link_to_text(website_link):
    load = PyPDFLoader(website_link) 
    doc_text = load.load_and_split() 
    return doc_text

def get_embeddings_and_store_pdf(chunk_text):
    if not isinstance(chunk_text, list):
        raise ValueError("Text must be a list of text documents")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        create_embedding = FAISS.from_texts(chunk_text, embedding=embedding_model)
        create_embedding.save_local("embeddings_index")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")

def get_embeddings_and_store_link(link_text):
    if not isinstance(link_text, list):
        raise ValueError("Text must be a list of text documents")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        create_embedding=FAISS.from_documents(link_text, embedding=embedding_model) 
        create_embedding.save_local("embeddings_index")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")

def start_conversation_chain():
    my_prompt = '''
        Answer the following question with the given context
        context:\n{context}?\n
        question:\n{question}\n
    ''' 
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt_template = PromptTemplate(template=my_prompt, input_variables=["context", "question"])
    conversation_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return conversation_chain

def get_generated_user_input(user_question):
    text_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        stored_embeddings = FAISS.load_local("embeddings_index", text_embedding, allow_dangerous_deserialization=True)
        check_pdf_similarity = stored_embeddings.similarity_search(user_question)
        convo_chain = start_conversation_chain()
        response = convo_chain({"input_documents": check_pdf_similarity, "question": user_question}, return_only_outputs=True)
        return response['output_text']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def generate_ai_response(user_question):
    if user_question: 
        if  "my pdf" in user_question.lower():
            pdf_convo=get_generated_user_input(user_question)
            return pdf_convo  
        else:
            ai_response=begin_conversation.send_message(user_question)
            return ai_response.text
        
def clear_chat_convo():
    st.session_state.msg_history=[]

def main():
    try:
        with open('style.css') as f: 
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found")
    
    if "msg_history" not in st.session_state:  
        st.session_state.msg_history = [] 

    for message in st.session_state.msg_history:
        with st.chat_message(message["role"]):
            st.write(message["response"])

    st.sidebar.markdown("<div style='display: flex; justify-content: center;'><h3>Choose One To Proceed</h3></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.sidebar.markdown("<div style='display: flex; justify-content: center;'><h3>Chat PDF File <h3></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload One Or More PDF Files", type="pdf",accept_multiple_files=True)
        if uploaded_file is not None:
            if st.sidebar.button("Process PDF File"):
                with st.spinner("Processing..."):
                    try:
                        texts = extract_and_split_text(uploaded_file) 
                        get_embeddings_and_store_pdf(texts)
                        st.success("Proceed to asking PDF")
                    except Exception as e:
                        st.error(f"Error during PDF processing: {e}")
        else:
            st.sidebar.info("Upload PDF to proceed")

    with st.sidebar:
        st.sidebar.markdown("<div style='display: flex; justify-content: center;'><h3>Chat With PDF Link</h3></div>", unsafe_allow_html=True)
        link_input = st.sidebar.text_input("Enter PDF Document URL")
        if link_input:
            if link_input.strip().lower().endswith(".pdf"): 
                if st.sidebar.button("Load PDF Link"):
                    with st.spinner("Processing.."):
                        try:
                            extract_link = convert_link_to_text(link_input) 
                            get_embeddings_and_store_link(extract_link)
                            st.success("Proceed asking PDF Link")
                        except Exception as e:
                            st.error(f"Error occurred processing link: {e}")
            else:
                st.sidebar.error("Please enter a link pointing to a PDF file") 

    user_question = st.chat_input("Ask VarietyBot...")

    if user_question:
        try:
            with st.chat_message("user"):
                st.write(user_question)
            st.session_state.msg_history.append({"role": "user", "response": user_question})
            pdf_response = generate_ai_response(user_question)
            if pdf_response:
                with st.chat_message("assistant"):
                    st.write(pdf_response)
                    st.session_state.msg_history.append({"role": "assistant", "response": pdf_response})
        except Exception as e:
            st.error(f"Error handling User Question: {e}")
    
    st.sidebar.header(" ")
    st.sidebar.button("Click to Clear Chat History", on_click=clear_chat_convo)
            
if __name__ == "__main__":
    main()