import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def display_instruction_window():
    with st.expander("💡 How to Use this Chatbot", expanded=False):
        st.markdown("""
            ### Instructions:
            1. Go to https://aistudio.google.com/app/apikey
            2. Click on 'Create API key' and copy the 39 digit key.
            3. Paste it on the box for the API key given in the side panel.
            4. Upload one or more PDFs in the second box given in the side panel.
            5. Ask questions related to the summary of the PDF or about specific topics given in the PDF.
                    
            Enjoy the conversation!
            """)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the
    provided context, just say, "Answer is not available in the context", don't provide a wrong answer.
    If the user says 'thanks' or 'thank you', just say 'You're the most welcome!"\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with any PDF using Gemini 🤖")

    with st.sidebar:
        st.title('Google Gemini Chatbot: Chat-with-PDF GPT')

        api_key = st.text_input('Enter Google API key and press Enter:', type='password')

        if len(api_key) != 39:
            st.warning('Please enter a valid Google API key!', icon='⚠️')
        else:
            os.environ['GOOGLE_API_KEY'] = api_key
            st.success('API key accepted! Proceed to upload PDFs and ask questions.', icon='👉')
            genai.configure(api_key=api_key)

    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    for message in st.session_state["conversation_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a Question from the PDF File(s)")

    if user_question:

        with st.chat_message("user"):
            st.markdown(user_question)

        answer = user_input(user_question)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["conversation_history"].append({"role": "user", "content": user_question})
        st.session_state["conversation_history"].append({"role": "assistant", "content": answer})

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    display_instruction_window()

if __name__ == "__main__":
    main()
