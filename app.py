import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

## Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit UI Title
st.title("Checking Q&A")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
You are a medical expert in vision science. Based on the provided context, give a detailed, research-backed answer.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Function to create document embeddings only once
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Load documents from the directory
        st.session_state.docs = st.session_state.loader.load()  # Load the documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Create vector store

        st.write("Document embeddings have been created and stored!")

# Button to precompute embeddings
if st.button("Initialize Document Embeddings"):
    vector_embedding()

# Realtime conversation after embeddings are created
prompt1 = st.text_input("Ask a question about the documents")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please initialize document embeddings first!")
    else:
        # Create the document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start}s")
        st.write(response['answer'])

        # Show relevant document chunks
        with st.expander("Relevant Document Chunks"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")


 
 