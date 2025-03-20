import streamlit as st
import os
import logging
import re
from dotenv import load_dotenv
import openai

# Import langchain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma  # Using Chroma instead of Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Loading environment variables...")

# Check OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()
else:
    logger.info("OPENAI_API_KEY found")

def is_valid_url(url):
    """Check if the URL is valid."""
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?))|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return bool(re.match(url_pattern, url))

def create_vectorstore(website_data, chunk_size, chunk_overlap, embedding_model):
    """Create a vector store from website data using Chroma."""
    logger.info(f"Creating vector store with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(website_data)
    logger.info(f"Split into {len(splits)} chunks")
    
    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
    
    # Create Chroma vectorstore (works locally, no external service needed)
    import uuid
    collection_name = f"website_data_{uuid.uuid4().hex[:8]}"  # Create unique collection name
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    logger.info(f"Vector store created successfully with collection: {collection_name}")
    return vectorstore

def main():
    st.set_page_config(page_title="Website Q&A Assistant", layout="wide")
    
    st.title("📚 Website Q&A Assistant")
    st.write("Upload a website URL and ask questions about its content!")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore_dict" not in st.session_state:
        st.session_state.vectorstore_dict = {}
    
    # Sidebar for configurations
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM settings
        st.subheader("Language Model")
        llm_model = st.selectbox(
            "Select LLM Model",
            options=["gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Embedding settings
        st.subheader("Embeddings")
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-small", "text-embedding-3-large"],
            index=0
        )
        
        # Chunking settings
        st.subheader("Text Chunking")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        # Vector database info
        st.subheader("Vector Database")
        st.write("Using Chroma (local vector database)")
        
        # Clear cache button
        if st.button("Clear Cache"):
            st.session_state.vectorstore_dict = {}
            st.session_state.chat_history = []
            st.success("Cache cleared!")
    
    # URL input in the main area
    url = st.text_input("🔗 Enter Website URL:", placeholder="https://example.com")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        process_button = st.button("Process Website")
    with col2:
        clear_chat = st.button("Clear Chat")
        if clear_chat:
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # URL validation
    if url and process_button:
        if not is_valid_url(url):
            st.error("Please enter a valid URL (e.g., https://example.com)")
        else:
            try:
                with st.spinner("📥 Loading website content..."):
                    website_data = WebBaseLoader(url).load()
                
                with st.spinner("🔍 Processing text and creating embeddings..."):
                    vectorstore = create_vectorstore(
                        website_data, chunk_size, chunk_overlap, embedding_model
                    )
                    st.session_state.vectorstore_dict[url] = vectorstore
                    st.success(f"✅ Website processed successfully!")
            
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                logger.exception("Unexpected error processing website")
    
    # Display chat interface if URL has been processed
    if url and url in st.session_state.vectorstore_dict:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for question, answer in st.session_state.chat_history:
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
        
        # User question input
        user_query = st.text_input("❓ Ask a question about the website:", key="user_query")
        
        if user_query:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Create QA chain
                    llm = ChatOpenAI(model=llm_model, temperature=temperature, openai_api_key=openai_api_key)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    
                    for q, a in st.session_state.chat_history:
                        memory.chat_memory.add_user_message(q)
                        memory.chat_memory.add_ai_message(a)
                    
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectorstore_dict[url].as_retriever(search_kwargs={"k": 4}),
                        memory=memory,
                        verbose=True
                    )
                    
                    # Get answer
                    response = qa_chain({"question": user_query})
                    answer = response["answer"]
                    
                    # Store in chat history and refresh
                    st.session_state.chat_history.append((user_query, answer))
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    logger.exception("Error in QA chain")

if __name__ == "__main__":
    main()