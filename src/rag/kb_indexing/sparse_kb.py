from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from src.rag.kb_indexing.kb_indexing_pipeline import SparseKBIndexingPipeline


if __name__ == '__main__':
    # Define Parameter for indexing pipeline
    save_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local/sparse'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'local', 'files') 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    loader = PyPDFLoader

    vectorstore = SparseKBIndexingPipeline(
        loader=loader,
        splitter=splitter,
    ).index_documents(path, save_path)

