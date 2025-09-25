from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os


if __name__ == '__main__':
    pdf_path = '/local/files'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, 'local', 'files')
    pdf_file = [os.path.join(pdf_path, file_name) for file_name in os.listdir(pdf_path)]
    print('Number of PDF files: ',len(pdf_file))
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    all_chunks = []

    # Split Documents
    for f in pdf_file:
        loader = PyPDFLoader(f)
        chunks = loader.load_and_split(splitter)
        all_chunks.extend(chunks)

    # Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    save_path = os.path.join(current_dir, "local", "FAISS")
    vectorstore.save_local(save_path)    