from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from src.rag.kb_indexing.kb_indexing_pipeline import DenseKBIndexingPipeline


if __name__ == '__main__':
    # Define Parameter for indexing pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'local', 'files') 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    loader = PyPDFLoader
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    similarity_metric = 'cosine_similarity'
    # similarity_metric = 'dot_product'
    # similarity_metric = 'l2'

    vectorstore = DenseKBIndexingPipeline(
        loader=loader,
        splitter=splitter,
        embeddings=embeddings,
        similarity_metric=similarity_metric
    ).index_documents(path)

    save_path = f'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local/{similarity_metric}'
    vectorstore.save_local(save_path)
    # print('Vectorstore saved!')

    # print(vectorstore.similarity_search(query='Piece of cake', k=1)[0].page_content)
    # print()
    # # print(vectorstore.similarity_search_with_relevance_scores(query='Piece of cake', k=1)[0])
    # # print()

    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    # print(retriever.invoke('Piece of Cake')[0].page_content)
    # save_path = os.path.join(current_dir, "local", "FAISS")
    # vectorstore.save_local(save_path)