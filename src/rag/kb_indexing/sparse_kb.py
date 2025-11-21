from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from datasets import load_dataset
from src.rag.kb_indexing.kb_indexing_pipeline import SparseKBIndexingPipeline, get_examples_from_HotpotQA, get_idxs_from_HotpotQA


if __name__ == '__main__':
    # Define Parameter for indexing pipeline
    save_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local/easy_medium_hotpotQA/sparse'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(current_dir, 'local', 'files') 

    ###################################
    ########### Parameter: ############
    ###################################
    splitter_chunk_size = 100
    spliter_chunk_overlap = 10
    ###################################

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_chunk_size,
        chunk_overlap=spliter_chunk_overlap,
    )
    # loader = PyPDFLoader

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    df = ds['train'].to_pandas()

    # idxs = get_idxs_from_HotpotQA(df)
    idxs = get_idxs_from_HotpotQA(df, nr_of_examples=10, levels=['easy'])
    docs = get_examples_from_HotpotQA(df, idxs)

    vectorstore = SparseKBIndexingPipeline(
        # loader=loader,
        splitter=splitter,
    ).index_documents(documents=docs, save_path=save_path, split=True)
    print('Sparse KB saved!')

