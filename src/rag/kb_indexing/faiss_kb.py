from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os, json
from datasets import load_dataset
from src.rag.kb_indexing.kb_indexing_pipeline import DenseKBIndexingPipeline, get_idxs_from_HotpotQA, get_examples_from_HotpotQA


if __name__ == '__main__':
    # Define Parameter for indexing pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'local', 'files')

    ###################################
    ########### Parameter: ############
    ###################################
    splitter_chunk_size = 200
    spliter_chunk_overlap = 50
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_metric = 'cosine_similarity'
    # similarity_metric = 'dot_product'
    # similarity_metric = 'l2'
    ##################################
    save_path = f'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local/easy_medium_hotpotQA'


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_chunk_size,
        chunk_overlap=spliter_chunk_overlap,
    )
    # loader = PyPDFLoader
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    df = ds['train'].to_pandas()

    parameter_dict = {
        "splitter_chunk_size": splitter_chunk_size,
        "spliter_chunk_overlap": spliter_chunk_overlap,
        "embedding_model_name": embedding_model_name,
    }
    with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
        json.dump(parameter_dict, f, indent=4)

    idxs = get_idxs_from_HotpotQA(df, nr_of_examples=2, levels=['easy', 'medium'])
    df_sel = df.iloc[idxs]
    df_sel.to_csv(os.path.join(save_path, 'questions_used.csv'))

    docs = get_examples_from_HotpotQA(df=df, idxs=idxs, single_sentence=False)

    vectorstore = DenseKBIndexingPipeline(
        embeddings=embeddings,
        similarity_metric=similarity_metric,
        # loader=loader,
        splitter=splitter,
    ).index_documents(docs, split=True)

    vectorstore.save_local(os.path.join(save_path, similarity_metric))
    print('Vectorstore saved!')

    # print(vectorstore.similarity_search(query='Piece of cake', k=1)[0].page_content)
    # print()
    # # print(vectorstore.similarity_search_with_relevance_scores(query='Piece of cake', k=1)[0])
    # # print()

    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    # print(retriever.invoke('Piece of Cake')[0].page_content)
    # save_path = os.path.join(current_dir, "local", "FAISS")
    # vectorstore.save_local(save_path)