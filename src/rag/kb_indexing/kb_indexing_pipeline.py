from abc import ABC
from langchain_community.vectorstores import FAISS
import langchain_text_splitters 
import langchain.embeddings
import langchain.document_loaders
import os
import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore


class DenseKBIndexingPipeline(ABC):

    def __init__(self, 
                 loader: langchain.document_loaders, 
                 splitter: langchain_text_splitters, 
                 embeddings: langchain.embeddings,
                 similarity_metric:str,
                ):
        super().__init__()
        assert similarity_metric in ['cosine_similarity', 'dot_product', 'l2'], 'The chosen similarity_metric is not implementend, please chose an existing similarity metric (cosine_similarity, dot_product, l2)'
        
        self.loader = loader
        self.splitter = splitter
        self.embeddings = embeddings
        self.similarity_metric = similarity_metric


    def index_documents(self, path:str) -> FAISS:
        # prepare path to files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, path)
        files = [os.path.join(file_path, file_name) for file_name in os.listdir(file_path)]

        # laod and split files
        all_chunks = []
        for f in files:
            l = self.loader(f)
            chunks = l.load_and_split(self.splitter)
            all_chunks.extend(chunks)

        chunk_content = [d.page_content for d in all_chunks]

        # embed files
        chunk_embeddings = self.embeddings.embed_documents(chunk_content)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")

        # generate indexes for Vector Store
        dim = chunk_embeddings.shape[1]
        if self.similarity_metric == 'l2':
            index = IndexFlatL2(dim)
        elif self.similarity_metric == 'dot_product':
            index = IndexFlatIP(dim)
        elif self.similarity_metric == 'cosine_similarity':
            index = IndexFlatIP(dim)
            chunk_embeddings /= np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        else:
            raise 

        index.add(chunk_embeddings)

        # generate docstore + index to docstore
        docstore = InMemoryDocstore({str(i): d for i, d in enumerate(all_chunks)})
        index_to_docstore_id = {i: str(i) for i in range(len(all_chunks))}

        # create FAISS vector db
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        return vectorstore
