from abc import ABC
from langchain_community.vectorstores import FAISS
import langchain_text_splitters 
import langchain.embeddings
import langchain.document_loaders
import os
import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from langchain_core.documents.base import Document
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class IndexingPipeline(ABC):

    def __init__(self, lower:bool, stop_word:bool):
        super().__init__()
        nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.lower = lower
        self.stop_word = stop_word

    def load_docs_from_path(self, path:str) -> list[Document]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, path)
        files = [os.path.join(file_path, file_name) for file_name in os.listdir(file_path)]

        # laod and split files
        all_docs = []
        for f in files:
            l = self.loader(f)
            chunks = l.load()
            all_docs.extend(chunks)

        return all_docs
    
    def index_documents(self, documents: Document):
        pass

    def normalize(self, chunk):
        if self.lower:
            print('lowercase chunk')
            chunk = chunk.lower()
        if self.stop_word:
            words = word_tokenize(chunk.lower())

            chunk_list = []
            for w in words:
                if w not in self.stop_word:
                    chunk_list.append(w)
            chunk = ' '.join(chunk_list)
        return chunk


class DenseKBIndexingPipeline(IndexingPipeline):

    def __init__(self, 
                 lower:bool,
                 stop_word:bool,
                 similarity_metric:str,
                 embeddings: langchain.embeddings,
                 loader: langchain.document_loaders=None, 
                 splitter: langchain_text_splitters=None,
                ):
        super().__init__(lower, stop_word)
        assert similarity_metric in ['cosine_similarity', 'dot_product', 'l2'], 'The chosen similarity_metric is not implementend, please chose an existing similarity metric (cosine_similarity, dot_product, l2)'
        
        self.loader = loader
        self.splitter = splitter
        self.embeddings = embeddings
        self.similarity_metric = similarity_metric


    def index_documents(self, documents:Document, split:bool) -> FAISS:
        # prepare path to files
        # current_dir = os.path.dirname(os.path.abspath(__file__))

        if split:
            documents = self.splitter.split_documents(documents)

        chunk_content = [': '.join([self.normalize(d.metadata['title']), self.normalize(d.page_content)]) for d in documents]

        # embed files DOES IT MAKE SENSE LIKE THIS?
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
        docstore = InMemoryDocstore({str(i): d for i, d in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        # create FAISS vector db
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        return vectorstore
    

    def index_from_path(self, path:str) -> FAISS:
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

        chunk_content = [self.normalize(d.page_content) for d in all_chunks]

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

class SparseKBIndexingPipeline(IndexingPipeline):

    def __init__(self,
                lower:bool,
                stop_word:bool,
                loader: langchain.document_loaders=None, 
                splitter: langchain_text_splitters=None, 
                ):
        super().__init__(lower, stop_word)
        self.loader = loader
        self.splitter = splitter

        self.schema = Schema(
            doc_id=ID(stored=True, unique=True),
            title=TEXT(stored=True),
            content=TEXT(stored=True),
            source=TEXT(stored=True),
        )


    def index_from_path(self, path:str, save_path:str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            ix = index.create_in(save_path, self.schema)
        else:
            ix = index.create_in(save_path, self.schema)
        
        writer = ix.writer()

        # prepare path to files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, path)
        files = [os.path.join(file_path, file_name) for file_name in os.listdir(file_path)]

        # laod and split files
        all_chunks = []
        for f in files:
            l = self.loader(f)
            chunks = l.load_and_split(self.splitter)
            for c in chunks:
                all_chunks.append(self.normalize(c))
            # all_chunks.extend(chunks)

        # chunk_content = [d.page_content for d in all_chunks]

        for i, chunk in enumerate(all_chunks):
            writer.add_document(
                doc_id=str(i),
                # title=
                content=chunk.page_content,
                source=chunk.metadata['source'],
            )
        
        writer.commit()

    
    def index_documents(self, documents:list[Document], save_path:str, split:bool):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            ix = index.create_in(save_path, self.schema)
        else:
            ix = index.create_in(save_path, self.schema)
        
        writer = ix.writer()

        if split:
            documents = self.splitter.split_documents(documents)

        for i, chunk in enumerate(documents):
            writer.add_document(
                doc_id=str(i),
                title=self.normalize(chunk.metadata['title']),
                content=self.normalize(chunk.page_content),
                source=chunk.metadata['source'],
            )
        
        writer.commit()


def get_idxs_from_HotpotQA(
        df:pd.DataFrame, 
        nr_of_examples:int=1, 
        types:list[str]=['bridge', 'comparison'], 
        levels:list[str]=['easy', 'medium', 'hard'],
        random:bool=False) -> list[int]:

    rand_sel_idx = []

    for l in levels:
        for t in types:
            sel_inds = list(df[(df.type == t) & (df.level == l)].index)
            if random:
                rand_sel_idx.extend(random.sample(sel_inds, nr_of_examples))
            else:
                rand_sel_idx.extend(sel_inds[:nr_of_examples])

    return rand_sel_idx


def get_examples_from_HotpotQA(df:pd.DataFrame, idxs:list[int], single_sentence:bool=False) -> list[Document]:

    context_list = df.iloc[idxs]['context'].to_list()[0]

    docs = []

    for i, t in enumerate(context_list['title']):
        if single_sentence:
            for s in context_list['sentences'][i]:
                temp_doc = Document(page_content=s, metadata={
                    'title': t,
                    'source': 'dataset'
                })
                docs.append(temp_doc)
        else:
            temp_doc = Document(
                page_content=''.join(context_list['sentences'][i]),
                metadata={
                    'title':t,
                    'source': 'dataset',
                }
            )
            docs.append(temp_doc)
    return docs
