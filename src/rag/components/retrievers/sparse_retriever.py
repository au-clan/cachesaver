from src.rag.components import RetrieverBase
from whoosh.qparser import QueryParser, OrGroup
from langchain_core.documents.base import Document

class Sparse_Retriever(RetrieverBase):

    def __init__(self, ix, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ix = ix
        self.k = k

    def retrieve(self, query):
        docs = []
        with self.ix.searcher() as searcher:
            qu = QueryParser("content", self.ix.schema, group=OrGroup.factory(0.9)).parse(query)
            results = searcher.search(qu, limit=self.k)
        
            for hit in results:
                d = Document(
                    page_content=hit['content'], 
                    metadata={k:v for k, v in hit.fields().items() if k not in ('content', 'doc_id')}
                    # metadata={'source': hit['source']}
                )
                docs.append(d)

        return docs
        # retriever = self.vectorstore.as_retriever(search_kwargs=self.kwargs)
        # docs = retriever.invoke(query)
        # return docs