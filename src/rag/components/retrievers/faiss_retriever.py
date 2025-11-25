from src.rag.components import RetrieverBase

class Faiss_Retriever(RetrieverBase):

    def __init__(self, retriever_kwargs, vectorstore,*args, **kwargs, ):
        super().__init__(*args, **kwargs)
        assert retriever_kwargs['k'] >= 1, "The number of retrieved documents needs to be biggern then 0"
        self.vectorstore = vectorstore
        self.retriever_kwargs = retriever_kwargs

    def retrieve(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs=self.retriever_kwargs)
        docs = retriever.invoke(query)
        return list(enumerate(docs))