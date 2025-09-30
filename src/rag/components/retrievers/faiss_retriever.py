from src.rag.components import RetrieverBase

class Faiss_Retriever(RetrieverBase):

    def retrieve(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs=self.kwargs)
        docs = retriever.invoke(query)
        return docs