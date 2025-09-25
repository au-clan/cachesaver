from src.rag.components import RetrieverBase
# from components.base_components import RetrieverBase


class Faiss_Retriever(RetrieverBase):

    def retrieve(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs={'k':self.k})
        docs = retriever.get_relevant_documents(query)
        return docs