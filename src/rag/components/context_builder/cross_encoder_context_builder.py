from src.rag.components import ContextBuilderBase

class CrossEncderContextBuilder(ContextBuilderBase):

    def __init__(self, k:int, cross_enc_model,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.k = k
        self.cross_enc_model = cross_enc_model
    
    def build(self, docs, query):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.cross_enc_model.predict(pairs)
        reranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        
        top_k_docs = reranked_docs[:self.k]
        context = '\n\n'.join([f"data: {d.page_content}; metadata: {d.metadata['source']}" for d in top_k_docs])

        return context
