from src.rag.components import ContextBuilderBase

class RRFFusionContextBuilder(ContextBuilderBase):
    def build(self, docs, query):
        k = 60
        context = '\n\n'.join([f"data: {d.page_content}; metadata: {d.metadata['source']}" for d in docs])
        return context

