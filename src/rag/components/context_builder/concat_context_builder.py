from src.rag.components import ContextBuilderBase

class ConcatContextBuilder(ContextBuilderBase):
    def build(self, docs, query):
        context = '\n\n'.join([f"data: {d.page_content}; metadata: {d.metadata['source']}" for d in docs])
        return context

