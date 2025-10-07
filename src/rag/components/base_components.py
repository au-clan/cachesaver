from abc import ABC, abstractmethod


#############################################################################
############################ Abstract Classes ###############################
#############################################################################


class QueryAugmentationBase(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def augment(self, prompt:str):
        ...


class RetrieverBase(ABC):

    def __init__(self, *args, **kwargs,):
        pass

    @abstractmethod
    def retrieve(self):
        ...


class ContextBuilderBase(ABC):

    def __init__(self, *args, **kwargs):
        # super().__init__()
        pass

    @abstractmethod
    def build(self, docs, query):
        ...


class PromptGenerationBase(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def build(self, context: str, prompt:str):
        ...


#############################################################################
################################# Pipeline ##################################
#############################################################################


class RAG_pipeline():
    
    def __init__(self, 
                 query_augmentation:QueryAugmentationBase, 
                 retriever:RetrieverBase, 
                 context_builder:ContextBuilderBase,
                 prompt_generation:PromptGenerationBase):
        
        self.query_augmentation = query_augmentation
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_generation = prompt_generation

    async def execute(self, prompt: str):
        query = await self.query_augmentation.augment(prompt)
        docs = self.retriever.retrieve(query)
        context = self.context_builder.build(docs, query)
        generated_prompt = self.prompt_generation.build(context, prompt)
        return generated_prompt

