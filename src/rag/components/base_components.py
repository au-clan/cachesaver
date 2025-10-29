from abc import ABC, abstractmethod


#############################################################################
############################ Abstract Classes ###############################
#############################################################################


class QueryAugmentationBase(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    async def augment(self, prompt:str):
        ...


class RetrieverBase(ABC):

    def __init__(self, *args, **kwargs,):
        pass

    @abstractmethod
    def retrieve(self, query):
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

# class RetrieverPipeline():

#     def __init__(self,
#                 query_augmentation: QueryAugmentationBase,
#                 retriever: RetrieverBase):
#         self.query_augmentation = query_augmentation
#         self.retriever = retriever

#     async def execute(self, prompt: str):
#         query = await self.query_augmentation.augment(prompt)
#         docs = self.retriever.retrieve(query)
#         return query, docs

class RAGPipeline():
    
    def __init__(self, 
                 query_augmentation:QueryAugmentationBase, 
                 retriever_list:list[RetrieverBase], 
                #  retriever_pipeline: list[RetrieverPipeline],
                 context_builder:ContextBuilderBase,
                 prompt_generation:PromptGenerationBase):
        
        self.query_augmentation = query_augmentation
        self.retriever_list = retriever_list
        # self.retriever_pipeline = retriever_pipeline
        self.context_builder = context_builder
        self.prompt_generation = prompt_generation
        self.docs = []

    async def execute(self, prompt: str):
        query = await self.query_augmentation.augment(prompt)
        # query = self.query_augmentation.augment(prompt)
        for retriever in self.retriever_list:
            docs_part = retriever.retrieve(query)
            self.docs.extend(docs_part)
        context = self.context_builder.build(self.docs, query)
        generated_prompt = self.prompt_generation.build(context, prompt)
        return generated_prompt

