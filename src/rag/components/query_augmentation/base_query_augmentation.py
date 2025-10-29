from src.rag.components import QueryAugmentationBase

class PassQueryAugmentation(QueryAugmentationBase):

    async def augment(self, prompt:str):
        return prompt