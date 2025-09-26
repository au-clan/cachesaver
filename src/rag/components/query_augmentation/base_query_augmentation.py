from src.rag.components import QueryAugmentationBase

class PassQueryAugmentation(QueryAugmentationBase):

    def augment(self, prompt:str):
        return prompt