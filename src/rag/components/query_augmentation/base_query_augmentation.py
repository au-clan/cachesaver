from src.rag.components import QueryAugmentationBase
# from components.base_components import QueryAugmentationBase

class PassQueryAugmentation(QueryAugmentationBase):

    def augment(self, prompt:str):
        return prompt