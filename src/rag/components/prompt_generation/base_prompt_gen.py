from src.rag.components import PromptGenerationBase
# from components.base_components import PromptGenerationBase


class BasePromptGeneration(PromptGenerationBase):

    def build(self, context:str, prompt:str):
        new_prompt = f'Context: {context} \n\n Prompt: {prompt}'
        return new_prompt