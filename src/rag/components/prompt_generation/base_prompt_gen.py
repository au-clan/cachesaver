from src.rag.components import PromptGenerationBase

class BasePromptGeneration(PromptGenerationBase):

    def build(self, context:str, prompt:str):
        new_prompt = f"""Answer using ONLY using the provided context. If the answer is not included, say 'I do not know'. Do not use prior knowledge.
            \n\n Context: {context} 
            \n\n Prompt: {prompt}"""
        return new_prompt