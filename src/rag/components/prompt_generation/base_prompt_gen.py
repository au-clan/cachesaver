from src.rag.components import PromptGenerationBase
from langchain.prompts import PromptTemplate

class BasePromptGeneration(PromptGenerationBase):

    def __init__(self, prompt_template:PromptTemplate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_template = prompt_template


    def build(self, context:str, prompt:str):
        # new_prompt = f"""Answer using ONLY using the provided context. If the answer is not included, say 'I do not know'. Do not use prior knowledge.
        #     \n\n Context: {context} 
        #     \n\n Prompt: {prompt}"""
        
        # new_prompt = f"""
        #         You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        #         If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        #         Question: {prompt} 
        #         Context: {context} 
        #         Answer:
        #     """

        prompt_invoked = self.prompt_template.invoke(
            {"context": context, "question": prompt}
        ).to_messages()

        new_prompt = prompt_invoked[0].content
        return new_prompt