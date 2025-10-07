from src.models.api import API
from src.rag.components import QueryAugmentationBase
from langchain_core.prompts import PromptTemplate

class RewritingQueryAugmentation(QueryAugmentationBase):

    def __init__(self, client:API, client_kwargs:dict, prompt_template:PromptTemplate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.prompt_template = prompt_template
        self.client_kwargs = client_kwargs

    async def augment(self, prompt:str):
        prompt_invoked = self.prompt_template.invoke({
            "question": prompt
        }).to_messages()
        prompt_invoked = prompt_invoked[0].content
        
        response = await self.client.request(
            prompt=prompt_invoked,
            **self.client_kwargs
        )

        return response[0]