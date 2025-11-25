from src.rag.components import QueryAugmentationBase
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NormalizeQueryAugmentation(QueryAugmentationBase):

    def __init__(self, lowercase:bool, stop_word:bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lowercase = lowercase
        self.stop_word = stop_word
        nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

    async def augment(self, prompt:str):
        if self.lowercase:
            prompt = prompt.lower()
        if self.stop_word:
            words = word_tokenize(prompt.lower())

            prompt_list = []
            for w in words:
                if w not in self.stop_word:
                    prompt_list.append(w)
            prompt = ' '.join(prompt_list)
        return prompt