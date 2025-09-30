from src.rag.components import QueryAugmentationBase
# https://www.nltk.org/
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class SynonymExtensionQueryAugmentation(QueryAugmentationBase):

    def __init__(self, max_nr_synonyms):
        super().__init__()
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.max_nr_synonyms = max_nr_synonyms

    def get_synonyms(self, word, max_syns, seen_synonyms):
    
        seen_synonyms_local = seen_synonyms.copy()
        synonym_list = []
        for syn in wordnet.synsets(word):
            lemmas = syn.lemmas()
            for l in lemmas:
                synonym = l.name().replace('_', ' ').lower()
                if synonym not in seen_synonyms:
                    seen_synonyms_local.add(synonym)
                    synonym_list.extend([synonym])

        seen_synonyms.update(synonym_list[:max_syns])
        return synonym_list[:max_syns], seen_synonyms

    def augment(self, prompt:str):
        words = word_tokenize(prompt.lower())

        exp_query = []
        seen_synonyms = set(words)
        for w in words:
            exp_query.extend([w])
            if w not in self.stop_words:
                synonyms, seen_synonyms = self.get_synonyms(word=w, max_syns=self.max_nr_synonyms, seen_synonyms=seen_synonyms)
                exp_query.extend(synonyms)
        
        return ' '.join(exp_query)