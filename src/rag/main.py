from diskcache import Cache
from src.rag.components import RAG_pipeline
import src.rag.components.context_builder as cb
import src.rag.components.prompt_generation as pg
import src.rag.components.query_augmentation as qa
import src.rag.components.retrievers as ret

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from cachesaver.pipelines import OnlineAPI
from langchain_community.vectorstores import FAISS
import os, asyncio

from openai import AsyncOpenAI

from src.typedefs import DecodingParameters
from src.models import API, OnlineLLM, GroqAPILLM
from src.utils import tokens2cost
from sentence_transformers import CrossEncoder
from whoosh import index
from langchain import hub
from langchain_core.prompts import PromptTemplate



# Hyperparameter
batch_size = 1
timeout=2
allow_batch_overflow = 1
correctness = 1
ns_ratio=0
value_cache=True

cache_path="caches/developping"

benchmark="hotpotqa"
method="tot_bfs"
split="mini"

model_name="gpt-5-nano"
temperature=1.0
max_completion_tokens=10_000
top_p=1.0
stop=None
logprobs=None

# Decoding Parameters
params = DecodingParameters(
    temperature=temperature,
    max_completion_tokens=max_completion_tokens,
    top_p=top_p,
    stop=stop,
    logprobs=logprobs
)


# Utils function
def get_cachesaver_client():
    """
    Just a random function to intialize CacheSaver client
    """
    cache = Cache(cache_path)

    # Model
    model = OnlineLLM(provider="openai")

    # Pipeline
    pipeline = OnlineAPI(
        model=model,
        cache=cache,
        batch_size=batch_size,
        timeout=timeout,
        allow_batch_overflow=allow_batch_overflow,
        correctness=bool(correctness)
    )

    # CacheSaver Client
    client_cachesaver = API(
        pipeline=pipeline,
        model=model_name
    )
    return client_cachesaver


async def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    ### Initialize Cash Client
    cash_cli = get_cachesaver_client()

    ### PARAMETERS: QUERY AUGMENTATION
    query_rewriting_text = """You are a helpful assistant that generates multiple search queries based on a single input query.

        Perform query expansion. If there are multiple common ways of phrasing a user question
        or common synonyms for key words in the question, make sure to return multiple versions
        of the query with the different phrasings.

        If there are acronyms or words you are not familiar with, do not try to rephrase them.

        Return 2 different versions of the question.
        
        Question: {question}
        """
    
    query_rewriting_template = PromptTemplate.from_template(query_rewriting_text)
    client_kwargs = {
        'params':params,
        'n':1,
        'request_id':f"sth",
        'namespace':"temp",
    }

    ### PARAMETERS: RETRIEVER
    # similarity_metric = 'cosine_similarity'
    # similarity_metric = 'l2'
    # similarity_metric = 'dot_product'

    # load_path_dense = os.path.join(current_dir, "local", similarity_metric)
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vectorstore = FAISS.load_local(
    #     load_path_dense,
    #     embeddings,
    #     allow_dangerous_deserialization=True
    # )

    load_path_sparse = os.path.join(current_dir, "local", "sparse")
    ix = index.open_dir(load_path_sparse)

    ### PARAMETERS: CONTEXT BUILDER
    cross_enc_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    ### PARAMETERS: PROMPT GENERATION
    prompt_template = hub.pull("rlm/rag-prompt")



    ### QUERY AUGMENTATION
    query_aug = qa.PassQueryAugmentation()
    # query_aug = qa.SynonymExtensionQueryAugmentation(max_nr_synonyms=1)
    query_aug = qa.RewritingQueryAugmentation(client=cash_cli, prompt_template=query_rewriting_template, client_kwargs=client_kwargs)

    ### RETRIEVER
    # retriever = ret.Faiss_Retriever(vectorstore=vectorstore, retriever_kwargs={'k':3})
    retriever = ret.Sparse_Retriever(ix=ix, k=3)
    
    ### CONTEXT BUILDER
    # context_builder = cb.ConcatContextBuilder()
    context_builder = cb.CrossEncderContextBuilder(k=3, cross_enc_model=cross_enc_model)

    ### PROMPT GERNERATION
    prompt_generation = pg.BasePromptGeneration(prompt_template=prompt_template)

    ### PIPELINE
    rag_pipeline = RAG_pipeline(
        query_augmentation=query_aug,
        retriever=retriever,
        context_builder=context_builder,
        prompt_generation=prompt_generation
    )

    prompt = "How much does a piece of cake cost?"
    # print(f'Prompt: {prompt}\n')

    new_prompt = await rag_pipeline.execute(prompt)
    # print(f'New Prompt: {new_prompt}')

    # print(' ================= ')
    # response = await cash_cli.request(
    #         prompt = new_prompt,
    #         params = params,
    #         n = 1,
    #         request_id = f"sth",
    #         namespace="temp",
    #     )
    # total_tokens_used = cash_cli.tokens['default']['total']
    # print('Total Number of tokens:', total_tokens_used)
    # print('Total Cost:', tokens2cost(total_tokens_used, model_name))
    # print(' ================= ')
    
    # print(response)
    
if __name__ == '__main__':
    asyncio.run(main())