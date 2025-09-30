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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # similarity_metric = 'cosine_similarity'
    # similarity_metric = 'l2'
    similarity_metric = 'dot_product'
    save_path = os.path.join(current_dir, "local", similarity_metric)

    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # query_aug = qa.PassQueryAugmentation()
    query_aug = qa.SynonymExtensionQueryAugmentation(max_nr_synonyms=1)
    retriever = ret.Faiss_Retriever(vectorstore=vectorstore, kwargs={'k':1})
    context_builder = cb.ConcatContextBuilder()
    prompt_generation = pg.BasePromptGeneration()

    rag_pipeline = RAG_pipeline(
        query_augmentation=query_aug,
        retriever=retriever,
        context_builder=context_builder,
        prompt_generation=prompt_generation
    )

    prompt = "How much does a piece of cake cost?"
    print(f'Prompt: {prompt}\n')

    new_prompt = rag_pipeline.execute(prompt)
    print(f'New Prompt: {new_prompt}')

    cash_cli = get_cachesaver_client()
    print(' ================= ')
    response = await cash_cli.request(
            prompt = new_prompt,
            params = params,
            n = 1,
            request_id = f"sth",
            namespace="temp",
        )
    print(' ================= ')
    print(response)
    
if __name__ == '__main__':
    asyncio.run(main())