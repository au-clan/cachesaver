from diskcache import Cache
from src.rag.components import RAG_pipeline
import src.rag.components.context_builder as cb
import src.rag.components.prompt_generation as pg
import src.rag.components.query_augmentation as qa
import src.rag.components.retrievers as ret

from langchain.embeddings import HuggingFaceEmbeddings
from cachesaver.pipelines import OnlineAPI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os

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

model="gpt-5-nano"
temperature=1.5
max_completion_tokens=10000
top_p=1.0
stop=None
logprobs=None

# Utils function
def get_cachesaver_client():
    """
    Just a random function to intialize CacheSaver client
    """
    cache = Cache(cache_path)

    # Model
    # model = OnlineLLM(provider="openai")
    model = GroqAPILLM(False)

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
        model="gpt-5-nano"
    )

    return client_cachesaver

if __name__ == '__main__':
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "local", "FAISS")

    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    query_aug = qa.PassQueryAugmentation()
    retriever = ret.Faiss_Retriever(vectorstore=vectorstore, k=1)
    context_builder = cb.ConcatContextBuilder()
    prompt_generation = pg.BasePromptGeneration()

    rag_pipeline = RAG_pipeline(
        query_augmentation=query_aug,
        retriever=retriever,
        context_builder=context_builder,
        prompt_generation=prompt_generation
    )

    prompt = "How much does a table cost?"
    print(f'Prompt: {prompt}\n')

    new_prompt = rag_pipeline.execute(prompt)
    print(f'New Prompt: {new_prompt}')

    params = DecodingParameters(
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stop=stop,
        logprobs=logprobs
    )

    cash_cli = get_cachesaver_client()

    # response = await cash_cli.request(
    #     prompt=new_prompt,
    #     params='',
    #     n = 1,
    #     request_id = f"test",
    #     namespace="test_experiment",
    # )
