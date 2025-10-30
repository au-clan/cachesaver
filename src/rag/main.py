from diskcache import Cache
from src.rag.components import RAGPipeline
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
from src.rag.eval.eval_helper import eval_loop, get_hotpotQA_questions

from src.typedefs import DecodingParameters
from src.models import API, OnlineLLM, GroqAPILLM
from src.utils import tokens2cost
from sentence_transformers import CrossEncoder
from whoosh import index
from langchain import hub
from langchain_core.prompts import PromptTemplate
from src.rag.components import prompt_templates as template
from src.rag.eval import eval_helper as evh
from src.rag.eval import experiment_helper as exh



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

import yaml
from pathlib import Path

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
    config_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/experiments/config.yml'
    config = yaml.safe_load(Path(config_path).read_text())
    cash_cli = get_cachesaver_client()

    rag_pipeline = exh.rag_pipeline_from_config(config=config, cash_client=cash_cli)
    print(rag_pipeline)
    print(rag_pipeline.query_augmentation)
    print(rag_pipeline.retriever_list)
    print(rag_pipeline.context_builder)
    print(rag_pipeline.prompt_generation)

    return

    current_dir = os.path.dirname(os.path.abspath(__file__))

    ### Initialize Cash Client
    cash_cli = get_cachesaver_client()

    ### PARAMETERS: QUERY AUGMENTATION
    client_kwargs = {
        'params':params,
        'n':1,
        'request_id':f"sth",
        'namespace':"temp",
    }

    ### PARAMETERS: RETRIEVER
    # similarity_metric = 'cosine_similarity'
    # similarity_metric = 'l2'
    similarity_metric = 'dot_product'
    k_dense = 6
    k_sparse = 6

    base_path = os.path.join(current_dir, "local", "test_hotpotQA")

    load_path_dense = os.path.join(base_path, similarity_metric)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        load_path_dense,
        embeddings,
        allow_dangerous_deserialization=True
    )

    load_path_sparse = os.path.join(base_path, "sparse")
    ix = index.open_dir(load_path_sparse)

    ### PARAMETERS: CONTEXT BUILDER
    k_context_builder = 6
    cross_enc_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_enc_model = CrossEncoder(cross_enc_model_name)
    
    ### PARAMETERS: PROMPT GENERATION
    # prompt_template = hub.pull("rlm/rag-prompt")


    ### QUERY AUGMENTATION
    query_aug = qa.PassQueryAugmentation()
    # query_aug = qa.SynonymExtensionQueryAugmentation(max_nr_synonyms=1)
    # query_aug = qa.RewritingQueryAugmentation(
    #     client=cash_cli, 
    #     # prompt_template=template.query_rewriting_template, 
    #     prompt_template=template.hyde_query_template,
    #     # prompt_template=template.query_decompose_template, 
    #     client_kwargs=client_kwargs
    #     )

    ### RETRIEVER
    retriever_1 = ret.Faiss_Retriever(vectorstore=vectorstore, retriever_kwargs={'k':k_dense})
    retriever_2 = ret.Sparse_Retriever(ix=ix, k=k_sparse)
    
    ### CONTEXT BUILDER
    context_builder = cb.ConcatContextBuilder()
    # context_builder = cb.CrossEncderContextBuilder(k=k_context_builder, cross_enc_model=cross_enc_model)

    ### PROMPT GERNERATION
    prompt_generation = pg.BasePromptGeneration(prompt_template=template.prompt_template)

    ### PIPELINE
    rag_pipeline = RAGPipeline(
        query_augmentation=query_aug,
        retriever_list=[retriever_1],
        context_builder=context_builder,
        prompt_generation=prompt_generation
    )

    # question_answer_pairs = get_hotpotQA_questions(os.path.join(base_path, 'questions_used.csv'))

    # metric, total_tokens_used, runtime = await eval_loop(
    #     rag_pipeline=rag_pipeline,
    #     question_answer_pairs=question_answer_pairs,
    #     cash_client=cash_cli,
    #     client_params=params,
    #     verbose=True,
    #     )
    
    # print('Total Cost', tokens2cost(total_tokens_used, model_name))

    # return

    # prompt = "How much does a piece of cake cost?"
    # prompt = '750 7th Avenue and 101 Park Avenue, are located in which city?'
    prompt = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    # prompt = """
    #     Compare and contrast the ethical implications of using large language models (LLMs) in clinical diagnostic decision support systems 
    #     with the privacy concerns associated with federated learning models trained on distributed electronic health records (EHRs), and discuss 
    #     the regulatory frameworks in the EU and the US that currently or might eventually govern both technologies.
    # """
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
    # response = response[0]
    # total_tokens_used = cash_cli.tokens['default']['total']
    # print('Total Number of tokens:', total_tokens_used)
    # print('Total Cost:', tokens2cost(total_tokens_used, model_name))
    # print(' ================= ')
    # # print(type(response), type(response[0]))
    
    # print(response)
    # print(response == 'New York City')
    # print(response == 'Badr Hari')


if __name__ == '__main__':
    asyncio.run(main())