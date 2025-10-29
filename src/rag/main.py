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
    client_kwargs = {
        'params':params,
        'n':1,
        'request_id':f"sth",
        'namespace':"temp",
    }

    query_rewriting_text = """You are a helpful assistant that generates multiple search queries based on a single input query.

        Perform query expansion. If there are multiple common ways of phrasing a user question
        or common synonyms for key words in the question, make sure to return multiple versions
        of the query with the different phrasings.

        If there are acronyms or words you are not familiar with, do not try to rephrase them.

        Return 2 different versions of the question.
        
        Question: {question}
        """
    query_rewriting_template = PromptTemplate.from_template(query_rewriting_text)

    hyde_query_template = PromptTemplate(
        input_variables=["question"],
        template="""Given this question: '{question}'

        Please write a detailed, informative document that directly answers this question. 
        The document should be comprehensive and approximately 500 characters long.
        Write as if you're explaining this topic in a textbook or educational material.

        Document:"""
    )

    query_decompose_prompt = """
        You are a helpful assistant that prepares queries that will be sent to a search component.
        Sometimes, these queries are very complex.
        Your job is to simplify complex queries into multiple queries that can be answered
        in isolation to eachother.

        If the query is simple, then keep it as it is.
        Examples
        1. Query: Did Microsoft or Google make more money last year?
        Decomposed Questions: [Question(question='How much profit did Microsoft make last year?', answer=None), Question(question='How much profit did Google make last year?', answer=None)]
        2. Query: What is the capital of France?
        Decomposed Questions: [Question(question='What is the capital of France?', answer=None)]
        3. Query: {question}
        Decomposed Questions:
    """
    query_decompose_template = PromptTemplate.from_template(query_decompose_prompt)

    
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
    prompt_template = hub.pull("rlm/rag-prompt")


    ### QUERY AUGMENTATION
    query_aug = qa.PassQueryAugmentation()
    # query_aug = qa.SynonymExtensionQueryAugmentation(max_nr_synonyms=1)
    # query_aug = qa.RewritingQueryAugmentation(
    #     client=cash_cli, 
    #     # prompt_template=query_rewriting_template, 
    #     prompt_template=hyde_query_template,
    #     # prompt_template=query_decompose_template, 
    #     client_kwargs=client_kwargs
    #     )

    ### RETRIEVER
    retriever_1 = ret.Faiss_Retriever(vectorstore=vectorstore, retriever_kwargs={'k':k_dense})
    retriever_2 = ret.Sparse_Retriever(ix=ix, k=k_sparse)
    
    ### CONTEXT BUILDER
    context_builder = cb.ConcatContextBuilder()
    # context_builder = cb.CrossEncderContextBuilder(k=k_context_builder, cross_enc_model=cross_enc_model)

    ### PROMPT GERNERATION
    prompt_generation = pg.BasePromptGeneration(prompt_template=prompt_template)

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