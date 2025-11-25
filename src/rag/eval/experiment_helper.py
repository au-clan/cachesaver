from pathlib import Path
from src.rag.components.base_components import RAGPipeline
import src.rag.components.context_builder as cb
import src.rag.components.prompt_generation as pg
import src.rag.components.query_augmentation as qa
import src.rag.components.retrievers as ret
from src.rag.components import prompt_templates as template
from src.typedefs import DecodingParameters
from src.rag.eval.eval_helper import eval_loop, get_cachesaver_client, get_hotpotQA_questions

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from whoosh import index
from langchain_community.vectorstores import FAISS

import os, yaml
from datetime import datetime


def rag_pipeline_from_config(config:dict, cash_client) -> RAGPipeline:
    base_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local'
    base_path_dataset = os.path.join(base_path, config['data'])

    # Define Cashsaver client Params
    cash_client_params = DecodingParameters(
        logprobs=None,
        stop=None,
        **config['client_kwargs']['decoding_params']
        # temperature=temperature,
        # max_completion_tokens=max_completion_tokens,
        # top_p=top_p,
        # stop=stop,
        # logprobs=logprobs
    )

    client_kwargs = {
        'params':cash_client_params,
        'n':config['client_kwargs']['n'],
        'request_id':f"sth",
        'namespace':"temp",
    }
    
    # Define Query Augmentation
    query_aug_component = config['query_augmentation']['component']
    if 'normalize' in query_aug_component:
        base_path_dataset = f'{base_path_dataset}_normalize'

    query_aug_list = []
    for qa_comp in query_aug_component:
        if qa_comp == 'pass':
            query_aug = qa.PassQueryAugmentation()
        elif qa_comp == 'normalize':
            query_aug = qa.NormalizeQueryAugmentation(
                lowercase=config['query_augmentation']['kwargs']['lowercase'],
                stop_word=config['query_augmentation']['kwargs']['stop_word']
            )
        elif qa_comp == 'synonym_extension':
            query_aug = qa.SynonymExtensionQueryAugmentation(max_nr_synonyms=config['query_augmentation']['kwargs']['nr_synonyms'])
        elif qa_comp in ['rewriting', 'hyde', 'decompose', 'multi_query']:
            if qa_comp == 'rewriting':
                query_aug_template = template.query_rewriting_template
            elif qa_comp == 'hyde':
                query_aug_template = template.hyde_query_template
            elif qa_comp == 'decompose':
                query_aug_template = template.query_decompose_template
            elif qa_comp == 'multi_query':
                query_aug_template = template.multi_query_rewriting_template
                
            query_aug = qa.RewritingQueryAugmentation(
                client=cash_client, 
                prompt_template=query_aug_template,
                client_kwargs=client_kwargs
                )
            query_aug_list.append(query_aug)
        else:
            raise AttributeError('The QueryAugmentation component is not defined! Select between: pass, synonym_extension, rewriting, hyde, decompose')


    # Define Retriever
    retriever_list = []
    for i, ret_type in enumerate(config['retriever']['type']):
        if ret_type == 'sparse':
            load_path_sparse = os.path.join(base_path_dataset, "sparse")
            ix = index.open_dir(load_path_sparse)
            retriever = ret.Sparse_Retriever(ix=ix, k=config['retriever']['kwargs']['k'][i])
            retriever_list.append(retriever)
        else:
            load_path_dense = os.path.join(base_path_dataset, ret_type)
            embeddings = HuggingFaceEmbeddings(model_name=config['retriever']['kwargs']['embedding_model'])
            vectorstore = FAISS.load_local(
                load_path_dense,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = ret.Faiss_Retriever(vectorstore=vectorstore, retriever_kwargs={'k':config['retriever']['kwargs']['k'][i]})
            retriever_list.append(retriever)


    # Define Context Builder
    if config['context_builder']['component'] == 'cross_encoder':
        cross_enc_model = CrossEncoder(config['context_builder']['kwargs']['cross_enc_model_name'])
        context_builder = cb.CrossEncderContextBuilder(k=config['context_builder']['kwargs']['k_context_builder'], cross_enc_model=cross_enc_model)
    elif config['context_builder']['component'] == 'concat':
        context_builder = cb.ConcatContextBuilder()
    else:
        raise AttributeError('The CrossEncoder component is not defined! Select between: cross_encoder, concat')

    # Define Prompt Generation Component -> right now fixed
    prompt_generation = pg.BasePromptGeneration(prompt_template=template.prompt_template)

    rag_pipeline = RAGPipeline(
        query_augmentation=query_aug_list,
        retriever_list=retriever_list,
        context_builder=context_builder,
        prompt_generation=prompt_generation
    )

    return rag_pipeline


def save_experiment_results(
        config:dict, 
        result_metrics:dict, 
        generation_dict:dict,
        rag_ret_docs:dict,
        path:str='C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/experiments/local', 
    ):
    exp_timestamp = f"{config['experiment_name']}/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_dir = os.path.join(path, exp_timestamp)
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    with open(f"{save_dir}/results.yaml", "w") as f:
        yaml.safe_dump(result_metrics, f)

    with open(f"{save_dir}/generations.yaml", "w") as f:
        yaml.safe_dump(generation_dict, f)

    with open(f"{save_dir}/rag_ret_docs.yaml", "w") as f:
        yaml.safe_dump(rag_ret_docs, f)


async def experiment_loop(config:dict, verbose:bool=False):
    save_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/experiments/local'

    # Define RAG Pipeline
    # config = yaml.safe_load(Path(config_path).read_text())
    cash_client = get_cachesaver_client(cache_path=os.path.join(save_path, config['experiment_name']), **config['cachesaver_config'])
    rag_pipeline = rag_pipeline_from_config(config, cash_client)

    # load questions
    base_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/local'
    question_answer_pairs = get_hotpotQA_questions(os.path.join(base_path , config['data'], 'questions_used.csv'))

    # define client params
    params = DecodingParameters(
        logprobs=None,
        stop=None,
        **config['client_kwargs']['decoding_params']
    )

    result_dict, generation_dict, rag_ret_docs = await eval_loop(
        rag_pipeline=rag_pipeline,
        question_answer_pairs=question_answer_pairs,
        cash_client=cash_client,
        client_params=params,
        verbose=verbose,
    )

    save_experiment_results(config=config, result_metrics=result_dict, generation_dict=generation_dict, rag_ret_docs=rag_ret_docs)


async def run_multiple_experiments(config_path:str):
    
    multiple_config = yaml.safe_load(Path(config_path).read_text())