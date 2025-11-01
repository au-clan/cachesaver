import re, string, time
from collections import Counter
from datasets import load_dataset
from diskcache import Cache
import pandas as pd
from src.models.api import API
from cachesaver.pipelines import OnlineAPI
from src.models.online import OnlineLLM
from src.rag.components.base_components import RAGPipeline
from src.typedefs import DecodingParameters
from src.utils import tokens2cost
from ..kb_indexing.kb_indexing_pipeline import get_idxs_from_HotpotQA
import numpy as np


def update_dict(dict_a, dict_b):
    for key, value_b in dict_b.items():
        if key in dict_a:
            dict_a[key] += value_b
        else:
            dict_a[key] = value_b
    return dict_a


# HotpotQA evaluation helper functions (https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# Own Code
# Defined default parameters for Cachesaver Client
def get_cachesaver_client(
        batch_size:int=1,
        timeout:int=2,
        allow_batch_overflow:int=1,
        correctness:int=1, 
        cache_path:str="caches/developping",
        model_name:str="gpt-5-nano"
    ):
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

def transform_supporting_facts(context:dict, supporting_facts:dict):
    sup_facts = []
    for title, sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
        cont_idx = np.where(context['title'] == title)
        context_sentence = context['sentences'][cont_idx][sent_id]
        sup_facts.append({'title': title, 'page_content': context_sentence})
    return sup_facts

def test_retrieved_docs(retrieved_docs:list, supporting_facts_list:list):
    retrieved_docs_titles = [d.metadata['title'] for d in retrieved_docs]
    retrieved_docs_page_cont = [d.page_content for d in retrieved_docs]

    binary_doc_incl = 0.0
    N = len(supporting_facts_list)
    for i in range(N):
        sup_fact = supporting_facts_list[i]
        if sup_fact['title'] in retrieved_docs_titles and sup_fact['page_content'] in retrieved_docs_page_cont:
            binary_doc_incl += 1
    binary_doc_incl /= N
    return binary_doc_incl

def update_answer(metrics, prediction, answer, level):
    em = exact_match_score(prediction, answer)
    f1, prec, recall = f1_score(prediction, answer)
    metrics['em'] += float(em)
    metrics[f'em_{level}'] += float(em)
    metrics['f1'] += f1
    metrics[f'f1_{level}'] += f1
    metrics['prec'] += prec
    metrics[f'prec_{level}'] += prec
    metrics['recall'] += recall
    metrics[f'recall_{level}'] += recall

    return em, prec, recall

def get_hotpotQA_questions(load_path:str) -> list[tuple]:
    # ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    # df = ds['train'].to_pandas()

    # idxs = get_idxs_from_HotpotQA(df)

    question_df = pd.read_csv(load_path)
    
    # sel_rows = df.iloc[idxs]
    questions = question_df['question'].to_numpy()
    answers = question_df['answer'].to_numpy()
    level = question_df['level'].to_numpy()
    # question_type = question_df['type'].to_numpy()

    return list(zip(questions, answers, level)) 

async def eval_loop(
        rag_pipeline:RAGPipeline, 
        question_answer_pairs:list[tuple], 
        cash_client, 
        client_params:DecodingParameters,
        verbose:bool):
    
    if verbose: 
        print("="* 10, "Start Test", "="* 10)
        print(f"Number of Questions: {len(question_answer_pairs)}")
        print("="* 30)
    start_time = time.time()

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 
               'em_easy': 0, 'f1_easy': 0, 'prec_easy': 0, 'recall_easy': 0,
               'em_medium': 0, 'f1_medium': 0, 'prec_medium': 0, 'recall_medium': 0,
               'em_hard': 0, 'f1_hard': 0, 'prec_hard': 0, 'recall_hard': 0,
               }
    total_tokens_used = {'in': 0, 'out': 0, 'cached':0}
    # total_cost = {'in': 0, 'out': 0, 'total': 0} 

    generation_dict = {}

    for i, (ques, answ, lev) in enumerate(question_answer_pairs):
        ques_w_context = await rag_pipeline.execute(ques)
        # rag_ret_docs = rag_pipeline.docs

        response = await cash_client.request(
            prompt = ques_w_context,
            params = client_params,
            n = 1,
            request_id = f"hpqa",
            namespace="hpqa",
        )
        response = response[0]

        generation_dict.update({i: {
            'question': ques,
            'answer': answ,
            'level': lev,
            'response': response
        }})

        tokens_used_run = cash_client.tokens['default']['total']
        print('TOKENS USED RUN:',tokens_used_run)
        total_tokens_used = update_dict(total_tokens_used, tokens_used_run)
        # tokens_cost_run = tokens2cost(tokens_used_run, model_name)
        # total_tokens_cos = update_dict(total_tokens_used, tokens_used_run)
        em, prec, recall = update_answer(metrics=metrics, prediction=response, answer=answ, level=lev)
        if verbose: 
            print('Total Number of tokens:', tokens_used_run)
            print('prediction:', response)
            print('answer:', answ)

            print('Exact Match |', em)
            print('Precision |', prec)
            print('Recall |', recall)
            print('-'* 30)
    runtime = time.time() - start_time

    N = len(question_answer_pairs)
    for k in metrics.keys():
        metrics[k] /= N

    result_dict = {
        'metrics': metrics,
        'runtime': runtime,
        'tokens_used': total_tokens_used
    }

    # metrics['runtime'] = runtime

    if verbose: 
        print(" =" * 10)
        print(total_tokens_used)
        print(metrics)
        print('Runtime:', runtime)
    return result_dict, generation_dict #metrics, total_tokens_used, generation_dict