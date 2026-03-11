from retrieval.result_collector import search_and_append_results
from evaluation.context_precision import calculate_individual_score_async
from evaluation.generation_metric import calculate_generation_metrics
import pandas as pd
import json
import numpy as np
from config_logger import logger
import os
from dotenv import load_dotenv
import openai

load_dotenv('.env')

TOP_K_CONTEXT = 3

_client = None
_model_name = None

def _get_client():
    global _client, _model_name
    if _client is None:
        _model_name = os.environ['GPT4O_MINI_OPENAI_DEPLOYMENT_NAME']
        _client = openai.OpenAI(
            base_url=os.environ['GPT4O_MINI_OPENAI_RESOURCE_ENDPOINT'],
            api_key=os.environ['GPT4O_MINI_OPENAI_API_KEY'],
        )
    return _client, _model_name


async def get_answer(query, context):
    system_prompt = """
    You are a helpful assistant who answers users' questions based on multiple contexts given to you.
    Keep your answer in easy-to-understand markdown format and to the point. The content given below is from multiple contexts.
    Try to keep the explanation as short as possible. Don't deviate from the instructions otherwise you will be penalised heavily.
    """

    user_prompt = f'''
    You have to answer the following question based on the context provided.
    If you are not sure about the answer, you can say "I don't know" or "I am not sure".

    Question: {query}
    Context: {context[0] + context[1] + context[2]}
    You have to generate a brief and accurate answer to the question based on the context provided without deviating.
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client, model_name = _get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        stream=False
    )

    return response.choices[0].message.content


async def run_evaluation_pipeline(
    ground_truth_file="./data/ground_truth/qa_dataset.csv",
    index_name='sample_index',
    retriever_weights=[0.7, 0.3],
    search_method="bm25",
    reranked=False,
    test_retrieval=True,
    test_generation=False,
):
    _, model_name = _get_client()
    metric_results_file = "./results/" + str(index_name) + "_" + str(search_method) + "_" + str(retriever_weights) + "_" + str(model_name.replace("/", "-")) + "_metric_results.csv" if reranked else "./results/" + str(index_name) + "_" + str(search_method) + "_" + str(retriever_weights) + "_unranked" + "_" + str(model_name) + "_metric_results.csv"
    retrieval_results_file = "./results/" + str(index_name) + "_" + str(search_method) + "_" + str(retriever_weights) + "_results.csv" if reranked else "./results/" + str(index_name) + "_" + str(search_method) + "_" + str(retriever_weights) + "_unranked" + "_results.csv"

    search_and_append_results(
        input_file=ground_truth_file,
        output_file=retrieval_results_file,
        index_name=index_name,
        retriever_weights=retriever_weights,
        search_method=search_method,
        reranked=reranked
    )

    df = pd.read_csv(retrieval_results_file, encoding='utf-8-sig')
    individual_cp = []
    rr = []
    mrr = 0
    generation_scores = []
    llm_responses = []
    final_mrr = 0
    avg_generation_score = 0

    for _, row in df.iterrows():
        query = row['Question']
        answer = row['Answer']
        gt_context = row['Context']
        retrieved_context = json.loads(row['Retrieved_Docs'])

        if test_retrieval:
            scores = await calculate_individual_score_async(gt_context, retrieved_context)
            individual_cp.append(json.dumps(scores, ensure_ascii=False))
            rank = np.argmax(scores) + 1
            rr.append(1 / rank)
            mrr += 1 / rank

        if test_generation:
            llm_response = await get_answer(query, retrieved_context[:TOP_K_CONTEXT])
            generation_score = await calculate_generation_metrics(answer, llm_response)
            generation_scores.append(generation_score)
            llm_responses.append(llm_response)

    if test_retrieval:
        final_mrr = mrr / len(df)
        logger.warning(f"Config - Index: {index_name}, Method: {search_method}, Weights: {retriever_weights}, Reranked: {reranked} => MRR: {final_mrr}")
        df['individual_cp'] = individual_cp
        df['rr'] = rr

    if test_generation:
        avg_generation_score = np.mean(generation_scores)
        logger.warning(f"Model: {model_name}, Index: {index_name}, Method: {search_method}, Weights: {retriever_weights}, Reranked: {reranked} => Avg Score: {avg_generation_score}")
        df['llm_response'] = llm_responses
        df['generation_score'] = generation_scores

    df.to_csv(metric_results_file, index=False)

    return final_mrr, avg_generation_score, model_name
