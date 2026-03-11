from retrieval.elasticsearch_retriever import ElasticsearchRetriever
from dotenv import load_dotenv
import os
from openai import OpenAI
from sentence_transformers import CrossEncoder
import torch
import json

# cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
cross_encoder = CrossEncoder('unicamp-dl/mMiniLM-L6-v2-mmarco-v2',
                            default_activation_function=torch.nn.Sigmoid(), 
                            max_length=500
                        )


def search(query: str, search_method: str="bm25", k: int=10, index_name: str="test_index_marker_markdown3level", retriever_weights=[0.7, 0.3]) -> list:
    """
    Perform a search using the given query and filters.

    Args:
        query (str): The search query.
        filters (str): The filters to apply to the search.

    Returns:
        list: The search results.

    """
    retriever = ElasticsearchRetriever(
    es_user=os.getenv('ES_USER'),
    es_password=os.getenv('ES_PASSWORD'),
    scheme='http',
    host=os.getenv('ES_HOST', 'localhost'),
    port=os.getenv('ES_PORT', '9200'),
    index=index_name
    )

    
    if search_method == "bm25":
        results = retriever.get_bm25_documents(
            query=query,
            # query_body=query_body,
            k=k
        )
    elif search_method == "hybrid":
        results = retriever.hybrid_search(
                query=query,
                k=k,
                retriever_weights=retriever_weights
            )
    elif search_method == "knn":
        results = retriever.get_knn_documents(
                query=query,
                k=k
            )
    elif search_method == "hybrid_cc":
        results = retriever.hybrid_search_cc(
                query=query,
                k=k,
                retriever_weights=retriever_weights
            )
    return results


def print_chunks(query: str, search_method: str="hybrid", k: int=5):
    """
    Search for relevant chunks and print the top 5 results with document ID.

    Args:
        query (str): The search query.
        search_method (str): The search method to use (e.g., "hybrid").
        k (int): Number of top results to print.
    """
    # Perform the search using the existing search function
    results = search(query, search_method, k)

    # Check if results exist
    if results:
        print(f"Question: {query}")
        print("Top 5 Relevant Chunks:")
        for i, result in enumerate(results, start=1):
            # print(result)
            doc_id = result["id"]  # Assuming the document ID is in '_id'
            chunk = result["page_content"] # Adjust 'content' based on your index schema
            print(f"{i}. Doc ID: {doc_id} - Chunk: {chunk}")
    else:
        print(f"No relevant results found for the query: {query}")



def rerank(query: str, docs: list):
    """
    Rerank the given list of results based on the given query using reranker.
 
    Args:
        query (str): The user's question.
        results (list): The list of results to rerank.
 
    Returns:
        list: The reranked list of results.
 
    The function constructs a prompt for the reranker model using the given query and the results.
   
    """
    if len(docs) == 0:
        return docs        
    sentence_pairs = [[query, doc] for doc in docs]
    ce_scores = cross_encoder.predict(sentence_pairs)
    scores = [{"data": data, 'ce_score': float(ce_scores[idx])} for idx, data in enumerate(docs)]
   
    # Sort list by CrossEncoder scores
    docs = sorted(scores, key=lambda x: x["ce_score"], reverse=True)
    return [json.loads(doc["data"]) for doc in docs]



