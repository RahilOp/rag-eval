from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ContextEntityRecall, FactualCorrectness, SemanticSimilarity, RougeScore, NonLLMStringSimilarity
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

# Load the .env file
load_dotenv('.env')


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

OPENAI_ENGINE = os.getenv("OPENAI_ENGINE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


openai_model =  LangchainLLMWrapper(ChatOpenAI(
    model=os.getenv("OPENAI_ENGINE"),
    temperature=0
))

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
 
# Initialize the OpenAI client
client = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
)
 
def get_embeddings(input_text: str) -> List[float]:
    """
    Fetches embeddings for the input text using the OpenAI client.
    
    Parameters:
    - input_text (str): The input text to generate embeddings for.
    
    Returns:
    - List[float]: The embeddings for the input text as a list of floats.
    """
    # start = time()
   
    # Create embeddings using the client
    emb = client.embeddings.create(
        model="BAAI/bge-m3",
        input=input_text,
    )
   
    # print(f"Time taken to fetch embeddings: {time() - start} seconds")
   
    # Return the embedding as a list of floats
    return emb.data[0].embedding  # type: ignore


async def get_rouge_score(reference_context: str, retrieved_context: str) -> float:
    """
    Asynchronously calculates the ROUGE score for a single turn using RAGAS.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_context (str): Retrieved contexts for the query.
    
    Returns:
    - float: ROUGE score.
    """
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    RougeScore_sample = SingleTurnSample(
            response=retrieved_context,
            reference=reference_context,
    )

    # Initialize the ROUGE score metric
    RougeScore_scorer = RougeScore()

    # Calculate the ROUGE score
    rouge_score = await RougeScore_scorer.single_turn_ascore(sample = RougeScore_sample)

    return rouge_score
async def get_factual_correctness(reference_context: str, retrieved_context: str) -> float:
    """
    Asynchronously calculates the factual correctness score for a single turn using RAGAS.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_context (str): Retrieved contexts for the query.
    
    Returns:
    - float: Factual correctness score.
    """
    
    # Initialize the FactualCorrectness scorer
    FactualCorrectness_scorer = FactualCorrectness()
    FactualCorrectness_scorer.llm = openai_model
    
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    FactualCorrectness_sample = SingleTurnSample(
            response=retrieved_context,
            reference=reference_context,
    )
    
    # Calculate the factual correctness score
    factual_score = await FactualCorrectness_scorer.single_turn_ascore(sample = FactualCorrectness_sample)
    
    return factual_score

async def get_metric_score(reference_context, retrieved_context):
    """
    Asynchronously calculates a weighted score for a single turn using RAGAS and semantic similarity.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_context (str): Retrieved contexts for the query.
    
    Returns:
    - float: Weighted score.
    """

    # Define weights for the different metrics
    p = 0.2  # semantic similarity
    q = 0.25  # factual correctness
    r = 0.4  # context entity recall
    s = 0.4  # ROUGE score

    # Calculate the semantic similarity score
    ContextEntityRecall_scorer = ContextEntityRecall(llm=openai_model)
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    ContextEntityRecall_sample = SingleTurnSample(
        reference=reference_context,
        retrieved_contexts=[retrieved_context]
    )

    context_entity_recall_score = await ContextEntityRecall_scorer.single_turn_ascore(ContextEntityRecall_sample)
    
    # Calculate the factual correctness score
    # factual_score = await get_factual_correctness(reference_context, retrieved_context)
    
    # Calculate the ROUGE score
    rouge_score = await get_rouge_score(reference_context, retrieved_context)
    
    # Calculate the semantic similarity score
    semantic_score = cosine_similarity(
        np.array(get_embeddings(reference_context)).reshape(1, -1), 
        np.array(get_embeddings(retrieved_context)).reshape(1, -1)
    )[0][0]
    

    # print("context_entity_recall_score", context_entity_recall_score)
    # print("factual_score", factual_score)
    # print("RougeScore", rouge_score)
    # print("semantic_score", semantic_score)
    # Calculate the weighted score
    weighted_score = p * semantic_score + r * context_entity_recall_score + s * rouge_score
    
    return weighted_score

async def calculate_individual_score_async(reference_context, retrieved_contexts):
    """
    Asynchronously calculates context precision for a single turn using RAGAS.

    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_contexts (list): List of retrieved contexts for the query. List(str)

    Returns:
    - List of float: Semantic scores.
    """
    results = []

    # Process each retrieved context sequentially
    for single_retrieved_context in retrieved_contexts:
        try:
            # Await the result for each retrieved context
            score = await get_metric_score(reference_context, single_retrieved_context)
            results.append(score)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing retrieved context: {single_retrieved_context}, Error: {e}")
            results.append(0.0)  # Append a default score in case of an error

    return results

# Example usage in an async context
async def main():
    reference_context = "Tokyo Tower"
    retrieved_contexts = "The Eiffel Tower is located in Paris."
    
    metric_score = await get_metric_score(reference_context, retrieved_contexts)
    print("Metric score:", metric_score)

# Running the example
# import asyncio
# asyncio.run(main())
