from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity, RougeScore
from ragas import evaluate

from dotenv import load_dotenv

# Load the .env file
load_dotenv('.env')

import os
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")


azure_configs = {
    "base_url": AZURE_OPENAI_ENDPOINT,
    "model_deployment": AZURE_OPENAI_DEPLOYMENT,
    "model_name": "gpt-4",  # Model name as understood by Langchain, alias to GPT-4 in this case
    "embedding_deployment": AZURE_EMBEDDING_DEPLOYMENT,
    "embedding_name": "text-embedding-3-large"  # Set an alias or appropriate model name for embeddings
}


# Load Azure OpenAI credentials from environment variables
evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
))

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
))


semantic_scorer = SemanticSimilarity()
semantic_scorer.embeddings = evaluator_embeddings

rouge_scorer = RougeScore()

# Initialize the context precision metric
context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

async def calculate_context_precision_async(query, reference_context, retrieved_contexts):
    """
    Asynchronously calculates context precision for a single turn using RAGAS.
    
    Parameters:
    - query (str): The user's query or input.
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_contexts (list): List of retrieved contexts for the query.
    
    Returns:
    - float: Context precision score.
    """
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    sample = SingleTurnSample(
        user_input=query,
        reference=reference_context,
        retrieved_contexts=retrieved_contexts
    )
    
    # Await and calculate the context precision score for this sample
    precision_score = await context_precision.single_turn_ascore(sample)
    
    return precision_score


async def calculate_semantic_similarity_async(reference_context, retrieved_context):
    """
    Asynchronously calculates context precision for a single turn using RAGAS.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_context (str): Retrieved contexts for the query.
    
    Returns:
    - float: Semantic score.
    """
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    sample = SingleTurnSample(
        reference=reference_context,
        response=retrieved_context
    )
    
    semantic_score = await semantic_scorer.single_turn_ascore(sample)
    
    return semantic_score

async def get_rouge_score(reference_context, retrieved_context):
    """
    Asynchronously calculates context precision for a single turn using RAGAS.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_context (str): Retrieved contexts for the query.
    
    Returns:
    - float: Semantic score.
    """
    # Create a SingleTurnSample with query, reference answer, and retrieved contexts
    sample = SingleTurnSample(
        reference=reference_context,
        response=retrieved_context
    )
    
    rouge_score = await rouge_scorer.single_turn_ascore(sample)
    
    return rouge_score


async def calculate_individual_score_async(reference_context, retrieved_contexts):
    """
    Asynchronously calculates context precision for a single turn using RAGAS.
    
    Parameters:
    - reference_context (str): The ground truth or reference context for the query.
    - retrieved_contexts (list): List of retrieved contexts for the query. List(str)
    
    Returns:
    - List of float: Semantic scores.
    """
    rouge_scores = []
    for single_retrieved_context in retrieved_contexts:
        rouge_score = await get_rouge_score(reference_context, single_retrieved_context)
        rouge_scores.append(rouge_score)
    
    return rouge_scores

# Example usage in an async context
async def main():
    query = "Where is the Eiffel Tower located?"
    reference_context = "Tokyo Tower"
    retrieved_contexts = "The Eiffel Tower is located in Paris."
    
    semantic_score = await get_rouge_score(reference_context, retrieved_contexts)
    print("Semantic score:", semantic_score)

# Running the example
# import asyncio
# asyncio.run(main())
