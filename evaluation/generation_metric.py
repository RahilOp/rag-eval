from sentence_transformers import SentenceTransformer, util
import torch

# Load the E5 model
model = SentenceTransformer("intfloat/multilingual-e5-small")

async def calculate_generation_metrics(reference: str, hypothesis: str) -> float:
    """
    Calculate the cosine similarity between the reference and hypothesis using E5 embeddings.
    
    Args:
    - reference (str): The ground truth/reference text.
    - hypothesis (str): The generated text to evaluate.

    Returns:
    - float: The cosine similarity score.
    """
    # Compute embeddings
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    hyp_embedding = model.encode(hypothesis, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_sim = util.pytorch_cos_sim(ref_embedding, hyp_embedding).item()
    
    return cosine_sim


