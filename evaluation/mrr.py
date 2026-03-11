import json
import pandas as pd

def reciprocal_rank(retrieved_doc_ids, relevant_doc_ids):
    """
    Calculate the reciprocal rank for a single query.
    Returns 1/rank of the first relevant doc if found, else 0.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_ids:
            return 1 / rank
    return 0.0

def mean_reciprocal_rank(input_file="./data/ground_truth/qa_dataset.csv"):
    """
    Calculate Mean Reciprocal Rank (MRR) for a list of queries and save results back to the same file.
    """

    # Load test dataset
    df = pd.read_csv(input_file)

    # Extract test queries and document IDs
    relevant_doc_ids = df[f'Relevant_doc']
    retrieved_doc_ids = df[f'Retrieved_Doc_IDs'].apply(lambda x: json.loads(x) if pd.notna(x) else []).tolist()

    # Calculate reciprocal rank for each query
    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_doc_ids, relevant_doc_ids)
    ]

    # Add reciprocal ranks to DataFrame as a new column
    df[f'Reciprocal_Rank'] = reciprocal_ranks

    # Calculate Mean Reciprocal Rank
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    # Save updated DataFrame back to the original file
    df.to_csv(input_file, index=False)

    # Print Mean Reciprocal Rank
    print(f"Mean Reciprocal Rank (MRR): {mrr}")
    return mrr

