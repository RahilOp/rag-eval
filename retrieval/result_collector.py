from retrieval.search import search, rerank
import pandas as pd
import json

def search_and_append_results(input_file="./data/ground_truth/qa_dataset.csv", output_file="./results/results.csv", top_k=10, index_name="sample_index", search_method="bm25", retriever_weights=[0.7, 0.3], reranked=False):
    """
    Read queries from a CSV file, search them in Elasticsearch, and append retrieved document chunks in a new column.
    """
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    if 'Question' not in df.columns:
        print("CSV file must contain a 'Question' column.")
        for column in df.columns:
            print(column)
        return

    # Initialize list to store retrieved document IDs for each query
    retrieved_docs = []

    # Iterate over each query in the DataFrame
    for query in df['Question']:
        # Retrieve document IDs from Elasticsearch for the current query
        retrieved_docs_search = search(query, search_method=search_method, k=top_k, index_name=index_name, retriever_weights=retriever_weights)

        if reranked:
            results_arr = []
            for index in range(0,len(retrieved_docs_search)):
                # entry = {
                #     'id' : results[index].id,
                #     'page_content': results[index].page_content,
                #     'metadata':results[index].metadata
                # }
                results_arr.append(json.dumps(retrieved_docs_search[index], ensure_ascii = False))

            retrieved_docs_search = rerank(query, results_arr)

        # print(len(retrieved_docs_search))
        retrieved_docs_string = []
        for i in range(0,len(retrieved_docs_search)):
            retrieved_docs_string.append(retrieved_docs_search[i]['page_content'])
           

        retrieved_docs.append(json.dumps(retrieved_docs_string, ensure_ascii=False))
    

    # Append the retrieved document IDs as a new column in the DataFrame
    df['Retrieved_Docs'] = retrieved_docs

    # Save the updated DataFrame back to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")



