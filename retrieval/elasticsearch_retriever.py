from typing import List, Dict
# from langchain.schema import Document
# from langchain.schema.embeddings import Embeddings
from elasticsearch import Elasticsearch
from retrieval.embeddings import get_embeddings
 
from openai import OpenAI
from typing import List
from time import time

 
class ElasticsearchRetriever():
 
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        es_user: str = None,
        es_password: str = None,
        index: str = 'default_index',
        ) -> None:
        """
        Initializes an instance of the class.

        Args:
            host (str): The hostname or IP address of the Elasticsearch server. Defaults to "localhost".
            port (int): The port number of the Elasticsearch server. Defaults to 9200.
            scheme (str): The scheme to use for the Elasticsearch connection. Defaults to "http".
            es_user (str): The username for the Elasticsearch authentication.
            es_password (str): The password for the Elasticsearch authentication.
            index (str): The name of the Elasticsearch index to use.

        Returns:
            None
        """
        self.client = Elasticsearch(
            f"{scheme}://{host}:{port}",
            http_auth=(es_user, es_password),
            verify_certs=False)
        # self.embedding_model = embedding_model
        self.index = index
        self.search_after = []
 
    def hybrid_search_cc(self,
        query: str,
        filter: List[Dict]=[],
        k: int=10,
        fetch_k: int=100,
        vector_query_field: str="vector",
        text_field: str="text",
        retriever_weights: List[float]=[0.7, 0.3],
        load_more: bool=False,
        **kwargs
    ):
        show_load_more = True
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [text_field],
                                "boost": retriever_weights[0],
                            }
                        }
                    ],
                    "filter": filter
                }
            },
            "knn": {
                "field": vector_query_field,
                "filter": filter,
                "query_vector": get_embeddings(query),
                "k": k,
                "num_candidates": fetch_k,
                "boost": retriever_weights[1],
            },
            "size": k
        }
       
        if load_more:
            query_body["search_after"] = self.search_after
 
        response = self.client.search(
            index=self.index,
            body=query_body,
            **kwargs
        )
       
        # if(len(response["hits"]["hits"]) < k):
        #     show_load_more = False
 
        hits = [hit for hit in response["hits"]["hits"]]
       
        # Create a list of Document objects from the hits
        docs = [
            {
                "id":hit["_id"],
                "page_content":hit["_source"]['text'],
                "metadata":hit["_source"]["metadata"],
            }
            for hit in hits
        ]
       
       
        return docs
   
 
    def sharepoint_search(self,
        query: str,
        filter: List[Dict]=[],
        k: int=10,
        fetch_k: int=100,
        vector_query_field: str="vector",
        text_field: List[str]=["body", "name"],
        retriever_weights: List[float]=[0.7, 0.3],
        load_more: bool=False,
        **kwargs
    ):
        show_load_more = True
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["body", "name"],
                                # "operator": "or",
                            }
                        }
                    ],
                    "filter": filter
                }
            },            
            # "knn": {
            #     "field": vector_query_field,
            #     "filter": filter,
            #     "query_vector": self.embedding_model.embed_query(query),
            #     "k": k,
            #     "num_candidates": fetch_k,
            #     "boost": retriever_weights[1],
            # },
           "sort":[
                {"_score": "desc"},
                {"name.enum": "asc"}
           ],
            "size": k
        }
       
 
        if load_more and self.search_after:
            query_body["search_after"] = self.search_after
 
        response = self.client.search(
            index=self.index,
            body=query_body,
            **kwargs
        )
        if(len(response["hits"]["hits"]) < k):
            show_load_more = False
 
        hits = [hit for hit in response["hits"]["hits"]]
       
        def parse_metadata(entry):
            return {
                "file_type": entry["file"]['mimeType'] if "file" in entry else "" ,
                "url": entry['webUrl'] if "webUrl" in entry else "",
                "name": entry['name'] if "name" in entry else "",
                'created_by': entry["createdBy"]["user"]["email"] if "createdBy" in entry and "user" in entry["createdBy"] and "email" in entry["createdBy"]["user"] else "",
            }
 
        # Create a list of Document objects from the hits
        docs = [
            {
                "id":hit["_id"],
                "page_content":hit["_source"]['body'] if "body" in hit['_source'] else "",
                "metadata": parse_metadata(hit['_source']),
            }
            for hit in hits
        ]
        hits = response.get("hits", {}).get("hits", [])
        self.search_after = hits[-1].get('sort') if hits else None
 
 
        # self.search_after = response["hits"]['hits'][-1]['sort']
 
        return docs, show_load_more
 
 
    def hybrid_search(
        self,
        query: str,
        filter: List[Dict]=[],
        k: int=10,
        fetch_k: int=100,
        vector_query_field: str="vector",
        text_field: str="text",
        retriever_weights: List[float]=[0.7, 0.3],
        load_more: bool=False
    ) -> List[dict]:
        """
        Retrieves a list of documents based on the given index, query, and filter.
        Offers an option to set the retriever weights to perform hybrid search.
 
        Args:
            index (str): The index to search in.
            query (str): The query string.
            filter (List[Dict], optional): A list of filter dictionaries. Defaults to [].
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            fetch_k (int, optional): The number of documents to fetch before reranking. Defaults to 100.
            vector_query_field (str, optional): The name of the field containing the vector query. Defaults to "vector".
            text_field (str, optional): The name of the field containing the text. Defaults to "text".
            retriever_weights (List[float], optional): The weights for the BM25 and KNN retrievers. Defaults to [0.7, 0.3].
 
        Returns:
            List[Document]: A list of Document objects representing the retrieved documents.
        """
       
        # Obtain BM25 documents
        bm25_docs = self.get_bm25_documents(
            query=query,
            filter=filter,
            k=k//2,
            text_field=text_field)
       
        # Obtain KNN documents
        knn_docs = self.get_knn_documents(
            query=query,
            filter=filter,
            k=k//2,
            fetch_k=fetch_k,
            vector_query_field=vector_query_field)
       
        # Perform Weighted Reciprocal Rank Fusion
        reranked_docs = self.weighted_reciprocal_rank(doc_lists=[bm25_docs, knn_docs], weights=retriever_weights)
       
        return reranked_docs
   
 
    def get_bm25_documents(
        self,
        query: str,
        filter: List[Dict]=[],
        text_field: str="text",
        k: int=10
    ) -> List[dict]:
        """
        Retrieves a list of BM25 documents based on the given parameters.
       
        Args:
            index (str): The name of the index to search in.
            query (str): The query string.
            filter (List[Dict], optional): The filter to apply to the query. Defaults to [].
            text_field (str, optional): The name of the field to match the query against. Defaults to "text".
            k (int, optional): The maximum number of documents to retrieve. Defaults to 10.
       
        Returns:
            List[Document]: A list of Document objects.
        """
       
        # Construct the query body
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ['text']
                            }
                        }
                    ],
                    "filter": filter,
                }
            },
        }
 
        # Perform the search
        bm25_response = self.client.search(
            index=self.index,
            **query_body,
            size=k)
       
        # Extract the hits from the response
        hits = [hit for hit in bm25_response["hits"]["hits"]]
       
        # Create a list of Document objects from the hits
        bm25_docs = [
            {
                "id": hit["_id"],
                "page_content": hit["_source"]['text'],
                "metadata": hit["_source"]["metadata"],
            }
            for hit in hits
        ]
 
        return bm25_docs
       
 
    def get_knn_documents(
        self,
        query: str,
        filter: List[Dict]=[],
        vector_query_field: str="vector",
        fetch_k: int=100,
        k: int=10
    ) -> List[dict]:
        """
        Retrieves the k nearest neighbor documents from the specified index based on a query.
 
        Args:ca
            index (str): The name of the index to search.
            query (str): The query string.
            filter (List[Dict], optional): A list of filters to apply to the search. Defaults to an empty list.
            vector_query_field (str, optional): The name of the field containing the vectors to search. Defaults to "vector".
            fetch_k (int, optional): The number of candidates to retrieve. Defaults to 100.
            k (int, optional): The number of nearest neighbor documents to retrieve. Defaults to 10.
 
        Returns:
            List[Document]: A list of Document objects representing the k nearest neighbor documents.
        """
       
        # Embed the query using the embedding model
        query_vector = get_embeddings(query)
       
        # Construct the query body for the KNN search
        query_body = {
            "knn": {
                "filter": filter,
                "field": vector_query_field,
                "k": k,
                "query_vector": query_vector,
                "num_candidates": fetch_k,
            },
        }
       
        # Perform the KNN search using the Elasticsearch client
        knn_response = self.client.search(
            index=self.index,
            **query_body,
            size=k
        )
 
        # Extract the hits from the search response
        hits = [hit for hit in knn_response["hits"]["hits"]]
       
        # Create Document objects from the hits
        knn_docs = [
            {
                "id": hit["_id"],
                "page_content": hit["_source"]['text'],
                "metadata": hit["_source"]["metadata"],
            }
            for hit in hits
        ]
       
        return knn_docs
 
 
    def weighted_reciprocal_rank(
        self,
        doc_lists: List[List[dict]],
        weights: List[float]
    ) -> List[dict]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
 
        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.
            weights: A list of weights to be set for BM25 and KNN retrievers.
 
        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
       
        # RRF Constant
        c: int = 60
 
        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc["page_content"])
               
        # print(f"All documents: {len(all_documents)} \n {all_documents}")
 
        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}
 
        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + c))
                rrf_score_dic[doc["page_content"]] += rrf_score
 
        # Sort documents by their RRF scores in descending order
        sorted_documents = sorted(
            rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
        )
 
        # Map the sorted page_content back to the original document objects
        page_content_to_doc_map = {
            doc["page_content"]: doc for doc_list in doc_lists for doc in doc_list
        }
        sorted_docs = [
            page_content_to_doc_map[page_content] for page_content in sorted_documents
        ]
        return sorted_docs
 
 