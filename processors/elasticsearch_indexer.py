from langchain.vectorstores.elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

class ElasticsearchIndexer:
    def __init__(self, index_name: str, es_url: str = os.getenv("es_url"), es_user: str = os.getenv("es_user"), es_password: str = os.getenv("es_password")):
        self.index_name = index_name
        self.es_url = es_url
        self.es_user = es_user
        self.es_password = es_password
        # self.es = ElasticsearchStore(
        #     index_name=self.index_name,
        #     url=self.es_url,
        #     username=self.es_user,
        #     password=self.es_password,
        # )

    def index_data(self, chunks, embeddings):
        if type(chunks[0]) == str:
            self.from_texts(chunks, embeddings)
        else:
            self.from_documents(chunks, embeddings)


    def from_documents(self, documents, embeddings):
        # Configure Elasticsearch client with a custom timeout
        es_client = Elasticsearch(
            hosts=[self.es_url], 
            timeout=60,  # Timeout set to 30 seconds
        )

        db = ElasticsearchStore.from_documents(
        es_connection=es_client,
        # es_url=self.es_url,
        index_name=self.index_name,
        embedding=embeddings,
        documents=documents,
        # es_user=es_user,
        # es_password=es_password,
        es_api_key=os.getenv("ES_API_KEY"),
        # query_field='content',
        # distance_strategy="COSINE",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        # vector_query_field='vector',
        )

    def from_texts(self, texts, embeddings):
        # Configure Elasticsearch client with a custom timeout
        es_client = Elasticsearch(
            hosts=[self.es_url],
            timeout=60,  # Timeout set to 30 seconds
        )

        db = ElasticsearchStore.from_texts(
        es_connection=es_client,
        # es_url=self.es_url,
        index_name=self.index_name,
        embedding=embeddings,
        texts=texts,
        # metadatas=[metadatas],
        # es_user=es_user,
        # es_password=es_password,
        es_api_key=os.getenv("ES_API_KEY"),
        # query_field='content',
        # distance_strategy="COSINE",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        # vector_query_field='vector'
        )