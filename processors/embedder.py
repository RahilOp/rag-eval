# # from langchain.embeddings import HuggingFaceEmbeddings
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# from typing import List

# # Load environment variables from the .env file
# load_dotenv()

# EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
# EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")



# class EmbeddingHandler:
#     def __init__(self, model_name="BAAI/bge-m3", device="cuda"):
#         # Initialize the OpenAI client
#         self.client = OpenAI(
#             api_key=EMBEDDING_API_KEY,
#             base_url=EMBEDDING_BASE_URL,
#         )
#     def get_embeddings(self, input_text: str) -> List[float]:
#         """Fetch embeddings for the input text using the OpenAI client."""
#         # start = time()
        
#         # Create embeddings using the client
#         emb = self.client.embeddings.create(
#             model="BAAI/bge-m3",
#             input=input_text,
#         )
        
#         # print(f"Time taken to fetch embeddings: {time() - start} seconds")
        
#         # Return the embedding as a list of floats
#         return emb.data[0].embedding  # type: ignore
    



from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingHandler:
    def __init__(self, model_name="BAAI/bge-m3", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': False}
        )

    def get_embeddings(self):
        return self.embeddings


