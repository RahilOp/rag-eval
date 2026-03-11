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