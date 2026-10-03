from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )