from langchain_ollama import OllamaEmbeddings


def get_embeddings(model="llama3.2"):
    return OllamaEmbeddings(model=model)