from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)

from src.ingestion import (
    naive_chunks,
    rerank_chunks,
    parent_splitters
)

from src.prompts import get_prompt


def build_naive_rag(documents, embeddings, llm, persist_dir):

    chunks = naive_chunks(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        }
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def build_parent_rag(documents, embeddings, llm, persist_dir):

    child_splitter, parent_splitter = parent_splitters()

    store = InMemoryStore()

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter

    )

    BATCH_SIZE = 100
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        print(f"Processando lote {i//BATCH_SIZE + 1} de {len(documents)//BATCH_SIZE + 1}...")
        retriever.add_documents(batch, ids=None)

    print("Indexação completa!")

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def build_rerank_rag(
    documents,
    embeddings,
    llm,
    persist_dir,
    cohere_api_key
):

    chunks = rerank_chunks(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    reranker = CohereRerank(
        cohere_api_key=cohere_api_key,
        top_n=3,
        model="rerank-v3.5"
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    setup = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": compression_retriever
    })

    chain = (
        setup
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    return compression_retriever, chain