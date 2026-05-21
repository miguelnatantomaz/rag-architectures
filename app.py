from src.ingestion import load_pdf
from src.embeddings import get_embeddings
from src.llm import get_llm
from halo import Halo
from src.healthcheck import check_ollama
from src.rag_types import (
    build_naive_rag,
    build_parent_rag,
    build_rerank_rag
)

from src.utils import show_chunks


def main():
    print("\nChecking Ollama...\n")

    check_ollama()

    print("\n=== Local RAG Architectures ===\n")

    pdf_path = input("PDF path: ")

    rag_type = input(
        "\nChoose RAG type:\n"
        "1 - Naive RAG\n"
        "2 - Parent RAG\n"
        "3 - Rerank RAG\n\n"
        "Choice: "
    )

    show_chunks_option = input(
        "\nDo you want to visualize retrieved chunks? (y/n): "
    ).lower()

    show_retrieved_chunks = show_chunks_option == "y"

    spinner = Halo(text='Loading PDF...', spinner='dots')
    spinner.start()

    documents = load_pdf(pdf_path)

    spinner.succeed('PDF loaded successfully.')

    llm = get_llm()
    embeddings = get_embeddings()

    if rag_type == "1":
        spinner = Halo(text='Building Naive RAG...', spinner='dots')
        spinner.start()
        retriever, chain = build_naive_rag(
            documents,
            embeddings,
            llm,
            persist_dir="chroma_db/naive"
        )
        spinner.succeed('Naive RAG ready.')

    elif rag_type == "2":
        spinner = Halo(text='Building Parent RAG...', spinner='dots')
        spinner.start()
        retriever, chain = build_parent_rag(
            documents,
            embeddings,
            llm,
            persist_dir="chroma_db/parent"
        )
        spinner.succeed('Parent RAG ready.')

    elif rag_type == "3":

        cohere_api_key = input("\nEnter Cohere API Key: ")

        spinner = Halo(text='Building Rerank RAG...', spinner='dots')
        spinner.start()
        retriever, chain = build_rerank_rag(
            documents,
            embeddings,
            llm,
            persist_dir="chroma_db/rerank",
            cohere_api_key=cohere_api_key
        )
        spinner.succeed('Rerank RAG ready.')

    else:
        print("Invalid option.")
        return
    


    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            print("\nExiting application...")
            break

        docs = retriever.invoke(question)

        if show_retrieved_chunks:
            show_chunks(docs)

        answer = chain.invoke(question)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n\n[INFO] Application interrupted by user.\n")