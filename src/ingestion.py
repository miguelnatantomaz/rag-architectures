from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys


def load_pdf(pdf_path):

    if not os.path.exists(pdf_path):

        print(f"\n[ERROR] PDF not found: {pdf_path}\n")

        sys.exit()

    if not pdf_path.lower().endswith(".pdf"):

        print("\n[ERROR] The selected file is not a PDF.\n")

        sys.exit()

    try:

        loader = PyPDFLoader(
            pdf_path,
            extract_images=False
        )

        return loader.load()

    except Exception as e:

        print("\n[ERROR] Failed to load PDF.\n")

        print(f"Details: {e}\n")

        sys.exit()


def naive_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)


def rerank_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=20,
    )

    return splitter.split_documents(documents)


def parent_splitters():
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    return child_splitter, parent_splitter