def show_chunks(docs):

    print("\n================ RETRIEVED CHUNKS ================\n")

    for i, doc in enumerate(docs):

        print(f"--- Chunk {i+1} ---\n")

        print(doc.page_content[:1200])

        print("\nMetadata:")
        print(doc.metadata)

        print("\n=================================================\n")