from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_splitter(chunk_size: int, chunk_overlap: int):
    """
    Factory function to create a text splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

