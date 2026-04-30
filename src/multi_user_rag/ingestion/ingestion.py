from langchain_core.documents import Document

from multi_user_rag.ingestion.loader import load_simplewiki
from multi_user_rag.ingestion.splitter import create_splitter
from multi_user_rag.ingestion.vector_store import VectorStoreService
from multi_user_rag.ingestion.user_loader import load_user_documents
from multi_user_rag.config import (CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL, DISTANCE_METRIC, USER_DATA_DIRECTORY)


class IngestionService:

    def __init__(self):
        # Create splitter
        self.splitter = create_splitter(
                                        chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP
                                       )

        # Create vector store
        self.vector_store = VectorStoreService(
                                                embedding_model_name=EMBEDDING_MODEL,
                                                persist_directory=PERSIST_DIRECTORY,
                                                collection_name=COLLECTION_NAME,
                                                distance_metric= DISTANCE_METRIC
                                              )

    def ingest_simplewiki(self):
        
        # Load text
        print("Loading SimpleWiki dataset...")
        documents = load_simplewiki()

        # Split
        print("Splitting documents...")
        chunks = self.splitter.split_documents(documents)

        # Store
        print("Adding chunks to vector store...") 
        self.vector_store.add_documents(chunks)

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
        }

    def ingest_user_documents(self, user_id: str):

        print(f"Ingesting documents for user: {user_id}")

        documents = load_user_documents(user_id, USER_DATA_DIRECTORY)

        if not documents:
            print("No user documents found.")
            return

        chunks = self.splitter.split_documents(documents)

        self.vector_store.add_documents(chunks)

        print(f"Ingested {len(chunks)} user chunks.")