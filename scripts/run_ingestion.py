import sys
import os
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from multi_user_rag.ingestion.ingestion import IngestionService


def main():

    print("\n========== RAG Ingestion Pipeline ==========\n")
    ingestor=IngestionService()
    
    print("Starting ingestion...\n")
    ingestor.ingest_simplewiki()

    print("\nIngestion completed successfully.")
    print("Vector store is ready for inference.\n")


if __name__ == "__main__":
    main()