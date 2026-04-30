import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from multi_user_rag.ingestion.ingestion import IngestionService


def main():

    user_id = input("Enter user_id: ")

    ingestion = IngestionService()

    ingestion.ingest_user_documents(user_id)


if __name__ == "__main__":
    main()
