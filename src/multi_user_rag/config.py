CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "data/vector_store"
COLLECTION_NAME = "multi_user_collection"
DISTANCE_METRIC = "cosine"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL="gpt-4o-mini"
EVALUATOR_MODEL="gpt-4o"
TEMPERATURE=0
RAW_DATA_PATH = "data/raw/simplewiki-2020-11-01.jsonl.gz"
USER_DATA_DIRECTORY = "data/user_uploads"
WIKI_KEYWORDS = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural network",
    "transformer",
    "retrieval",
    "language model",
    "india",
    "animals",
    "environment", 
    "football", 
    "premiere league",
    "f1",
    "le mans",
    "dakar"
]