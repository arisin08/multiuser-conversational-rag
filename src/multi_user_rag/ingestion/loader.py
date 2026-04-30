import gzip
import json
from typing import List
from langchain_core.documents import Document
from multi_user_rag.config import (RAW_DATA_PATH, WIKI_KEYWORDS)

def load_simplewiki():
    """
    Loads first paragraphs from simplewiki jsonl.gz file.
    """
    filepath=RAW_DATA_PATH
    passages = []
    docs=[]

    with gzip.open(filepath, "rt", encoding="utf8") as f:
        for line in f:
            data = json.loads(line.strip())
            passages.append(data["paragraphs"][0])

        # # We subset our data so we only use a subset of wikipedia to run things faster
        # passages= [passage for passage in passages for x in WIKI_KEYWORDS if x in passage.lower().split()]

        docs=[Document(page_content=doc) for doc in passages]    

    return docs
