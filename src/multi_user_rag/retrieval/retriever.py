from multi_user_rag.ingestion.vector_store import VectorStoreService
from multi_user_rag.config import PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL, DISTANCE_METRIC


class RetrieverService:

    def __init__(self):

        self.vector_store_service = VectorStoreService(
                                                        embedding_model_name=EMBEDDING_MODEL,
                                                        persist_directory=PERSIST_DIRECTORY,
                                                        collection_name=COLLECTION_NAME,
                                                        distance_metric=DISTANCE_METRIC
                                                      )

    def get_retriever(self, user_id: str, search_type: str = "similarity", k: int = 5):

        if user_id:
            return self.vector_store_service.as_retriever(
                                                    search_type=search_type,
                                                    search_kwargs={
                                                                    "k": k,
                                                                    "filter": {
                                                                                "$or": [
                                                                                        {"source": "simplewiki"},
                                                                                        {"user_id": user_id}
                                                                                       ]
                                                                                }
                                                                  }
                                                 )
        
        return self.vector_store_service.as_retriever(
                                                        search_type=search_type, 
                                                        search_kwargs={"k": k}
                                                     )

 

