from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStoreService:

    def __init__(self, embedding_model_name: str, persist_directory: str, collection_name: str, distance_metric :str):

        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

        self.vector_store = Chroma(
                                    collection_name=collection_name,
                                    embedding_function=self.embedding_model,
                                    persist_directory=persist_directory,
                                    collection_metadata={"hnsw:space":distance_metric} 
                                  )

    def add_documents(self, documents, batch_size=500):
        total = len(documents)

        print(f"Adding {total} documents in batches of {batch_size}...")

        for i in range(0, total, batch_size):

            batch = documents[i:i + batch_size]

            self.vector_store.add_documents(batch)

            print(f"Inserted batch {i} to {min(i+batch_size, total)}")
            


    def as_retriever(self, search_type="similarity", search_kwargs=None):
        return self.vector_store.as_retriever(
                                                search_type=search_type,
                                                search_kwargs=search_kwargs or {"k": 5}
                                             )