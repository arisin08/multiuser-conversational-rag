import os
from langchain_core.documents import Document


def load_user_documents(user_id: str, directory: str):

    user_dir = os.path.join(directory, user_id)

    documents = []

    if not os.path.exists(user_dir):
        return documents

    for filename in os.listdir(user_dir):

        filepath = os.path.join(user_dir, filename)

        if filename.endswith(".txt"):

            with open(filepath, "r", encoding="utf-8") as f:

                text = f.read()

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "user",
                            "user_id": user_id,
                            "filename": filename
                        }
                    )
                )

    return documents
