import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")) 
)

from multi_user_rag.chains.conversation_chain import ConversationChain

class InferenceService:
    def __init__(self):
        self.conversation_service=ConversationChain()
        self.chains={}

        
    def invoke_response(self, user_id, prompt):

        if user_id not in self.chains:
            self.chains[user_id]=self.conversation_service.build_chain(user_id)

        chain = self.chains[user_id]
        
        response=chain.invoke(
                                {"input":prompt},
                                config={
                                        "configurable":{"session_id":user_id} 
                                    }
                            )
        return  {
                    "answer":response["answer"],
                    "context": response["context"]
                }
def main():
    inference_service = InferenceService()

    user_id = input("User_Id: ")
    print("\nConversational RAG System (type 'exit' to quit)\n")

    while True:
        prompt = input("Prompt: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        result = inference_service.invoke_response(user_id, prompt)

        print("\nAssistant:", result["answer"])
        print("-" * 50)
    
if __name__ == "__main__":
    main()