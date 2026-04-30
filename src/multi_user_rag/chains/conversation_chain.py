from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from multi_user_rag.chains.rag_chain import RAGChainService
from multi_user_rag.config import (GPT_MODEL,TEMPERATURE)

class ConversationChain:
    def __init__(self):
        self.rag_service= RAGChainService()
        

    #used to retrieve conversation history based on specific user id
    def get_session_history_db(self,session_id):
        history=SQLChatMessageHistory(
                                        session_id=session_id,
                                        connection="sqlite:///memory.db"
                                     )
        return history


    def build_chain(self, user_id):
        qa_rag_chain=self.rag_service.conversational_rag_chain(user_id)
     
        #conversation rag chain
        return RunnableWithMessageHistory(
                                            qa_rag_chain,
                                            self.get_session_history_db,
                                            input_messages_key='input',
                                            history_messages_key="chat_history",
                                            output_messages_key="answer"
        ) 
