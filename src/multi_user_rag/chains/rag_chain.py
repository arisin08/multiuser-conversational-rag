from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from multi_user_rag.retrieval.retriever import RetrieverService
from multi_user_rag.config import (GPT_MODEL, TEMPERATURE)


class RAGChainService:
    def __init__(self):
        self.gpt=ChatOpenAI(model=GPT_MODEL, temperature=TEMPERATURE)

        #Rephraser
        self.rephrase_system_prompt ="""Given a chat history and the latest user question which might refrence context in the chat history, 
                                        formulate a standalone question which can be understood without the chat history.
                                        Do Not Answer the question just reformulate it if needed and otherwise return it as it is.
                                     """

        self.rephrase_prompt=ChatPromptTemplate.from_messages(
                                                                [
                                                                    ("system" , self.rephrase_system_prompt),
                                                                    MessagesPlaceholder("chat_history"),
                                                                    ("human","{input}")
                                                                ]
                                                             )
        #Multi user rag response generator
        self.qa_system_input= """You are an assistant for question-answering tasks.
                                Use the following pieces of retrieved context to answer the question.
                                If the answer is not present in the context, just say that you don't know,
                                Keep answer to the point.
                                
                                Context:
                                {context}
                            
                              """

        self.qa_prompt= ChatPromptTemplate.from_messages(
                                                    [
                                                        ("system", self.qa_system_input),
                                                        MessagesPlaceholder("chat_history"),
                                                        ("human", """Question:
                                                                        {input}
                                                                            
                                                                        Answer:
                                                        """
                                                        )

                                                    ]

                                                )                                                            
        

    #subset historical conversation based on last k conversation messages
    def memory_buffer_window(self, messages, lastk_conversations=10):
        return messages[-(lastk_conversations*2):] # since each conversation is 2 messages


    def history_aware_retriever(self, user_id):
        retriever_service=RetrieverService()
        similarity_retriever=retriever_service.get_retriever(user_id=user_id)         
        return create_history_aware_retriever(self.gpt, similarity_retriever, self.rephrase_prompt)
    

    #custom RAG chain 
    def conversational_rag_chain(self, user_id):

        question_answer_chain = (
                                RunnablePassthrough.assign(chat_history=lambda x : self.memory_buffer_window(x["chat_history"]))
                                |
                                self.qa_prompt
                                |
                                self.gpt
                                |
                                StrOutputParser()
        )                      

        retirever=self.history_aware_retriever(user_id)
        qa_rag_chain=create_retrieval_chain(retirever, question_answer_chain)

        return qa_rag_chain



    

