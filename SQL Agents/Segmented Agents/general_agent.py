from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from shared import memory

class GeneralAgent:
    def __init__(self):
        self.openai = ChatOpenAI(model_name="gpt-4o-mini")

        self.memory = memory  # Memória compartilhada


    def respond_to_general_query(self, query):


        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente assistente geral que responde a perguntas não específicas sobre dados."),
            SystemMessage(content=f"Contexto anterior:\n{memory}"),
            HumanMessage(content=f"{query}")
        ])
        messages = chat_template.format_messages()
        response = self.openai.invoke(messages)
    
        self.memory.chat_memory.messages.append(AIMessage(content=f"GeneralAgent: {response.content}"))

        print("[DEBUG] Histórico:", self.memory)

        return response.content
    
    def invoke(self, query):
        return self.respond_to_general_query(query)
